"""Utility to extract CIMP5 data from GEE."""
from dateutil.relativedelta import relativedelta
import requests
import argparse
import datetime
import concurrent
import glob
import logging
import sys
import os
import pickle
import time
import threading

import ee
import geemap
import geopandas


logging.basicConfig(
    level=logging.WARNING,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(levelname)s %(name)s'
        ' [%(funcName)s:%(lineno)d] %(message)s'),
    stream=sys.stdout)
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)

CACHE_DIR = '_cimp5_cache_dir'
DATASET_ID = 'NASA/NEX-GDDP'
DATASET_CRS = 'EPSG:4326'
SCENARIO_ID = 'rcp45'
PERCENTILE_TO_QUERY = 90
DATASET_SCALE = 27830
SCENARIO_LIST = ['historical', 'rcp45', 'rcp85']
BAND_NAMES = ['tasmin', 'tasmax', 'pr']
MODEL_LIST = [
    'ACCESS1-0',
    'bcc-csm1-1',
    'BNU-ESM',
    'CanESM2',
    'CCSM4',
    'CESM1-BGC',
    'CNRM-CM5',
    'CSIRO-Mk3-6-0',
    'GFDL-CM3',
    'GFDL-ESM2G',
    'GFDL-ESM2M',
    'inmcm4',
    'IPSL-CM5A-LR',
    'IPSL-CM5A-MR',
    'MIROC-ESM',
    'MIROC-ESM-CHEM',
    'MIROC5',
    'MPI-ESM-LR',
    'MPI-ESM-MR',
    'MRI-CGCM3',
    'NorESM1-M']
MODELS_BY_DATE_CACHEFILE = '_cimp5_models_by_date.dat'
QUOTA_READS_PER_MINUTE = 3000


def _throttle_query():
    """Sleep to avoid exceeding quota request."""
    time.sleep(1/QUOTA_READS_PER_MINUTE*60)


def main():
    """Entry point."""
    parser = argparse.ArgumentParser(description=(
        'Extract CIMP5 data from GEE given an AOI and date range. Produces '
        'a CSV table with the pattern `CIMP5_{unique_id}.csv` with monthly  '
        'means for precipitation and temperature broken down by model.'))
    parser.add_argument(
        'aoi_vector_path', help='Path to vector/shapefile of area of interest')
    parser.add_argument('--aggregate_by_field', help=(
        'If provided, this aggregates results by the unique values found in '
        'the field in `aoi_vector_path`'))
    parser.add_argument('start_date', type=str, help='start date YYYY-MM-DD')
    parser.add_argument('end_date', type=str, help='end date YYYY-MM-DD')
    parser.add_argument(
        '--authenticate', action='store_true',
        help='Pass this flag if you need to reauthenticate with GEE')
    args = parser.parse_args()
    aoi_vector = geopandas.read_file(args.aoi_vector_path)
    unique_id_set = [(slice(-1), None)]  # default to all features
    if args.aggregate_by_field:
        if args.aggregate_by_field not in aoi_vector:
            raise ValueError(
                f'`{args.aggregate_by_field}` was passed in as a query field '
                f'but was not found in `{args.aoi_vector_path}`, but instead '
                f'these fields are present: {", ".join(aoi_vector.columns)}')
        unique_id_set = [
            (aoi_vector[args.aggregate_by_field] == unique_id, unique_id)
            for unique_id in set(aoi_vector[args.aggregate_by_field])]

    start_day = datetime.datetime.strptime(args.start_date, '%Y-%m-%d')
    end_day = datetime.datetime.strptime(args.end_date, '%Y-%m-%d')

    current_month = start_day.month
    current_year = start_day.year

    year_month_date_lookup = {}

    while current_year < end_day.year or (
            current_year == end_day.year and current_month <= end_day.month):
        current_month_start = datetime.datetime(current_year, current_month, 1)
        next_month_start = current_month_start + relativedelta(months=1)
        current_month_end = min(next_month_start - relativedelta(days=1), end_day)
        month_str = current_month_start.strftime('%Y-%m')
        current_month_date_list = [
            (current_month_start +
             datetime.timedelta(days=delta_day)).strftime('%Y-%m-%d')
            for delta_day in range(
                (current_month_end-current_month_start).days+1)]

        year_month_date_lookup[month_str] = current_month_date_list

        if current_month == 12:
            current_month = 1
            current_year += 1
        else:
            current_month += 1

    all_dates = [
        date
        for date_list in year_month_date_lookup.values()
        for date in date_list]

    if args.authenticate:
        ee.Authenticate()
        return
    ee.Initialize()

    vector_basename = os.path.basename(os.path.splitext(
        args.aoi_vector_path)[0])
    base_dataset = ee.ImageCollection(DATASET_ID).select('pr')

    result_pattern_by_unique_id = dict()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for unique_id_index, unique_id_value in unique_id_set:
            # create tag that's either the vector basename, or if filtering on
            # a field is requested, the basename, field, and field value
            unique_id = (
                vector_basename if args.aggregate_by_field is None else
                f'{vector_basename}_{args.aggregate_by_field}_{unique_id_value}')
            result_by_date_pattern = os.path.join(
                CACHE_DIR, unique_id, f'_{unique_id}_*.dat')
            result_pattern_by_unique_id[unique_id] = result_by_date_pattern

            # save to shapefile and load into EE vector
            if unique_id_value is not None:
                filtered_aoi = aoi_vector[unique_id_index]
            else:
                filtered_aoi = aoi_vector
            local_shapefile_path = f'_local_ok_to_delete_{unique_id}.json'
            filtered_aoi = filtered_aoi.to_crs('EPSG:4326')
            filtered_aoi.to_file(local_shapefile_path)
            filtered_aoi = None
            ee_poly = geemap.geojson_to_ee(local_shapefile_path)
            os.remove(local_shapefile_path)

            for date in all_dates:
                date_dataset = base_dataset.filter(ee.Filter.date(date))
                scenario_dataset = date_dataset.filter(
                        ee.Filter.eq('scenario', SCENARIO_ID))
                clipped_dataset = scenario_dataset.filterBounds(
                    ee_poly)

                models = ee.List(clipped_dataset.aggregate_array('model')).distinct();
                #print(f'models: {models.getInfo()}')

                def reduce_by_model(model):
                    model_images = clipped_dataset.filterMetadata(
                        'model', 'equals', model)
                    summed_image = model_images.reduce(ee.Reducer.sum())
                    summed_image = summed_image.set('model', model)

                    return summed_image
                monthly_images = ee.ImageCollection(
                    models.map(reduce_by_model))
                percentile_image = monthly_images.reduce(
                    ee.Reducer.percentile([90]))
                mm_image = percentile_image.multiply(86400)

                # Reduce the ImageCollection to the desired percentile
                #print(ee_poly.geometry().bounds())

                # Generate the download URL

                download_image(mm_image, ee_poly.geometry().bounds(), 'output.tif')
                break

        LOGGER.info('done!')


def download_image(image, bounds, target_path):
    if not os.path.exists(target_path):
        url = image.getDownloadUrl({
            'scale': 27830,
            'region': bounds,
            'format': 'GEO_TIFF'
        })
        LOGGER.debug(f'saving {target_path}')
        response = requests.get(url)
        with open(target_path, 'wb') as fd:
            fd.write(response.content)
    else:
        LOGGER.info(f'{target_path} already exists, not overwriting')

if __name__ == '__main__':
    main()

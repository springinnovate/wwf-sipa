"""Utility to extract CIMP5 data from GEE."""
from osgeo import gdal
from shapely.geometry import Polygon
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
SCENARIO_ID = 'rcp85'
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
    else:
        unique_id = os.path.basename(os.path.splitext(args.aoi_vector_path)[0])

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
        year_month_date_lookup[month_str] = (
            current_month_start.strftime('%Y-%m-%d'),
            current_month_end.strftime('%Y-%m-%d')
            )

        if current_month == 12:
            current_month = 1
            current_year += 1
        else:
            current_month += 1

    if args.authenticate:
        ee.Authenticate()
        return
    ee.Initialize()

    vector_basename = os.path.basename(os.path.splitext(
        args.aoi_vector_path)[0])
    base_dataset = ee.ImageCollection(DATASET_ID).select('pr')

    for percentile in [10, 90]:
        for unique_id_index, unique_id_value in unique_id_set:
            # create tag that's either the vector basename, or if filtering on
            # a field is requested, the basename, field, and field value
            unique_id = (
                vector_basename if args.aggregate_by_field is None else
                f'{vector_basename}_{args.aggregate_by_field}_{unique_id_value}')

            # save to shapefile and load into EE vector
            if unique_id_value is not None:
                filtered_aoi = aoi_vector[unique_id_index]
            else:
                filtered_aoi = aoi_vector
            local_shapefile_path = f'_local_ok_to_delete_{unique_id}.json'
            filtered_aoi = filtered_aoi.to_crs('EPSG:4326')

            # Calculate bounding box
            bbox = filtered_aoi.total_bounds

            # Create a Polygon from the bounding box
            bbox_poly = Polygon([(bbox[0], bbox[1]),
                                 (bbox[2], bbox[1]),
                                 (bbox[2], bbox[3]),
                                 (bbox[0], bbox[3])])

            # Create a new GeoDataFrame for the bounding box
            bbox_gdf = geopandas.GeoDataFrame(
                {'geometry': [bbox_poly]}, crs=filtered_aoi.crs)

            bbox_gdf.to_file(local_shapefile_path)
            bbox_gdf = None
            ee_poly = geemap.geojson_to_ee(local_shapefile_path)
            os.remove(local_shapefile_path)

            for year_month_str, (start_date, end_date) in \
                    year_month_date_lookup.items():
                date_dataset = base_dataset.filter(ee.Filter.date(
                    start_date, end_date))
                scenario_dataset = date_dataset.filter(
                        ee.Filter.eq('scenario', SCENARIO_ID))
                clipped_dataset = scenario_dataset.filterBounds(
                    ee_poly)

                models = ee.List(
                    clipped_dataset.aggregate_array('model')).distinct()


                def createMask(precip_image):
                    precip_image = precip_image.multiply(86400)
                    mask = precip_image.gt(1)
                    return mask

                def reduce_by_sum_per_model(model):
                    model_images = clipped_dataset.filterMetadata(
                        'model', 'equals', model)
                    summed_image = model_images.reduce(ee.Reducer.sum())
                    summed_image = summed_image.set('model', model)
                    return summed_image

                def reduce_by_count_gt_per_model(model):
                    model_images = clipped_dataset.filterMetadata(
                        'model', 'equals', model)
                    mask_images = model_images.map(createMask)
                    summed_image = mask_images.reduce(ee.Reducer.sum())
                    summed_image = summed_image.set('model', model)
                    return summed_image

                monthly_precip_images = ee.ImageCollection(
                    models.map(reduce_by_sum_per_model))
                percentile_image = monthly_precip_images.reduce(
                    ee.Reducer.percentile([percentile]))
                mm_image = percentile_image.multiply(86400)
                target_path = os.path.join(
                    f'precip_{unique_id}_{2050}_{SCENARIO_ID}_{percentile}',
                    f'precip_{SCENARIO_ID}_{percentile}_{year_month_str.replace("-", "_")}.tif')
                print(f'fetching {target_path}')
                download_image(
                    mm_image, ee_poly.geometry().bounds(),
                    target_path)


                monthly_event_images = ee.ImageCollection(
                    models.map(reduce_by_count_gt_per_model))
                percentile_image = monthly_event_images.reduce(
                    ee.Reducer.percentile([percentile]))
                target_path = os.path.join(
                    f'n_events_{unique_id}_{2050}_{SCENARIO_ID}_{percentile}',
                    f'n_events_{SCENARIO_ID}_{percentile}_{year_month_str.replace("-", "_")}.tif')
                print(f'fetching {target_path}')
                download_image(
                    percentile_image, ee_poly.geometry().bounds(),
                    target_path)

        LOGGER.info('done!')


def download_image(image, bounds, target_path):
    try:
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
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
            r = gdal.OpenEx(target_path, gdal.OF_RASTER | gdal.GA_Update)
            b = r.GetRasterBand(1)
            nodata = b.GetNoDataValue()
            if nodata is not None:
                # make ndoata 0
                array = b.ReadAsArray()
                array[array == nodata] = 0
                b.WriteArray(array)
                b.SetNoDataValue(-1)
        else:
            LOGGER.info(f'{target_path} already exists, not overwriting')

    except ee.ee_exception.EEException:
        LOGGER.exception('cannot download with url, downloading to drive')
        export_params = {
            'image': image,
            'scale': 27830,
            'region': bounds,
            'folder': 'pull_cmip5',
            'fileFormat': 'GeoTIFF',
            'maxPixels': 1e12  # this is to avoid the 'computed value too large' error
        }
        task = ee.batch.Export.image.toDrive(**export_params)
        task.start()


if __name__ == '__main__':
    main()

# E = sum(e_r * P_r, r in 365?)
# e_r = 0.29*(1-0.72*exp(-0.082 * i_r))

# P_r rainfall amount
# i_r rainfall intensity

# e_{r} = 0.29\left[ {1 - 0.72exp\left( { - 0.082i_{r} } \right)} \right]
# E = \left( {\mathop \sum \limits_{r = 1}^{n} \left( {e_{r} \cdot P_{r} } \right)} \right)


# E = \left( {\mathop \sum \limits_{r = 1}^{n} \left( {e_{r} \cdot P_{r} } \right)} \right)
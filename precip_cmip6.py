"""Precip calculator based on CMIP6 NEXX dataset."""
import argparse
import functools
import logging
import os
import pickle
import sys

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


VALID_MODEL_LIST = (
    'ACCESS-ESM1-5',
    'BCC-CSM2-MR',
    'CanESM5',
    'CESM2',
    'CMCC-ESM2',
    'CNRM-ESM2-1',
    'EC-Earth3-Veg-LR',
    'FGOALS-g3',
    'GFDL-ESM4',
    'GISS-E2-1-G',
    'HadGEM3-GC31-MM',
    'IITM-ESM',
    'INM-CM5-0',
    'IPSL-CM6A-LR',
    'KACE-1-0-G',
    'KIOST-ESM',
    'MIROC-ES2L',
    'MPI-ESM1-2-HR',
    'MRI-ESM2-0',
    'NESM3',
    'NorESM2-MM',
    'TaiESM1',
    'UKESM1-0-LL',
)

DATASET_ID = 'NASA/GDDP-CMIP6'
DATASET_CRS = 'EPSG:4326'
DATASET_SCALE = 27830


def auto_memoize(func):
    cache_file = f"{func.__name__}_cache.pkl"

    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            cache = pickle.load(f)
    else:
        cache = {}

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        key = (args, frozenset(kwargs.items()))
        if key in cache:
            return cache[key]
        else:
            result = func(*args, **kwargs)
            cache[key] = result
            with open(cache_file, 'wb') as f:
                pickle.dump(cache, f)
            return result

    return wrapper


@auto_memoize
def get_valid_model_list(model_list, start_year, end_year, scenario_id):
    # Initialize the ImageCollection with your filters
    cmip6_dataset = ee.ImageCollection(DATASET_ID).select('pr').filter(
        ee.Filter.And(
            ee.Filter.inList('model', model_list),
            ee.Filter.eq('scenario', scenario_id),
            ee.Filter.calendarRange(start_year, end_year, 'year'))
    )

    # Aggregate model IDs
    unique_models = cmip6_dataset.aggregate_array('model').distinct()

    # Bring the list to Python
    unique_models_list = unique_models.getInfo()

    # Print or otherwise use the list
    print("Unique model IDs:", unique_models_list)
    return tuple(unique_models_list)


def authenticate():
    try:
        ee.Initialize()
        return
    except Exception:
        pass

    try:
        ee.Authenticate()
        ee.Initialize()
        return
    except Exception:
        pass

    try:
        gee_key_path = os.environ['GEE_KEY_PATH']
        credentials = ee.ServiceAccountCredentials(None, gee_key_path)
        ee.Initialize(credentials)
        return
    except Exception:
        pass

    ee.Initialize()


def main():
    parser = argparse.ArgumentParser(description=(
        'Fetch CMIP6 based erosivity given a year date range.'))
    parser.add_argument(
        '--aoi_vector_path', help='Path to vector/shapefile of area of interest')
    parser.add_argument('--where_statement', help=(
        'If provided, allows filtering by a field id and value of the form '
        'field_id=field_value'))
    parser.add_argument(
        '--scenario_id', help="Scenario ID ssp245, ssp585, historical")
    parser.add_argument('--date_range', nargs=2, type=str, help=(
        'Two date ranges in YYYY format to download between.'))
    parser.add_argument(
        '--status', action='store_true', help='To check task status')
    parser.add_argument(
        '--dataset_scale', type=float, default=DATASET_SCALE, help=(
            f'Override the base scale of {DATASET_SCALE}m to '
            f'whatever you desire.'))
    parser.add_argument(
        '--percentile', nargs='+', help='List of percentiles')
    args = parser.parse_args()
    authenticate()

    if args.status:
        # Loop through each task to print its status
        for task in ee.batch.Task.list():
            print(task)
            print("-----")
        return

    aoi_vector = geopandas.read_file(args.aoi_vector_path)
    if args.where_statement:
        field_id, field_val = args.where_statement.split('=')
        print(aoi_vector.size)
        aoi_vector = aoi_vector[aoi_vector[field_id] == field_val]
        print(aoi_vector)
        print(aoi_vector.size)

    local_shapefile_path = '_local_cmip6_aoi_ok_to_delete.json'
    aoi_vector = aoi_vector.to_crs('EPSG:4326')
    aoi_vector.to_file(local_shapefile_path)
    aoi_vector = None
    ee_poly = geemap.geojson_to_ee(local_shapefile_path)

    # Filter models dynamically based on data availability
    start_year = int(args.date_range[0])
    end_year = int(args.date_range[1])

    for target_month in list(range(1, 13)) + ['annual']:
        model_list = get_valid_model_list(
            VALID_MODEL_LIST, start_year, end_year, args.scenario_id)

        cmip6_dataset = ee.ImageCollection(DATASET_ID).select('pr').filter(
            ee.Filter.And(
                ee.Filter.inList('model', model_list),
                ee.Filter.eq('scenario', args.scenario_id),
                ee.Filter.calendarRange(start_year, end_year, 'year')))

        region_basename = os.path.splitext(
            os.path.basename(args.aoi_vector_path))[0]
        if target_month == 'annual':
            timeframe_str = target_month
        else:
            timeframe_str = f'{target_month:02d}'
        description = (
            f'precip_{region_basename}_{args.scenario_id}_'
            f'{start_year}_{end_year}_{timeframe_str}')

        def calculate_precip(model_name):
            model_data = cmip6_dataset.filter(
                ee.Filter.eq('model', model_name))
            yearly_collection = model_data.filter(
                ee.Filter.calendarRange(start_year, end_year, 'year'))
            if target_month != 'annual':
                yearly_collection = model_data.filter(
                    ee.Filter.calendarRange(
                        target_month, target_month, 'month'))
            total_precip = yearly_collection.reduce(ee.Reducer.sum())
            # convert to mm
            annual_precip = total_precip.multiply(
                86400/(end_year-start_year+1))
            return annual_precip.rename(model_name)

        # Calculate metrics for all models
        precip_by_model_list = [
            calculate_precip(model) for model in model_list]
        precip_by_model = precip_by_model_list[0]
        precip_by_model = precip_by_model.addBands(
            precip_by_model_list[1:])
        precip_image_clipped = precip_by_model.clip(ee_poly)
        folder_id = 'gee_output'

        for percentile in args.percentile:
            local_description = f'{description}_p{percentile}'
            ee.batch.Export.image.toDrive(
                image=precip_image_clipped.reduce(
                    ee.Reducer.percentile([float(percentile)])),
                description=local_description,
                folder=folder_id,
                scale=args.dataset_scale,
                crs=DATASET_CRS,
                region=ee_poly.geometry().bounds(),
                fileFormat='GeoTIFF',
                ).start()

            print(
                f'downloading precip raster to google drive: '
                f'{folder_id}/{local_description}')


if __name__ == '__main__':
    main()

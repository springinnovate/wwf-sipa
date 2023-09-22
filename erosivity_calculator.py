"""
Erosivity calculator based on
Hydrol. Earth Syst. Sci., 19, 4113–4126, 2015
www.hydrol-earth-syst-sci.net/19/4113/2015/
doi:10.5194/hess-19-4113-2015
© Author(s) 2015. CC Attribution 3.0 License.

Using annual rainfall erosivity calculation fo R = 1.2718*P**1.1801
"""
import io
import argparse
import os
import sys
import zipfile
import threading
from concurrent.futures import ThreadPoolExecutor
import logging

from ecoshard import geoprocessing
from osgeo import gdal
import ee
import geemap
import geopandas
import requests


logging.basicConfig(
    level=logging.WARNING,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(levelname)s %(name)s'
        ' [%(funcName)s:%(lineno)d] %(message)s'),
    stream=sys.stdout)
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)


VALID_MODEL_LIST = [
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
]
DATASET_ID = 'NASA/GDDP-CMIP6'
DATASET_CRS = 'EPSG:4326'
DATASET_SCALE = 27830


def check_dataset_collection(model, dataset_id, band_id, start_year, end_year):
    def band_checker(image):
        available_bands = image.bandNames()
        return ee.Image(ee.Algorithms.If(
            available_bands.contains(band_id), image,
            ee.Image.constant(0).rename('dummy_band')))

    try:
        collection = (
            ee.ImageCollection(dataset_id)
            .filter(ee.Filter.eq('model', model))
            .filter(ee.Filter.calendarRange(start_year, end_year, 'year'))
            .map(band_checker))

        # Filter out images with the 'dummy_band'
        collection = collection.filterMetadata(
            'system:band_names', 'not_equals', ['dummy_band'])

        size = collection.size().getInfo()
        print(f'{model} has {size} elements with the band(s) {band_id}')
        return size > 0

    except Exception as e:
        print(f"An error occurred: {e}")
        return False


def save_to_drive(
        image, folder, description, scale, ee_poly, clip_poly_path,
        target_raster_path):
    ee.batch.Export.image.toDrive({
        'image': image,
        'description': description,
        'folder': folder,
        'scale': scale,
        'crs': 'EPSG:4326',
        'region': ee_poly.geometry().bounds(),
        'fileFormat': 'GeoTIFF',
        })
    # url = image.toDrive({
    #     'scale': scale,
    #     'crs': 'EPSG:4326',
    #     'region': ee_poly.geometry().bounds(),
    #     'fileFormat': 'GeoTIFF',
    #     'folder': 'gee_output',
    #     'description': description,
    # })

    # response = requests.get(url)
    # if response.status_code == 200:
    #     content_type = response.headers.get('content-type')
    #     if 'application/zip' in content_type or 'application/octet-stream' in content_type:
    #         zip_buffer = io.BytesIO(response.content)
    #         target_preclip = f"pre_clip_{description}.tif"

    #         with zipfile.ZipFile(zip_buffer) as zip_ref:
    #             zip_ref.extractall(f"{description}")
    #             # Assuming there is only one tif file in the zip
    #             for filename in zip_ref.namelist():
    #                 if filename.endswith('.tif'):
    #                     # Optionally rename the file
    #                     if os.path.exists(target_preclip):
    #                         os.remove(target_preclip)
    #                     os.rename(f"{description}/{filename}", target_preclip)
    #                     print(f"Successfully downloaded and unzipped {filename}")
    #         r = gdal.OpenEx(target_preclip, gdal.OF_RASTER)
    #         b = r.GetRasterBand(1)
    #         b.SetNoDataValue(-9999)
    #         b = None
    #         r = None
    #         raster_info = geoprocessing.get_raster_info(target_preclip)

    #         geoprocessing.warp_raster(
    #             target_preclip, raster_info['pixel_size'],
    #             target_raster_path,
    #             'near',
    #             vector_mask_options={
    #                 'mask_vector_path': clip_poly_path,
    #                 'all_touched': True,
    #                 })
    #         os.remove(target_preclip)
    #         LOGGER.info(f'saved {target_raster_path}')
    #     else:
    #         print(f"Unexpected content type: {content_type}")
    #         print(response.content.decode('utf-8'))
    # else:
    #     print(f"Failed to download {description} from {url}")


def authenticate():
    try:
        ee.Initialize()
        return
    except Exception as e:
        print(e)
        pass

    try:
        gee_key_path = os.environ['GEE_KEY_PATH']
        credentials = ee.ServiceAccountCredentials(None, gee_key_path)
        ee.Initialize(credentials)
        return
    except Exception as e:
        print(e)
        pass

    ee.Authenticate()
    ee.Initialize()


def main():
    parser = argparse.ArgumentParser(description=(
        'Fetch CMIP6 temperature and precipitation monthly normals given a '
        'year date range.'))
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

    def filter_model(model):
        return model if check_dataset_collection(
            model, DATASET_ID, 'pr', start_year, end_year) else None

    cmip6_dataset = ee.ImageCollection(DATASET_ID).filter(
        ee.Filter.And(
            ee.Filter.inList('model', VALID_MODEL_LIST),
            ee.Filter.eq('scenario', args.scenario_id))).select('pr')

    region_basename = os.path.splitext(
        os.path.basename(args.aoi_vector_path))[0]
    description = (
        f'erosivity_{region_basename}_{args.scenario_id}_'
        f'{start_year}_{end_year}')
    target_raster_path = f'{description}.tif'

    yearly_collection = cmip6_dataset.filter(
        ee.Filter.calendarRange(start_year, end_year, 'year'))

    erosivity_image = yearly_collection.reduce(
        ee.Reducer.sum()).multiply(86400/((end_year-start_year+1)*len(
            VALID_MODEL_LIST))).pow(1.1801).multiply(1.2718)

    erosivity_image_clipped = erosivity_image.clip(ee_poly)
    folder_id = 'gee_output'
    save_to_drive(
        erosivity_image_clipped,
        folder_id,
        description, DATASET_SCALE, ee_poly,
        local_shapefile_path, target_raster_path)

    print(
        f'downloaded erosivity raster to google drive: '
        f'{folder_id}/{target_raster_path}')


if __name__ == '__main__':
    main()

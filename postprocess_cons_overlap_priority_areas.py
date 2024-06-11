"""
Conservation priorities overlap analysis:
1) how much of high value ES area (top 10% map) is outside of current
   protected area (PAs) network? Then,
2) overlap the areas of top 10% outside of the PA with KBAs and answer:
    2a) How much of high value ES area would be priorities for protection or
        restoration for ES?

Desired output: maps, % values
Inputs to use:
    PH PA   D:/repositories/wwf-sipa/data/protected_areas/PH_Combined_PAs
    PH KBA  D:/repositories/wwf-sipa/data/protected_areas/PH_KBA
    IDN PA  D:/repositories/wwf-sipa/data/protected_areas/ID_Combined PAs
    IDN KBA D:/repositories/wwf-sipa/data/protected_areas/Indonesia_KBA
"""
import collections
import os
import logging
import sys
import subprocess
import shutil
import tempfile

from ecoshard import taskgraph
from ecoshard import geoprocessing
from osgeo import gdal
import numpy
from osgeo import ogr

logging.basicConfig(
    level=logging.DEBUG,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(levelname)s %(name)s'
        ' [%(pathname)s.%(funcName)s:%(lineno)d] %(message)s'),
    stream=sys.stdout)
logging.getLogger('ecoshard.taskgraph').setLevel(logging.INFO)
LOGGER = logging.getLogger(__name__)

REGIONS_TO_ANALYZE = ['PH', 'IDN']
PROTECTED_AREAS = {
    'PH': 'D:/repositories/wwf-sipa/data/protected_areas/PH_Combined_PAs',
    'IDN': 'D:/repositories/wwf-sipa/data/protected_areas/ID_Combined PAs',
}

KEY_BIODIVERSITY_AREAS = {
    'PH': 'D:/repositories/wwf-sipa/data/protected_areas/PH_KBA',
    'IDN': 'D:/repositories/wwf-sipa/data/protected_areas/Indonesia_KBA',
}


SERVICE_OVERLAP_RASTERS = {
    'PH': [
        r"D:\repositories\wwf-sipa\fig_generator_dir\overlap_rasters\overlap_combos_top_10_PH_conservation_each ecosystem service.tif",
        r"D:\repositories\wwf-sipa\fig_generator_dir\overlap_rasters\overlap_combos_top_10_PH_restoration_each ecosystem service.tif",
    ],
    'IDN': [
        r"D:\repositories\wwf-sipa\fig_generator_dir\overlap_rasters\overlap_combos_top_10_IDN_conservation_each ecosystem service.tif",
        r"D:\repositories\wwf-sipa\fig_generator_dir\overlap_rasters\overlap_combos_top_10_IDN_restoration_each ecosystem service.tif",
    ],
}
ELLIPSOID_EPSG = 6933

RESULTS_DIR = f'workspace_{os.path.basename(os.path.splitext(__file__)[0])}'
WORKING_DIR = os.path.join(RESULTS_DIR, 'working_dir')
for dir_path in [RESULTS_DIR, WORKING_DIR]:
    os.makedirs(dir_path, exist_ok=True)


def main():
    """Entry point."""
    task_graph = taskgraph.TaskGraph(RESULTS_DIR, -1)

    raster_area_per_region = {}
    for region_id in REGIONS_TO_ANALYZE:
        for service_overlap_raster_path in SERVICE_OVERLAP_RASTERS[region_id]:
            service_overlap_in_pa_path = os.path.join(
                WORKING_DIR,
                f'%s_{region_id}_in_pa%s' % os.path.splitext(os.path.basename(
                    service_overlap_raster_path)))
            pa_overlap_task = task_graph.add_task(
                func=geoprocessing.mask_raster,
                args=(
                    (service_overlap_raster_path, 1),
                    PROTECTED_AREAS[region_id],
                    service_overlap_in_pa_path),
                kwargs={
                    'working_dir': WORKING_DIR,
                    'all_touched': True,
                    'allow_different_blocksize': True},
                target_path_list=[service_overlap_in_pa_path],
                task_name=f'pa overlap for {region_id}')

            service_overlap_in_kba_path = os.path.join(
                WORKING_DIR,
                f'%s_{region_id}_in_kba%s' % os.path.splitext(os.path.basename(
                    service_overlap_raster_path)))
            kba_overlap_task = task_graph.add_task(
                func=geoprocessing.mask_raster,
                args=(
                    (service_overlap_raster_path, 1),
                    KEY_BIODIVERSITY_AREAS[region_id],
                    service_overlap_in_kba_path),
                kwargs={
                    'working_dir': WORKING_DIR,
                    'all_touched': True,
                    'allow_different_blocksize': True},
                target_path_list=[service_overlap_in_kba_path],
                task_name=f'kba overlap for {region_id}')

            service_overlap_in_pa_excluding_kba_path = os.path.join(
                WORKING_DIR,
                f'%s_{region_id}_in_kba_excluding_pa%s' % os.path.splitext(
                    os.path.basename(service_overlap_raster_path)))
            kba_excluding_pa_overlap_task = task_graph.add_task(
                func=exclude_by_raster,
                args=(
                    service_overlap_in_kba_path,
                    service_overlap_in_pa_path,
                    service_overlap_in_pa_excluding_kba_path),
                target_path_list=[service_overlap_in_pa_excluding_kba_path],
                dependent_task_list=[kba_overlap_task, pa_overlap_task],
                task_name=f'exclude PA from KBA for {region_id}')

            raster_area = {}
            for key, raster_path, dependent_task_list in [
                    ('service', service_overlap_raster_path, []),
                    ('service_in_pa', service_overlap_in_pa_path, [pa_overlap_task]),
                    ('service_in_kpa', service_overlap_in_kba_path, [kba_overlap_task]),
                    ('service_in_kpa_excluding_pa', service_overlap_in_pa_excluding_kba_path, [kba_excluding_pa_overlap_task])]:
                raster_area[key] = task_graph.add_task(
                    func=calculate_pixel_area_km2,
                    args=(raster_path, ELLIPSOID_EPSG),
                    dependent_task_list=dependent_task_list,
                    store_result=True,
                    task_name=f'count valid in {region_id} {key}')
            service_overlap_basename = os.path.basename(
                os.path.splitext(service_overlap_raster_path)[0])
            raster_area_per_region[
                f'{region_id}_{service_overlap_basename}'] = raster_area

    task_graph.join()
    task_graph.close()
    pixel_count_file = open(os.path.join(
        RESULTS_DIR, 'area_analysis_of_priority_areas_v2.csv'), 'w')
    for region_id, raster_area_ in raster_area_per_region.items():
        pixel_count_file.write(f'{region_id}\n')
        pixel_count_file.write(f'service,area in km2,% of total service\n')
        first_val = None
        for service_key, task in raster_area_.items():
            area_in_km2 = task.get()
            if first_val is None:
                first_val = area_in_km2
            pixel_count_file.write(f'{service_key},{area_in_km2:.1f},{area_in_km2/first_val*100:.2f}%\n')
    pixel_count_file.close()
    LOGGER.info(f'all done, results at {RESULTS_DIR}')


def calculate_pixel_area_km2(base_raster_path, target_epsg):
    source_raster = gdal.OpenEx(base_raster_path, gdal.OF_RASTER)
    target_srs = osr.SpatialReference()
    target_srs.ImportFromEPSG(target_epsg)

    LOGGER.debug(f'about to warp {base_raster_path} to epsg:{target_epsg}')
    reprojected_raster = gdal.Warp(
        '',
        source_raster,
        format='MEM',
        dstSRS=target_srs)
    reprojected_band = reprojected_raster.GetRasterBand(1)
    nodata = reprojected_band.GetNoDataValue()
    LOGGER.debug(f'nodata value is {nodata}')
    LOGGER.debug(f'read warped {base_raster_path}')
    reprojected_data = reprojected_band.ReadAsArray()
    reprojected_geotransform = reprojected_raster.GetGeoTransform()
    reprojected_pixel_width, reprojected_pixel_height = (
        reprojected_geotransform[1], abs(reprojected_geotransform[5]))

    # Calculate the area of pixels with values > 1
    pixel_area = reprojected_pixel_width * reprojected_pixel_height
    count = ((reprojected_data > 0) & (reprojected_data != nodata)).sum()
    total_area = count * pixel_area / 1e6  # covert to km2

    LOGGER.debug(f'total area of {base_raster_path}: {total_area}km2')
    return total_area


def exclude_by_raster(
        base_raster_path, excluder_raster_path, target_raster_path):
    """Exclude regions in the base that are defined in the excluder."""
    base_nodata = geoprocessing.get_raster_info(
        base_raster_path)['nodata'][0]
    if base_nodata is None:
        base_nodata = 0
    excluder_nodata = geoprocessing.get_raster_info(
        excluder_raster_path)['nodata'][0]

    def _op(base_array, excluder_array):
        result = base_array.copy()
        if excluder_nodata is not None:
            result[excluder_array != excluder_nodata] = base_nodata
        result[excluder_array != 0] = base_nodata
        return result

    geoprocessing.raster_calculator(
        [(base_raster_path, 1), (excluder_raster_path, 1)], _op,
        target_raster_path,
        geoprocessing.get_raster_info(
            base_raster_path)['datatype'], base_nodata,
        allow_different_blocksize=True)


if __name__ == '__main__':
    main()
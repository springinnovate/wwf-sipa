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
        "D:/repositories/wwf-sipa/post_processing_results_no_road_recharge/summed_services/10_PH_conservation_inf_service_overlap_count.tif",
        "D:/repositories/wwf-sipa/post_processing_results_no_road_recharge/summed_services/10_PH_restoration_service_overlap_count.tif",],
    'IDN': [
        "D:/repositories/wwf-sipa/post_processing_results_no_road_recharge/summed_services/10_IDN_conservation_inf_service_overlap_count.tif",
        "D:/repositories/wwf-sipa/post_processing_results_no_road_recharge/summed_services/10_IDN_restoration_service_overlap_count.tif",],
    }

RESULTS_DIR = f'workspace_{os.path.basename(os.path.splitext(__file__)[0])}'
WORKING_DIR = os.path.join(RESULTS_DIR, 'working_dir')
for dir_path in [RESULTS_DIR, WORKING_DIR]:
    os.makedirs(dir_path, exist_ok=True)


def area_of_pixel_km2(pixel_size, center_lat):
    """Calculate km^2 area of a wgs84 square pixel.

    Adapted from: https://gis.stackexchange.com/a/127327/2397

    Parameters:
        pixel_size (float): length of side of pixel in degrees.
        center_lat (float): latitude of the center of the pixel. Note this
            value +/- half the `pixel-size` must not exceed 90/-90 degrees
            latitude or an invalid area will be calculated.

    Returns:
        Area of square pixel of side length `pixel_size` centered at
        `center_lat` in km^2.

    """
    a = 6378137  # meters
    b = 6356752.3142  # meters
    e = numpy.sqrt(1 - (b/a)**2)
    area_list = []
    for f in [center_lat+pixel_size/2, center_lat-pixel_size/2]:
        zm = 1 - e*numpy.sin(numpy.radians(f))
        zp = 1 + e*numpy.sin(numpy.radians(f))
        area_list.append(
            numpy.pi * b**2 * (
                numpy.log(zp/zm) / (2*e) +
                numpy.sin(numpy.radians(f)) / (zp*zm)))
    # mult by 1e-6 to convert m^2 to km^2
    return pixel_size / 360. * (area_list[0] - area_list[1]) * 1e-6


def main():
    """Entry point."""
    task_graph = taskgraph.TaskGraph(RESULTS_DIR, os.cpu_count()//2+2, 15.0)
    #task_graph = taskgraph.TaskGraph(RESULTS_DIR, -1)

    pixel_counts_per_region = {}
    for region_id in REGIONS_TO_ANALYZE:
        for service_overlap_raster_path in SERVICE_OVERLAP_RASTERS[region_id]:
            service_overlap_in_pa_path = os.path.join(
                WORKING_DIR,
                f'%s_{region_id}_in_pa%s' % os.path.splitext(os.path.basename(
                    service_overlap_raster_path))
                )
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
                    service_overlap_raster_path))
                )
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
                    os.path.basename(service_overlap_raster_path))
                )
            kba_excluding_pa_overlap_task = task_graph.add_task(
                func=exclude_by_raster,
                args=(
                    service_overlap_in_kba_path,
                    service_overlap_in_pa_path,
                    service_overlap_in_pa_excluding_kba_path),
                target_path_list=[service_overlap_in_pa_excluding_kba_path],
                dependent_task_list=[kba_overlap_task, pa_overlap_task],
                task_name=f'exclude PA from KBA for {region_id}')

            pixel_counts = {}
            for key, raster_path, dependent_task_list in [
                    ('service', service_overlap_raster_path, []),
                    ('service_in_pa', service_overlap_in_pa_path, [pa_overlap_task]),
                    ('service_in_kpa', service_overlap_in_kba_path, [kba_overlap_task]),
                    ('service_in_kpa_excluding_pa', service_overlap_in_pa_excluding_kba_path, [kba_excluding_pa_overlap_task])]:
                pixel_counts[key] = task_graph.add_task(
                    func=count_valid_pixels,
                    args=(raster_path,),
                    dependent_task_list=dependent_task_list,
                    store_result=True,
                    task_name=f'count valid in {region_id} {key}')
            service_overlap_basename = os.path.basename(
                os.path.splitext(service_overlap_raster_path)[0])
            pixel_counts_per_region[
                f'{region_id}_{service_overlap_basename}'] = pixel_counts

    task_graph.join()
    task_graph.close()
    pixel_count_file = open(os.path.join(
        RESULTS_DIR, 'area_analysis_of_priority_areas.csv'), 'w')
    for region_id, pixel_counts in pixel_counts_per_region.items():
        pixel_count_file.write(f'{region_id}\n')
        pixel_count_file.write(f'service,area in Ha,% of total service\n')
        first_val = None
        for service_key, task in pixel_counts.items():
            _, area_in_ha = task.get()
            if first_val is None:
                first_val = area_in_ha
            pixel_count_file.write(f'{service_key},{area_in_ha:.1f},{area_in_ha/first_val*100:.2f}%\n')
    pixel_count_file.close()
    LOGGER.info(f'all done, results at {RESULTS_DIR}')


def count_valid_pixels(raster_path):
    raster_info = geoprocessing.get_raster_info(raster_path)
    nodata = raster_info['nodata'][0]
    gt = raster_info['geotransform']

    valid_count = 0
    total_area_ha = 0
    for offset_dict, array in geoprocessing.iterblocks((raster_path, 1)):
        x_coord, y_coord = gdal.ApplyGeoTransform(
            gt,
            offset_dict['xoff']+offset_dict['win_xsize']/2,
            offset_dict['yoff']+offset_dict['win_ysize']/2)
        pixel_area_in_ha = 100*area_of_pixel_km2(raster_info['pixel_size'][0], y_coord)
        local_valid_count = numpy.count_nonzero(array != nodata)
        valid_count += local_valid_count
        total_area_ha += pixel_area_in_ha*local_valid_count
    return valid_count, total_area_ha


def exclude_by_raster(
        base_raster_path, excluder_raster_path, target_raster_path):
    """Exclude regions in the base that are defined in the excluder."""
    base_nodata = geoprocessing.get_raster_info(
        base_raster_path)['nodata'][0]
    excluder_nodata = geoprocessing.get_raster_info(
        excluder_raster_path)['nodata'][0]

    def _op(base_array, excluder_array):
        result = base_array.copy()
        result[excluder_array != excluder_nodata] = base_nodata
        return result

    geoprocessing.raster_calculator(
        [(base_raster_path, 1), (excluder_raster_path, 1)], _op,
        target_raster_path,
        geoprocessing.get_raster_info(
            base_raster_path)['datatype'], base_nodata,
        allow_different_blocksize=True)


if __name__ == '__main__':
    main()
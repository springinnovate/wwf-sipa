import argparse
import logging
import sys
import tempfile
import os
import shutil
from ecoshard import geoprocessing
from ecoshard import taskgraph
import numpy
import subprocess

from osgeo import ogr
logging.basicConfig(
    level=logging.DEBUG,
    stream=sys.stdout,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(levelname)s %(name)s'
        ' [%(pathname)s.%(funcName)s:%(lineno)d] %(message)s'))
LOGGER = logging.getLogger(__name__)


aggregate_vector = './data/admin_boundaries/IDN_gdam3.gpkg'

rasters_to_process = {
    'IDN_10th_percentile_service_conservation': ('10_IDN_conservation_inf_dspop__service_overlap_count.tif', '10_IDN_conservation_inf_road__service_overlap_count.tif'),
    'IDN_10th_percentile_service_restoration': ('10_IDN_restoration_dspop__service_overlap_count.tif', '10_IDN_restoration_road__service_overlap_count.tif'),
    'IDN_10th_percentile_service_flood_mitigation_conservation_inf': ['top_10th_percentile_service_dspop_flood_mitigation_IDN_conservation_inf.tif', 'top_10th_percentile_service_road_flood_mitigation_IDN_conservation_inf.tif'],
    'IDN_10th_percentile_service_flood_mitigation_restoration': ['top_10th_percentile_service_dspop_flood_mitigation_IDN_restoration.tif', 'top_10th_percentile_service_road_flood_mitigation_IDN_restoration.tif'],
    'IDN_10th_percentile_service_recharge_conservation_inf': ['top_10th_percentile_service_dspop_recharge_IDN_conservation_inf.tif', 'top_10th_percentile_service_road_recharge_IDN_conservation_inf.tif'],
    'IDN_10th_percentile_service_recharge_restoration': ['top_10th_percentile_service_dspop_recharge_IDN_restoration.tif', 'top_10th_percentile_service_road_recharge_IDN_restoration.tif'],
    'IDN_10th_percentile_service_sediment_conservation_inf': ['top_10th_percentile_service_dspop_sediment_IDN_conservation_inf.tif', 'top_10th_percentile_service_road_sediment_IDN_conservation_inf.tif'],
    'IDN_10th_percentile_service_sediment_restoration': ['top_10th_percentile_service_dspop_sediment_IDN_restoration.tif', 'top_10th_percentile_service_road_sediment_IDN_restoration.tif'],
}


def sum_zero_to_nodata(base_raster_path_list, target_raster_path):
    raster_info = geoprocessing.get_raster_info(base_raster_path_list[0])
    global_nodata = raster_info['nodata'][0]
    if global_nodata is None:
        global_nodata = 0

    nodata_list = [
        geoprocessing.get_raster_info(path)['nodata'][0]
        for path in base_raster_path_list]

    def _op(*base_array_list):
        result = numpy.zeros(base_array_list[0].shape)
        running_valid_mask = numpy.zeros(result.shape, dtype=bool)
        for base_array, local_nodata in zip(base_array_list, nodata_list):
            valid_mask = base_array != 0
            if local_nodata is not None:
                valid_mask = valid_mask & (base_array != local_nodata)
            result[valid_mask] += base_array[valid_mask]
            running_valid_mask |= valid_mask
        result[result == 0] = global_nodata
        return result

    pre_cog_target = f'pre_cog_{os.path.basename(target_raster_path)}'
    geoprocessing.raster_calculator(
        [(path, 1) for path in base_raster_path_list], _op, pre_cog_target,
        raster_info['datatype'], global_nodata,
        allow_different_blocksize=True)

    subprocess.check_call(
        f'gdal_translate {pre_cog_target} {target_raster_path} -of COG -co BIGTIFF=YES')
    os.remove(pre_cog_target)


def zonal_stats():
    """Do zonal stats over base raster in each polygon of the vector."""
    task_graph = taskgraph.TaskGraph(os.path.dirname(__file__), len(rasters_to_process), 10.0)

    result_dir = 'combined_result'
    os.makedirs(result_dir, exist_ok=True)
    for key, raster_path_list in rasters_to_process.items():
        LOGGER.info(f'processing {key}')
        target_path = os.path.join(result_dir, f'{key}.tif')
        task_graph.add_task(
            func=sum_zero_to_nodata,
            args=(raster_path_list, target_path),
            target_path_list=[target_path],
            task_name=f'sum for {key}')
    task_graph.join()
    task_graph.close()


if __name__ == '__main__':
    zonal_stats()
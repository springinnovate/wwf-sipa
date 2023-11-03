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

logging.basicConfig(
    level=logging.DEBUG,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(levelname)s %(name)s'
        ' [%(pathname)s.%(funcName)s:%(lineno)d] %(message)s'),
    stream=sys.stdout)
logging.getLogger('ecoshard.taskgraph').setLevel(logging.INFO)
LOGGER = logging.getLogger(__name__)


def join_mask(mask_a_path, mask_b_path, joined_mask_path):
    """Intersect and a b into joined."""
    a_nodata = geoprocessing.get_raster_info(mask_a_path)['nodata'][0]
    b_nodata = geoprocessing.get_raster_info(mask_a_path)['nodata'][0]
    target_nodata = 2

    def mask_op(array_a, array_b):
        valid_mask = numpy.ones(array_a.shape, dtype=bool)
        result = numpy.full(array_a.shape, target_nodata, dtype=numpy.byte)
        if a_nodata is not None:
            valid_mask &= array_a != a_nodata
        if b_nodata is not None:
            valid_mask &= array_b != b_nodata
        result[valid_mask] = (
            (array_a[valid_mask] == 1) & (array_b[valid_mask] == 1))
        return result

    aligned_dir = tempfile.mkdtemp(
        prefix='join_mask_', dir=os.path.dirname(joined_mask_path))
    aligned_target_raster_path_list = [
        os.path.join(aligned_dir, f'align_{os.path.basename(path)}')
        for path in [mask_a_path, mask_b_path]]

    pixel_size = geoprocessing.get_raster_info(
        mask_a_path)['pixel_size']
    geoprocessing.align_and_resize_raster_stack(
        [mask_a_path, mask_b_path], aligned_target_raster_path_list,
        ['near']*2, pixel_size, 'intersection')

    geoprocessing.raster_calculator(
        [(path, 1) for path in aligned_target_raster_path_list], mask_op,
        joined_mask_path, gdal.GDT_Byte, target_nodata)
    shutil.rmtree(aligned_dir)


def zonal_stats(raster_path, vector_path, table_path):
    """Do zonal stats over base raster in each polygon of the vector."""
    working_dir = tempfile.mkdtemp(
        prefix='zonal_stats_', dir=os.path.dirname(table_path))
    LOGGER.info(f'processing {raster_path}')
    stat_dict = geoprocessing.zonal_statistics(
        (raster_path, 1), vector_path,
        working_dir=working_dir,
        clean_working_dir=True,
        polygons_might_overlap=False)
    stat_list = ['count', 'max', 'min', 'nodata_count', 'sum']
    LOGGER.info(f'*********** building table at {table_path}')
    with open(table_path, 'w') as table_file:
        table_file.write(f'{raster_path}\n{vector_path}\n')
        table_file.write('fid,')
        table_file.write(f'{",".join(stat_list)},mean\n')
        for fid, stats in stat_dict.items():
            table_file.write(f'{fid},')
            for stat_id in stat_list:
                table_file.write(f'{stats[stat_id]},')
            if stats['count'] > 0:
                table_file.write(f'{stats["sum"]/stats["count"]}')
            else:
                table_file.write('NaN')
            table_file.write('\n')
    shutil.rmtree(working_dir)
    LOGGER.info(f'all done, table at {table_path}')


def add_rasters(raster_path_list, target_raster_path, target_datatype):
    """Add all rasters together, do not consider nodata issues."""
    working_dir = tempfile.mkdtemp(
        prefix='_ok_to_delete_add_rasters_', dir=os.path.dirname(target_raster_path))
    aligned_target_raster_path_list = [
        os.path.join(working_dir, f'align_{os.path.basename(path)}')
        for path in raster_path_list]
    pixel_size = geoprocessing.get_raster_info(
        raster_path_list[0])['pixel_size']
    geoprocessing.align_and_resize_raster_stack(
        raster_path_list, aligned_target_raster_path_list, ['near']*len(raster_path_list),
        pixel_size, 'intersection')

    def _sum_op(*array_list):
        return numpy.sum(array_list, axis=0)

    pre_cog_target = os.path.join(working_dir, os.path.basename(target_raster_path))
    geoprocessing.raster_calculator(
        [(path, 1) for path in aligned_target_raster_path_list], _sum_op, pre_cog_target,
        target_datatype, None)
    subprocess.check_call(
        f'gdal_translate {pre_cog_target} {target_raster_path} -of COG -co BIGTIFF=YES')
    shutil.rmtree(working_dir)


def make_top_nth_percentile_masks(
        base_raster_path, top_percentile_list, target_raster_path_pattern):
    """Mask base by mask such that any nodata in mask is set to nodata in base."""
    ordered_top_percentile_list = list(sorted(top_percentile_list, reverse=True))
    # need to convert this to "gte" format so if top 10th percent, we get 90th percentile
    raw_percentile_list = [100-float(x) for x in ordered_top_percentile_list]
    working_dir = tempfile.mkdtemp(
        prefix='percentile_sort_', dir=os.path.dirname(target_raster_path_pattern))
    os.makedirs(working_dir, exist_ok=True)
    percentile_values = geoprocessing.raster_band_percentile(
        (base_raster_path, 1), working_dir,
        raw_percentile_list,
        heap_buffer_size=2**28,
        ffi_buffer_size=2**10)
    base_info = geoprocessing.get_raster_info(base_raster_path)
    base_nodata = base_info['nodata'][0]

    target_raster_path_result_list = []
    for percentile_value, top_nth_percentile in zip(percentile_values, ordered_top_percentile_list):
        def mask_nth_percentile_op(base_array):
            result = numpy.zeros(base_array.shape)
            valid_mask = (base_array != base_nodata) & numpy.isfinite(base_array)
            valid_mask &= (base_array >= percentile_value)
            result[valid_mask] = 1
            return result

        target_raster_path = target_raster_path_pattern.format(percentile=top_nth_percentile)
        target_raster_path_result_list.append(target_raster_path)
        pre_cog_target_raster_path = os.path.join(working_dir, os.path.basename(target_raster_path))
        geoprocessing.single_thread_raster_calculator(
            [(base_raster_path, 1)], mask_nth_percentile_op,
            pre_cog_target_raster_path, gdal.GDT_Byte, None)
        subprocess.check_call(
            f'gdal_translate {pre_cog_target_raster_path} {target_raster_path} -of COG -co BIGTIFF=YES')
    shutil.rmtree(working_dir)
    return target_raster_path_result_list


def raster_op(
        op_str, base_raster_path_list, target_raster_path, target_nodata=None,
        target_datatype=None):
    working_dir = tempfile.mkdtemp(
        prefix='ok_to_delete_', dir=os.path.dirname(target_raster_path))
    target_basename = os.path.splitext(os.path.basename(target_raster_path))[0]
    aligned_target_raster_path_list = [
        os.path.join(working_dir, f'align_{target_basename}_{os.path.basename(path)}')
        for path in base_raster_path_list]
    pixel_size = geoprocessing.get_raster_info(
        base_raster_path_list[0])['pixel_size']
    geoprocessing.align_and_resize_raster_stack(
        base_raster_path_list, aligned_target_raster_path_list,
        ['near']*len(base_raster_path_list), pixel_size, 'intersection')

    raster_info = geoprocessing.get_raster_info(base_raster_path_list[0])

    nodata_list = [
        geoprocessing.get_raster_info(path)['nodata'][0]
        for path in aligned_target_raster_path_list]
    if target_nodata is None:
        target_nodata = nodata_list[0]

    def _op(*array_list):
        if op_str == '-':
            result = array_list[0]
            array_list = array_list[1:]
        elif op_str == '+':
            result = numpy.zeros(array_list[0].shape)
        elif op_str == '*':
            result = numpy.ones(array_list[0].shape)
        final_valid_mask = numpy.zeros(array_list[0].shape, dtype=bool)
        for array, nodata in zip(array_list, nodata_list):
            local_valid_mask = numpy.isfinite(array) & (array > 0)
            if nodata is not None:
                local_valid_mask &= (array != nodata)
            final_valid_mask |= local_valid_mask
            eval_str = (
                f'result[local_valid_mask] {op_str} array[local_valid_mask]')
            result[local_valid_mask] = eval(eval_str)

        result[~final_valid_mask] = target_nodata
        return result

    if target_datatype is None:
        target_datatype = raster_info['datatype']

    pre_cog_target_raster_path = os.path.join(
        working_dir, os.path.basename(target_raster_path))
    geoprocessing.single_thread_raster_calculator(
        [(path, 1) for path in aligned_target_raster_path_list],
        _op, pre_cog_target_raster_path, target_datatype, target_nodata)
    subprocess.check_call(
        f'gdal_translate {pre_cog_target_raster_path} {target_raster_path} -of COG -co BIGTIFF=YES')
    try:
        shutil.rmtree(working_dir)
    except PermissionError:
        LOGGER.exception(f'could not delete {working_dir}, but leaving it there to keep going')


def main():
    RESULTS_DIR = 'D:\\repositories\\wwf-sipa\\final_results'
    CLIMATE_RESILIENT_PERCENTILES = os.path.join(RESULTS_DIR, 'climate_resilient_results')
    for dir_path in [RESULTS_DIR, CLIMATE_RESILIENT_PERCENTILES]:
        os.makedirs(dir_path, exist_ok=True)

    # diff x benes x services (4) x scenarios (2) x climage (2)
    country_list = ['PH', 'IDN']
    scenario_list = ['restoration', 'conservation_inf']
    climate_list = ['ssp245']
    beneficiary_list = ['dspop', 'road']
    top_percentile_list = [25, 10]

    ADMIN_POLYGONS = {
        'PH': r"D:\repositories\wwf-sipa\data\admin_boundaries\PH_gdam2.gpkg",
        'IDN': r"D:\repositories\wwf-sipa\data\admin_boundaries\IDN_gdam3.gpkg"
        }

    DIFF_FLOOD_MITIGATION_IDN_CONSERVATION_INF = os.path.join(RESULTS_DIR, "diff_flood_mitigation_IDN_conservation_inf.tif")
    DIFF_FLOOD_MITIGATION_IDN_CONSERVATION_INF_SSP245 = os.path.join(RESULTS_DIR, "diff_flood_mitigation_IDN_conservation_inf_ssp245.tif")
    DIFF_FLOOD_MITIGATION_IDN_RESTORATION = os.path.join(RESULTS_DIR, "diff_flood_mitigation_IDN_restoration.tif")
    DIFF_FLOOD_MITIGATION_IDN_RESTORATION_SSP245 = os.path.join(RESULTS_DIR, "diff_flood_mitigation_IDN_restoration_ssp245.tif")
    DIFF_FLOOD_MITIGATION_PH_CONSERVATION_INF = os.path.join(RESULTS_DIR, "diff_flood_mitigation_PH_conservation_inf.tif")
    DIFF_FLOOD_MITIGATION_PH_CONSERVATION_INF_SSP245 = os.path.join(RESULTS_DIR, "diff_flood_mitigation_PH_conservation_inf_ssp245.tif")
    DIFF_FLOOD_MITIGATION_PH_RESTORATION = os.path.join(RESULTS_DIR, "diff_flood_mitigation_PH_restoration.tif")
    DIFF_FLOOD_MITIGATION_PH_RESTORATION_SSP245 = os.path.join(RESULTS_DIR, "diff_flood_mitigation_PH_restoration_ssp245.tif")
    DIFF_QUICKFLOW_IDN_CONSERVATION_INF = os.path.join(RESULTS_DIR, 'diff_quickflow_IDN_conservation_inf.tif')
    DIFF_QUICKFLOW_IDN_CONSERVATION_INF_SSP245 = os.path.join(RESULTS_DIR, 'diff_quickflow_IDN_conservation_inf_ssp245.tif')
    DIFF_QUICKFLOW_IDN_RESTORATION = os.path.join(RESULTS_DIR, 'diff_quickflow_IDN_restoration.tif')
    DIFF_QUICKFLOW_IDN_RESTORATION_SSP245 = os.path.join(RESULTS_DIR, 'diff_quickflow_IDN_restoration_ssp245.tif')
    DIFF_QUICKFLOW_PH_CONSERVATION_INF = os.path.join(RESULTS_DIR, 'diff_quickflow_PH_conservation_inf.tif')
    DIFF_QUICKFLOW_PH_CONSERVATION_INF_SSP245 = os.path.join(RESULTS_DIR, 'diff_quickflow_PH_conservation_inf_ssp245.tif')
    DIFF_QUICKFLOW_PH_RESTORATION = os.path.join(RESULTS_DIR, 'diff_quickflow_PH_restoration.tif')
    DIFF_QUICKFLOW_PH_RESTORATION_SSP245 = os.path.join(RESULTS_DIR, 'diff_quickflow_PH_restoration_ssp245.tif')
    DIFF_RECHARGE_IDN_CONSERVATION_INF = os.path.join(RESULTS_DIR, 'diff_recharge_IDN_conservation_inf.tif')
    DIFF_RECHARGE_IDN_CONSERVATION_INF_SSP245 = os.path.join(RESULTS_DIR, "diff_recharge_IDN_conservation_inf_ssp245.tif")
    DIFF_RECHARGE_IDN_RESTORATION = os.path.join(RESULTS_DIR, 'diff_recharge_IDN_restoration.tif')
    DIFF_RECHARGE_IDN_RESTORATION_SSP245 = os.path.join(RESULTS_DIR, "diff_recharge_IDN_restoration_ssp245.tif")
    DIFF_RECHARGE_PH_CONSERVATION_INF = os.path.join(RESULTS_DIR, 'diff_recharge_PH_conservation_inf.tif')
    DIFF_RECHARGE_PH_CONSERVATION_INF_SSP245 = os.path.join(RESULTS_DIR, "diff_recharge_PH_conservation_inf_ssp245.tif")
    DIFF_RECHARGE_PH_RESTORATION = os.path.join(RESULTS_DIR, 'diff_recharge_PH_restoration.tif')
    DIFF_RECHARGE_PH_RESTORATION_SSP245 = os.path.join(RESULTS_DIR, "diff_recharge_PH_restoration_ssp245.tif")
    DIFF_SEDIMENT_IDN_CONSERVATION_INF = os.path.join(RESULTS_DIR, 'diff_sediment_IDN_conservation_inf.tif')
    DIFF_SEDIMENT_IDN_CONSERVATION_INF_SSP245 = os.path.join(RESULTS_DIR, "diff_sediment_IDN_conservation_inf_ssp245.tif")
    DIFF_SEDIMENT_IDN_RESTORATION = os.path.join(RESULTS_DIR, 'diff_sediment_IDN_restoration.tif')
    DIFF_SEDIMENT_IDN_RESTORATION_SSP245 = os.path.join(RESULTS_DIR, "diff_sediment_IDN_restoration_ssp245.tif")
    DIFF_SEDIMENT_PH_CONSERVATION_INF = os.path.join(RESULTS_DIR, 'diff_sediment_PH_conservation_inf.tif')
    DIFF_SEDIMENT_PH_CONSERVATION_INF_SSP245 = os.path.join(RESULTS_DIR, "diff_sediment_PH_conservation_inf_ssp245.tif")
    DIFF_SEDIMENT_PH_RESTORATION = os.path.join(RESULTS_DIR, 'diff_sediment_PH_restoration.tif')
    DIFF_SEDIMENT_PH_RESTORATION_SSP245 = os.path.join(RESULTS_DIR, "diff_sediment_PH_restoration_ssp245.tif")
    FLOOD_MITIGATION_IDN_BASELINE_HISTORICAL_CLIMATE = os.path.join(RESULTS_DIR, "flood_mitigation_IDN_baseline_historical_climate.tif")
    FLOOD_MITIGATION_PH_BASELINE_HISTORICAL_CLIMATE = os.path.join(RESULTS_DIR, "flood_mitigation_PH_baseline_historical_climate.tif")

    DSPOP_SERVICE_FLOOD_MITIGATION_IDN_CONSERVATION_INF = os.path.join(RESULTS_DIR, "service_dspop_flood_mitigation_IDN_conservation_inf.tif")
    DSPOP_SERVICE_FLOOD_MITIGATION_IDN_RESTORATION = os.path.join(RESULTS_DIR, "service_dspop_flood_mitigation_IDN_restoration.tif")
    DSPOP_SERVICE_FLOOD_MITIGATION_PH_CONSERVATION_INF = os.path.join(RESULTS_DIR, "service_dspop_flood_mitigation_PH_conservation_inf.tif")
    DSPOP_SERVICE_FLOOD_MITIGATION_PH_RESTORATION = os.path.join(RESULTS_DIR, "service_dspop_flood_mitigation_PH_restoration.tif")
    DSPOP_SERVICE_RECHARGE_IDN_CONSERVATION_INF = os.path.join(RESULTS_DIR, "service_dspop_recharge_IDN_conservation_inf.tif")
    ROAD_SERVICE_FLOOD_MITIGATION_IDN_CONSERVATION_INF = os.path.join(RESULTS_DIR, "service_road_flood_mitigation_IDN_conservation_inf.tif")
    ROAD_SERVICE_FLOOD_MITIGATION_IDN_RESTORATION = os.path.join(RESULTS_DIR, "service_road_flood_mitigation_IDN_restoration.tif")
    ROAD_SERVICE_FLOOD_MITIGATION_PH_CONSERVATION_INF = os.path.join(RESULTS_DIR, "service_road_flood_mitigation_PH_conservation_inf.tif")
    ROAD_SERVICE_FLOOD_MITIGATION_PH_RESTORATION = os.path.join(RESULTS_DIR, "service_road_flood_mitigation_PH_restoration.tif")
    ROAD_SERVICE_RECHARGE_IDN_CONSERVATION_INF = os.path.join(RESULTS_DIR, "service_road_recharge_IDN_conservation_inf.tif")

    DSPOP_SERVICE_RECHARGE_IDN_RESTORATION = os.path.join(RESULTS_DIR, "service_dspop_recharge_IDN_restoration.tif")
    DSPOP_SERVICE_RECHARGE_PH_CONSERVATION_INF = os.path.join(RESULTS_DIR, "service_dspop_recharge_PH_conservation_inf.tif")
    DSPOP_SERVICE_RECHARGE_PH_RESTORATION = os.path.join(RESULTS_DIR, "service_dspop_recharge_PH_restoration.tif")
    DSPOP_SERVICE_SEDIMENT_IDN_CONSERVATION_INF = os.path.join(RESULTS_DIR, "service_dspop_sediment_IDN_conservation_inf.tif")
    DSPOP_SERVICE_SEDIMENT_IDN_RESTORATION = os.path.join(RESULTS_DIR, "service_dspop_sediment_IDN_restoration.tif")
    DSPOP_SERVICE_SEDIMENT_PH_CONSERVATION_INF = os.path.join(RESULTS_DIR, "service_dspop_sediment_PH_conservation_inf.tif")
    DSPOP_SERVICE_SEDIMENT_PH_RESTORATION = os.path.join(RESULTS_DIR, "service_dspop_sediment_PH_restoration.tif")
    ROAD_SERVICE_RECHARGE_IDN_RESTORATION = os.path.join(RESULTS_DIR, "service_road_recharge_IDN_restoration.tif")
    ROAD_SERVICE_RECHARGE_PH_CONSERVATION_INF = os.path.join(RESULTS_DIR, "service_road_recharge_PH_conservation_inf.tif")
    ROAD_SERVICE_RECHARGE_PH_RESTORATION = os.path.join(RESULTS_DIR, "service_road_recharge_PH_restoration.tif")
    ROAD_SERVICE_SEDIMENT_IDN_CONSERVATION_INF = os.path.join(RESULTS_DIR, "service_road_sediment_IDN_conservation_inf.tif")
    ROAD_SERVICE_SEDIMENT_IDN_RESTORATION = os.path.join(RESULTS_DIR, "service_road_sediment_IDN_restoration.tif")
    ROAD_SERVICE_SEDIMENT_PH_CONSERVATION_INF = os.path.join(RESULTS_DIR, "service_road_sediment_PH_conservation_inf.tif")
    ROAD_SERVICE_SEDIMENT_PH_RESTORATION = os.path.join(RESULTS_DIR, "service_road_sediment_PH_restoration.tif")

    DSPOP_SERVICE_FLOOD_MITIGATION_IDN_CONSERVATION_INF_SSP245 = os.path.join(RESULTS_DIR, "service_dspop_flood_mitigation_IDN_conservation_inf_ssp245.tif")
    DSPOP_SERVICE_FLOOD_MITIGATION_IDN_RESTORATION_SSP245 = os.path.join(RESULTS_DIR, "service_dspop_flood_mitigation_IDN_restoration_ssp245.tif")
    DSPOP_SERVICE_FLOOD_MITIGATION_PH_CONSERVATION_INF_SSP245 = os.path.join(RESULTS_DIR, "service_dspop_flood_mitigation_PH_conservation_inf_ssp245.tif")
    DSPOP_SERVICE_FLOOD_MITIGATION_PH_RESTORATION_SSP245 = os.path.join(RESULTS_DIR, "service_dspop_flood_mitigation_PH_restoration_ssp245.tif")
    DSPOP_SERVICE_RECHARGE_IDN_CONSERVATION_INF_SSP245 = os.path.join(RESULTS_DIR, "service_dspop_recharge_IDN_conservation_inf_ssp245.tif")
    ROAD_SERVICE_FLOOD_MITIGATION_IDN_CONSERVATION_INF_SSP245 = os.path.join(RESULTS_DIR, "service_road_flood_mitigation_IDN_conservation_inf_ssp245.tif")
    ROAD_SERVICE_FLOOD_MITIGATION_IDN_RESTORATION_SSP245 = os.path.join(RESULTS_DIR, "service_road_flood_mitigation_IDN_restoration_ssp245.tif")
    ROAD_SERVICE_FLOOD_MITIGATION_PH_CONSERVATION_INF_SSP245 = os.path.join(RESULTS_DIR, "service_road_flood_mitigation_PH_conservation_inf_ssp245.tif")
    ROAD_SERVICE_FLOOD_MITIGATION_PH_RESTORATION_SSP245 = os.path.join(RESULTS_DIR, "service_road_flood_mitigation_PH_restoration_ssp245.tif")
    ROAD_SERVICE_RECHARGE_IDN_CONSERVATION_INF_SSP245 = os.path.join(RESULTS_DIR, "service_road_recharge_IDN_conservation_inf_ssp245.tif")

    DSPOP_SERVICE_RECHARGE_IDN_RESTORATION_SSP245 = os.path.join(RESULTS_DIR, "service_dspop_recharge_IDN_restoration_ssp245.tif")
    DSPOP_SERVICE_RECHARGE_PH_CONSERVATION_INF_SSP245 = os.path.join(RESULTS_DIR, "service_dspop_recharge_PH_conservation_inf_ssp245.tif")
    DSPOP_SERVICE_RECHARGE_PH_RESTORATION_SSP245 = os.path.join(RESULTS_DIR, "service_dspop_recharge_PH_restoration_ssp245.tif")
    DSPOP_SERVICE_SEDIMENT_IDN_CONSERVATION_INF_SSP245 = os.path.join(RESULTS_DIR, "service_dspop_sediment_IDN_conservation_inf_ssp245.tif")
    DSPOP_SERVICE_SEDIMENT_IDN_RESTORATION_SSP245 = os.path.join(RESULTS_DIR, "service_dspop_sediment_IDN_restoration_ssp245.tif")
    DSPOP_SERVICE_SEDIMENT_PH_CONSERVATION_INF_SSP245 = os.path.join(RESULTS_DIR, "service_dspop_sediment_PH_conservation_inf_ssp245.tif")
    DSPOP_SERVICE_SEDIMENT_PH_RESTORATION_SSP245 = os.path.join(RESULTS_DIR, "service_dspop_sediment_PH_restoration_ssp245.tif")
    ROAD_SERVICE_RECHARGE_IDN_RESTORATION_SSP245 = os.path.join(RESULTS_DIR, "service_road_recharge_IDN_restoration_ssp245.tif")
    ROAD_SERVICE_RECHARGE_PH_CONSERVATION_INF_SSP245 = os.path.join(RESULTS_DIR, "service_road_recharge_PH_conservation_inf_ssp245.tif")
    ROAD_SERVICE_RECHARGE_PH_RESTORATION_SSP245 = os.path.join(RESULTS_DIR, "service_road_recharge_PH_restoration_ssp245.tif")
    ROAD_SERVICE_SEDIMENT_IDN_CONSERVATION_INF_SSP245 = os.path.join(RESULTS_DIR, "service_road_sediment_IDN_conservation_inf_ssp245.tif")
    ROAD_SERVICE_SEDIMENT_IDN_RESTORATION_SSP245 = os.path.join(RESULTS_DIR, "service_road_sediment_IDN_restoration_ssp245.tif")
    ROAD_SERVICE_SEDIMENT_PH_CONSERVATION_INF_SSP245 = os.path.join(RESULTS_DIR, "service_road_sediment_PH_conservation_inf_ssp245.tif")
    ROAD_SERVICE_SEDIMENT_PH_RESTORATION_SSP245 = os.path.join(RESULTS_DIR, "service_road_sediment_PH_restoration_ssp245.tif")

    DS_POP_SERVICE_CV_IDN_CONSERVATION_INF_RESULT = os.path.join(RESULTS_DIR, 'service_dspop_cv_idn_conservation_inf_result.tif')
    ROAD_SERVICE_CV_IDN_CONSERVATION_INF_RESULT = os.path.join(RESULTS_DIR, 'service_road_cv_idn_conservation_inf_result.tif')
    DS_POP_SERVICE_CV_IDN_RESTORATION_RESULT = os.path.join(RESULTS_DIR, 'service_dspop_cv_idn_restoration_result.tif')
    ROAD_SERVICE_CV_IDN_RESTORATION_RESULT = os.path.join(RESULTS_DIR, 'service_road_cv_idn_restoration_result.tif')
    DS_POP_SERVICE_CV_PH_CONSERVATION_INF_RESULT = os.path.join(RESULTS_DIR, 'service_dspop_cv_ph_conservation_inf_result.tif')
    ROAD_SERVICE_CV_PH_CONSERVATION_INF_RESULT = os.path.join(RESULTS_DIR, 'service_road_cv_ph_conservation_inf_result.tif')
    DS_POP_SERVICE_CV_PH_RESTORATION_RESULT = os.path.join(RESULTS_DIR, 'service_dspop_cv_ph_restoration_result.tif')
    ROAD_SERVICE_CV_PH_RESTORATION_RESULT = os.path.join(RESULTS_DIR, 'service_road_cv_ph_restoration_result.tif')

    # service first then beneficiary after

    # CHECK W BCK:
    # x diff between DSPOP and ROAD 'service_...' output files
    # x check that "diff" is an output to a multiply and that the filename makes sense
    ADD_RASTER_SET = [
        (
         r"D:\repositories\coastal_risk_reduction\workspace\IDN_baseline\value_rasters\forest_mangrove_population_less_than_2m_value_index.tif",
         r"D:\repositories\coastal_risk_reduction\workspace\IDN_baseline\value_rasters\forest_mangroves_population_less_than_2m_value_index.tif",
         r"D:\repositories\coastal_risk_reduction\workspace\IDN_baseline\value_rasters\reefs_population_less_than_2m_value_index.tif",
         r"D:\repositories\coastal_risk_reduction\workspace\IDN_baseline\value_rasters\saltmarsh_population_less_than_2m_value_index.tif",
         r"D:\repositories\coastal_risk_reduction\workspace\IDN_baseline\value_rasters\savanna_population_less_than_2m_value_index.tif",
         r"D:\repositories\coastal_risk_reduction\workspace\IDN_baseline\value_rasters\seagrass_population_less_than_2m_value_index.tif",
         r"D:\repositories\coastal_risk_reduction\workspace\IDN_baseline\value_rasters\secondary forest_population_less_than_2m_value_index.tif",
         r"D:\repositories\coastal_risk_reduction\workspace\IDN_baseline\value_rasters\woody_crop_population_less_than_2m_value_index.tif",
         DS_POP_SERVICE_CV_IDN_CONSERVATION_INF_RESULT
        ),
        (
         r"D:\repositories\coastal_risk_reduction\workspace\IDN_baseline\value_rasters\woody_crop_roads_within_15km_value_index.tif",
         r"D:\repositories\coastal_risk_reduction\workspace\IDN_baseline\value_rasters\forest_mangrove_roads_within_15km_value_index.tif",
         r"D:\repositories\coastal_risk_reduction\workspace\IDN_baseline\value_rasters\forest_mangroves_roads_within_15km_value_index.tif",
         r"D:\repositories\coastal_risk_reduction\workspace\IDN_baseline\value_rasters\reefs_roads_within_15km_value_index.tif",
         r"D:\repositories\coastal_risk_reduction\workspace\IDN_baseline\value_rasters\saltmarsh_roads_within_15km_value_index.tif",
         r"D:\repositories\coastal_risk_reduction\workspace\IDN_baseline\value_rasters\savanna_roads_within_15km_value_index.tif",
         r"D:\repositories\coastal_risk_reduction\workspace\IDN_baseline\value_rasters\seagrass_roads_within_15km_value_index.tif",
         r"D:\repositories\coastal_risk_reduction\workspace\IDN_baseline\value_rasters\secondary forest_roads_within_15km_value_index.tif",
         ROAD_SERVICE_CV_IDN_CONSERVATION_INF_RESULT
        ),
        (
         r"D:\repositories\coastal_risk_reduction\workspace\IDN_restoration\value_rasters\forest_mangrove_population_less_than_2m_value_index.tif",
         r"D:\repositories\coastal_risk_reduction\workspace\IDN_restoration\value_rasters\forest_mangroves_population_less_than_2m_value_index.tif",
         r"D:\repositories\coastal_risk_reduction\workspace\IDN_restoration\value_rasters\reefs_population_less_than_2m_value_index.tif",
         r"D:\repositories\coastal_risk_reduction\workspace\IDN_restoration\value_rasters\saltmarsh_population_less_than_2m_value_index.tif",
         r"D:\repositories\coastal_risk_reduction\workspace\IDN_restoration\value_rasters\savanna_population_less_than_2m_value_index.tif",
         r"D:\repositories\coastal_risk_reduction\workspace\IDN_restoration\value_rasters\seagrass_population_less_than_2m_value_index.tif",
         r"D:\repositories\coastal_risk_reduction\workspace\IDN_restoration\value_rasters\secondary forest_population_less_than_2m_value_index.tif",
         r"D:\repositories\coastal_risk_reduction\workspace\IDN_restoration\value_rasters\woody_crop_population_less_than_2m_value_index.tif",
         DS_POP_SERVICE_CV_IDN_RESTORATION_RESULT
        ),
        (
         r"D:\repositories\coastal_risk_reduction\workspace\IDN_restoration\value_rasters\woody_crop_roads_within_15km_value_index.tif",
         r"D:\repositories\coastal_risk_reduction\workspace\IDN_restoration\value_rasters\forest_mangrove_roads_within_15km_value_index.tif",
         r"D:\repositories\coastal_risk_reduction\workspace\IDN_restoration\value_rasters\forest_mangroves_roads_within_15km_value_index.tif",
         r"D:\repositories\coastal_risk_reduction\workspace\IDN_restoration\value_rasters\reefs_roads_within_15km_value_index.tif",
         r"D:\repositories\coastal_risk_reduction\workspace\IDN_restoration\value_rasters\saltmarsh_roads_within_15km_value_index.tif",
         r"D:\repositories\coastal_risk_reduction\workspace\IDN_restoration\value_rasters\savanna_roads_within_15km_value_index.tif",
         r"D:\repositories\coastal_risk_reduction\workspace\IDN_restoration\value_rasters\seagrass_roads_within_15km_value_index.tif",
         r"D:\repositories\coastal_risk_reduction\workspace\IDN_restoration\value_rasters\secondary forest_roads_within_15km_value_index.tif",
         ROAD_SERVICE_CV_IDN_RESTORATION_RESULT
        ),
        (
         r"D:\repositories\coastal_risk_reduction\workspace\ph_baseline\value_rasters\brush_population_less_than_2m_value_index.tif",
         r"D:\repositories\coastal_risk_reduction\workspace\ph_baseline\value_rasters\grassland_population_less_than_2m_value_index.tif",
         r"D:\repositories\coastal_risk_reduction\workspace\ph_baseline\value_rasters\perennial_crop_population_less_than_2m_value_index.tif",
         r"D:\repositories\coastal_risk_reduction\workspace\ph_baseline\value_rasters\forest_mangroves_population_less_than_2m_value_index.tif",
         r"D:\repositories\coastal_risk_reduction\workspace\ph_baseline\value_rasters\reefs_population_less_than_2m_value_index.tif",
         r"D:\repositories\coastal_risk_reduction\workspace\ph_baseline\value_rasters\saltmarsh_population_less_than_2m_value_index.tif",
         r"D:\repositories\coastal_risk_reduction\workspace\ph_baseline\value_rasters\seagrass_population_less_than_2m_value_index.tif",
         r"D:\repositories\coastal_risk_reduction\workspace\ph_baseline\value_rasters\secondary forest_population_less_than_2m_value_index.tif",
         DS_POP_SERVICE_CV_PH_CONSERVATION_INF_RESULT
        ),
        (
         r"D:\repositories\coastal_risk_reduction\workspace\ph_baseline\value_rasters\brush_roads_within_15km_value_index.tif",
         r"D:\repositories\coastal_risk_reduction\workspace\ph_baseline\value_rasters\grassland_roads_within_15km_value_index.tif",
         r"D:\repositories\coastal_risk_reduction\workspace\ph_baseline\value_rasters\perennial_crop_roads_within_15km_value_index.tif",
         r"D:\repositories\coastal_risk_reduction\workspace\ph_baseline\value_rasters\forest_mangroves_roads_within_15km_value_index.tif",
         r"D:\repositories\coastal_risk_reduction\workspace\ph_baseline\value_rasters\reefs_roads_within_15km_value_index.tif",
         r"D:\repositories\coastal_risk_reduction\workspace\ph_baseline\value_rasters\saltmarsh_roads_within_15km_value_index.tif",
         r"D:\repositories\coastal_risk_reduction\workspace\ph_baseline\value_rasters\seagrass_roads_within_15km_value_index.tif",
         r"D:\repositories\coastal_risk_reduction\workspace\ph_baseline\value_rasters\secondary forest_roads_within_15km_value_index.tif",
         ROAD_SERVICE_CV_PH_CONSERVATION_INF_RESULT
        ),
        (
         r"D:\repositories\coastal_risk_reduction\workspace\ph_restoration\value_rasters\brush_population_less_than_2m_value_index.tif",
         r"D:\repositories\coastal_risk_reduction\workspace\ph_restoration\value_rasters\grassland_population_less_than_2m_value_index.tif",
         r"D:\repositories\coastal_risk_reduction\workspace\ph_restoration\value_rasters\perennial_crop_population_less_than_2m_value_index.tif",
         r"D:\repositories\coastal_risk_reduction\workspace\ph_restoration\value_rasters\forest_mangroves_population_less_than_2m_value_index.tif",
         r"D:\repositories\coastal_risk_reduction\workspace\ph_restoration\value_rasters\reefs_population_less_than_2m_value_index.tif",
         r"D:\repositories\coastal_risk_reduction\workspace\ph_restoration\value_rasters\saltmarsh_population_less_than_2m_value_index.tif",
         r"D:\repositories\coastal_risk_reduction\workspace\ph_restoration\value_rasters\seagrass_population_less_than_2m_value_index.tif",
         r"D:\repositories\coastal_risk_reduction\workspace\ph_restoration\value_rasters\secondary forest_population_less_than_2m_value_index.tif",
         DS_POP_SERVICE_CV_PH_RESTORATION_RESULT
        ),
        (
         r"D:\repositories\coastal_risk_reduction\workspace\ph_restoration\value_rasters\brush_roads_within_15km_value_index.tif",
         r"D:\repositories\coastal_risk_reduction\workspace\ph_restoration\value_rasters\grassland_roads_within_15km_value_index.tif",
         r"D:\repositories\coastal_risk_reduction\workspace\ph_restoration\value_rasters\perennial_crop_roads_within_15km_value_index.tif",
         r"D:\repositories\coastal_risk_reduction\workspace\ph_restoration\value_rasters\forest_mangroves_roads_within_15km_value_index.tif",
         r"D:\repositories\coastal_risk_reduction\workspace\ph_restoration\value_rasters\reefs_roads_within_15km_value_index.tif",
         r"D:\repositories\coastal_risk_reduction\workspace\ph_restoration\value_rasters\saltmarsh_roads_within_15km_value_index.tif",
         r"D:\repositories\coastal_risk_reduction\workspace\ph_restoration\value_rasters\seagrass_roads_within_15km_value_index.tif",
         r"D:\repositories\coastal_risk_reduction\workspace\ph_restoration\value_rasters\secondary forest_roads_within_15km_value_index.tif",
         ROAD_SERVICE_CV_PH_RESTORATION_RESULT
        )
    ]

    SUBTRACT_RASTER_SET = [
        (r"D:\repositories\ndr_sdr_global\wwf_IDN_baseline_historical_climate\stitched_sed_export_wwf_IDN_baseline_historical_climate.tif",
         r"D:\repositories\ndr_sdr_global\wwf_IDN_restoration_historical_climate\stitched_sed_export_wwf_IDN_restoration_historical_climate.tif",
         DIFF_SEDIMENT_IDN_RESTORATION),
        (r"D:\repositories\ndr_sdr_global\wwf_IDN_baseline_ssp245_climate\stitched_sed_export_wwf_IDN_baseline_ssp245_climate.tif",
         r"D:\repositories\ndr_sdr_global\wwf_IDN_restoration_ssp245_climate\stitched_sed_export_wwf_IDN_restoration_ssp245_climate.tif",
         DIFF_SEDIMENT_IDN_RESTORATION_SSP245),
        (r"D:\repositories\ndr_sdr_global\wwf_IDN_infra_historical_climate\stitched_sed_export_wwf_IDN_infra_historical_climate.tif",
         r"D:\repositories\ndr_sdr_global\wwf_IDN_baseline_historical_climate\stitched_sed_export_wwf_IDN_baseline_historical_climate.tif",
         DIFF_SEDIMENT_IDN_CONSERVATION_INF),
        (r"D:\repositories\ndr_sdr_global\wwf_IDN_infra_ssp245_climate\stitched_sed_export_wwf_IDN_infra_ssp245_climate.tif",
         r"D:\repositories\ndr_sdr_global\wwf_IDN_baseline_ssp245_climate\stitched_sed_export_wwf_IDN_baseline_ssp245_climate.tif",
         DIFF_SEDIMENT_IDN_CONSERVATION_INF_SSP245),
        (r"D:\repositories\ndr_sdr_global\wwf_PH_baseline_historical_climate\stitched_sed_export_wwf_PH_baseline_historical_climate.tif",
         r"D:\repositories\ndr_sdr_global\wwf_PH_restoration_historical_climate\stitched_sed_export_wwf_PH_restoration_historical_climate.tif",
         DIFF_SEDIMENT_PH_RESTORATION),
        (r"D:\repositories\ndr_sdr_global\wwf_PH_baseline_ssp245_climate\stitched_sed_export_wwf_PH_baseline_ssp245_climate.tif",
         r"D:\repositories\ndr_sdr_global\wwf_PH_restoration_ssp245_climate\stitched_sed_export_wwf_PH_restoration_ssp245_climate.tif",
         DIFF_SEDIMENT_PH_RESTORATION_SSP245),
        (r"D:\repositories\ndr_sdr_global\wwf_PH_infra_historical_climate\stitched_sed_export_wwf_PH_infra_historical_climate.tif",
         r"D:\repositories\ndr_sdr_global\wwf_PH_baseline_historical_climate\stitched_sed_export_wwf_PH_baseline_historical_climate.tif",
         DIFF_SEDIMENT_PH_CONSERVATION_INF),
        (r"D:\repositories\ndr_sdr_global\wwf_PH_infra_ssp245_climate\stitched_sed_export_wwf_PH_infra_ssp245_climate.tif",
         r"D:\repositories\ndr_sdr_global\wwf_PH_baseline_ssp245_climate\stitched_sed_export_wwf_PH_baseline_ssp245_climate.tif",
         DIFF_SEDIMENT_PH_CONSERVATION_INF_SSP245),
        (r"D:\repositories\swy_global\workspace_swy_wwf_IDN_restoration_historical_climate\B_wwf_IDN_restoration_historical_climate.tif",
         r"D:\repositories\swy_global\workspace_swy_wwf_IDN_baseline_historical_climate\B_wwf_IDN_baseline_historical_climate.tif",
         DIFF_RECHARGE_IDN_RESTORATION),
        (r"D:\repositories\swy_global\workspace_swy_wwf_IDN_baseline_historical_climate\QF_wwf_IDN_baseline_historical_climate.tif",
         r"D:\repositories\swy_global\workspace_swy_wwf_IDN_restoration_historical_climate\QF_wwf_IDN_restoration_historical_climate.tif",
         DIFF_QUICKFLOW_IDN_RESTORATION),
        (r"D:\repositories\swy_global\workspace_swy_wwf_IDN_restoration_ssp245_climate10\B_wwf_IDN_restoration_ssp245_climate10.tif",
         r"D:\repositories\swy_global\workspace_swy_wwf_IDN_baseline_ssp245_climate10\B_wwf_IDN_baseline_ssp245_climate10.tif",
         DIFF_RECHARGE_IDN_RESTORATION_SSP245),
        (r"D:\repositories\swy_global\workspace_swy_wwf_IDN_baseline_ssp245_climate90\QF_wwf_IDN_baseline_ssp245_climate90.tif",
         r"D:\repositories\swy_global\workspace_swy_wwf_IDN_restoration_ssp245_climate90\QF_wwf_IDN_restoration_ssp245_climate90.tif",
         DIFF_QUICKFLOW_IDN_RESTORATION_SSP245),
        (r"D:\repositories\swy_global\workspace_swy_wwf_IDN_baseline_historical_climate\B_wwf_IDN_baseline_historical_climate.tif",
         r"D:\repositories\swy_global\workspace_swy_wwf_IDN_infra_historical_climate\B_wwf_IDN_infra_historical_climate.tif",
         DIFF_RECHARGE_IDN_CONSERVATION_INF),
        (r"D:\repositories\swy_global\workspace_swy_wwf_IDN_infra_historical_climate\QF_wwf_IDN_infra_historical_climate.tif",
         r"D:\repositories\swy_global\workspace_swy_wwf_IDN_baseline_historical_climate\QF_wwf_IDN_baseline_historical_climate.tif",
         DIFF_QUICKFLOW_IDN_CONSERVATION_INF),
        (r"D:\repositories\swy_global\workspace_swy_wwf_IDN_baseline_ssp245_climate10\B_wwf_IDN_baseline_ssp245_climate10.tif",
         r"D:\repositories\swy_global\workspace_swy_wwf_IDN_infra_ssp245_climate10\B_wwf_IDN_infra_ssp245_climate10.tif",
         DIFF_RECHARGE_IDN_CONSERVATION_INF_SSP245),
        (r"D:\repositories\swy_global\workspace_swy_wwf_IDN_infra_ssp245_climate90\QF_wwf_IDN_infra_ssp245_climate90.tif",
         r"D:\repositories\swy_global\workspace_swy_wwf_IDN_baseline_ssp245_climate90\QF_wwf_IDN_baseline_ssp245_climate90.tif",
         DIFF_QUICKFLOW_IDN_CONSERVATION_INF_SSP245),
        (r"D:\repositories\swy_global\workspace_swy_wwf_PH_restoration_historical_climate\B_wwf_PH_restoration_historical_climate.tif",
         r"D:\repositories\swy_global\workspace_swy_wwf_PH_baseline_historical_climate\B_wwf_PH_baseline_historical_climate.tif",
         DIFF_RECHARGE_PH_RESTORATION),
        (r"D:\repositories\swy_global\workspace_swy_wwf_PH_baseline_historical_climate\QF_wwf_PH_baseline_historical_climate.tif",
         r"D:\repositories\swy_global\workspace_swy_wwf_PH_restoration_historical_climate\QF_wwf_PH_restoration_historical_climate.tif",
         DIFF_QUICKFLOW_PH_RESTORATION),
        (r"D:\repositories\swy_global\workspace_swy_wwf_PH_restoration_ssp245_climate10\B_wwf_PH_restoration_ssp245_climate10.tif",
         r"D:\repositories\swy_global\workspace_swy_wwf_PH_baseline_ssp245_climate10\B_wwf_PH_baseline_ssp245_climate10.tif",
         DIFF_RECHARGE_PH_RESTORATION_SSP245),
        (r"D:\repositories\swy_global\workspace_swy_wwf_PH_baseline_ssp245_climate90\QF_wwf_PH_baseline_ssp245_climate90.tif",
         r"D:\repositories\swy_global\workspace_swy_wwf_PH_restoration_ssp245_climate90\QF_wwf_PH_restoration_ssp245_climate90.tif",
         DIFF_QUICKFLOW_PH_RESTORATION_SSP245),
        (r"D:\repositories\swy_global\workspace_swy_wwf_PH_baseline_historical_climate\B_wwf_PH_baseline_historical_climate.tif",
         r"D:\repositories\swy_global\workspace_swy_wwf_PH_infra_historical_climate\B_wwf_PH_infra_historical_climate.tif",
         DIFF_RECHARGE_PH_CONSERVATION_INF),
        (r"D:\repositories\swy_global\workspace_swy_wwf_PH_infra_historical_climate\QF_wwf_PH_infra_historical_climate.tif",
         r"D:\repositories\swy_global\workspace_swy_wwf_PH_baseline_historical_climate\QF_wwf_PH_baseline_historical_climate.tif",
         DIFF_QUICKFLOW_PH_CONSERVATION_INF),
        (r"D:\repositories\swy_global\workspace_swy_wwf_PH_baseline_ssp245_climate10\B_wwf_PH_baseline_ssp245_climate10.tif",
         r"D:\repositories\swy_global\workspace_swy_wwf_PH_infra_ssp245_climate10\B_wwf_PH_infra_ssp245_climate10.tif",
         DIFF_RECHARGE_PH_CONSERVATION_INF_SSP245),
        (r"D:\repositories\swy_global\workspace_swy_wwf_PH_infra_ssp245_climate90\QF_wwf_PH_infra_ssp245_climate90.tif",
         r"D:\repositories\swy_global\workspace_swy_wwf_PH_baseline_ssp245_climate90\QF_wwf_PH_baseline_ssp245_climate90.tif",
         DIFF_QUICKFLOW_PH_CONSERVATION_INF_SSP245)
        ]

    MULTIPLY_RASTER_SET = [
        (r"D:\repositories\wwf-sipa\idn_downstream_flood_risk.tif",
         r"D:\repositories\swy_global\workspace_swy_wwf_IDN_baseline_historical_climate\QF_wwf_IDN_baseline_historical_climate.tif",
         FLOOD_MITIGATION_IDN_BASELINE_HISTORICAL_CLIMATE),
        (r"D:\repositories\wwf-sipa\idn_downstream_flood_risk.tif",
         DIFF_QUICKFLOW_IDN_CONSERVATION_INF,
         DIFF_FLOOD_MITIGATION_IDN_CONSERVATION_INF),
        (r"D:\repositories\wwf-sipa\idn_downstream_flood_risk.tif",
         DIFF_QUICKFLOW_IDN_CONSERVATION_INF_SSP245,
         DIFF_FLOOD_MITIGATION_IDN_CONSERVATION_INF_SSP245),
        (r"D:\repositories\wwf-sipa\idn_downstream_flood_risk.tif",
         DIFF_QUICKFLOW_IDN_RESTORATION,
         DIFF_FLOOD_MITIGATION_IDN_RESTORATION),
        (r"D:\repositories\wwf-sipa\idn_downstream_flood_risk.tif",
         DIFF_QUICKFLOW_IDN_RESTORATION_SSP245,
         DIFF_FLOOD_MITIGATION_IDN_RESTORATION_SSP245),
        (r"D:\repositories\wwf-sipa\ph_downstream_flood_risk.tif",
         r"D:\repositories\swy_global\workspace_swy_wwf_PH_baseline_historical_climate\QF_wwf_PH_baseline_historical_climate.tif",
         FLOOD_MITIGATION_PH_BASELINE_HISTORICAL_CLIMATE),
        (r"D:\repositories\wwf-sipa\ph_downstream_flood_risk.tif",
         DIFF_QUICKFLOW_PH_CONSERVATION_INF,
         DIFF_FLOOD_MITIGATION_PH_CONSERVATION_INF),
        (r"D:\repositories\wwf-sipa\ph_downstream_flood_risk.tif",
         DIFF_QUICKFLOW_PH_CONSERVATION_INF_SSP245,
         DIFF_FLOOD_MITIGATION_PH_CONSERVATION_INF_SSP245),
        (r"D:\repositories\wwf-sipa\ph_downstream_flood_risk.tif",
         DIFF_QUICKFLOW_PH_RESTORATION,
         DIFF_FLOOD_MITIGATION_PH_RESTORATION),
        (r"D:\repositories\wwf-sipa\ph_downstream_flood_risk.tif",
         DIFF_QUICKFLOW_PH_RESTORATION_SSP245,
         DIFF_FLOOD_MITIGATION_PH_RESTORATION_SSP245),
        (DIFF_RECHARGE_IDN_RESTORATION,
         r"D:\repositories\wwf-sipa\downstream_beneficiary_workspace\num_of_downstream_beneficiaries_per_pixel_idn_downstream_dspop_benes_md5_e4d2c4.tif",
         DSPOP_SERVICE_RECHARGE_IDN_RESTORATION),
        (DIFF_FLOOD_MITIGATION_IDN_RESTORATION,
         r"D:\repositories\wwf-sipa\downstream_beneficiary_workspace\num_of_downstream_beneficiaries_per_pixel_idn_downstream_dspop_benes_md5_e4d2c4.tif",
         DSPOP_SERVICE_FLOOD_MITIGATION_IDN_RESTORATION),
        (DIFF_SEDIMENT_IDN_RESTORATION,
         r"D:\repositories\wwf-sipa\downstream_beneficiary_workspace\num_of_downstream_beneficiaries_per_pixel_idn_downstream_dspop_benes_md5_e4d2c4.tif",
         DSPOP_SERVICE_SEDIMENT_IDN_RESTORATION),
        (DIFF_RECHARGE_IDN_RESTORATION,
         r"D:\repositories\wwf-sipa\downstream_beneficiary_workspace\num_of_downstream_beneficiaries_per_pixel_idn_downstream_road2019_benes_md5_8ec2cd.tif",
         ROAD_SERVICE_RECHARGE_IDN_RESTORATION),
        (DIFF_FLOOD_MITIGATION_IDN_RESTORATION,
         r"D:\repositories\wwf-sipa\downstream_beneficiary_workspace\num_of_downstream_beneficiaries_per_pixel_idn_downstream_road2019_benes_md5_8ec2cd.tif",
         ROAD_SERVICE_FLOOD_MITIGATION_IDN_RESTORATION),
        (DIFF_SEDIMENT_IDN_RESTORATION,
         r"D:\repositories\wwf-sipa\downstream_beneficiary_workspace\num_of_downstream_beneficiaries_per_pixel_idn_downstream_road2019_benes_md5_8ec2cd.tif",
         ROAD_SERVICE_SEDIMENT_IDN_RESTORATION),
        (DIFF_RECHARGE_PH_RESTORATION,
         r"D:\repositories\wwf-sipa\downstream_beneficiary_workspace\num_of_downstream_beneficiaries_per_pixel_ph_downstream_dspop_benes_md5_028732.tif",
         DSPOP_SERVICE_RECHARGE_PH_RESTORATION),
        (DIFF_FLOOD_MITIGATION_PH_RESTORATION,
         r"D:\repositories\wwf-sipa\downstream_beneficiary_workspace\num_of_downstream_beneficiaries_per_pixel_ph_downstream_dspop_benes_md5_028732.tif",
         DSPOP_SERVICE_FLOOD_MITIGATION_PH_RESTORATION),
        (DIFF_SEDIMENT_PH_RESTORATION,
         r"D:\repositories\wwf-sipa\downstream_beneficiary_workspace\num_of_downstream_beneficiaries_per_pixel_ph_downstream_dspop_benes_md5_028732.tif",
         DSPOP_SERVICE_SEDIMENT_PH_RESTORATION),
        (DIFF_RECHARGE_PH_RESTORATION,
         r"D:\repositories\wwf-sipa\downstream_beneficiary_workspace\num_of_downstream_beneficiaries_per_pixel_ph_downstream_road2019_benes_md5_870a6c.tif",
         ROAD_SERVICE_RECHARGE_PH_RESTORATION),
        (DIFF_FLOOD_MITIGATION_PH_RESTORATION,
         r"D:\repositories\wwf-sipa\downstream_beneficiary_workspace\num_of_downstream_beneficiaries_per_pixel_ph_downstream_road2019_benes_md5_870a6c.tif",
         ROAD_SERVICE_FLOOD_MITIGATION_PH_RESTORATION),
        (DIFF_SEDIMENT_PH_RESTORATION,
         r"D:\repositories\wwf-sipa\downstream_beneficiary_workspace\num_of_downstream_beneficiaries_per_pixel_ph_downstream_road2019_benes_md5_870a6c.tif",
         ROAD_SERVICE_SEDIMENT_PH_RESTORATION),
        (DIFF_RECHARGE_IDN_CONSERVATION_INF,
         r"D:\repositories\wwf-sipa\downstream_beneficiary_workspace\num_of_downstream_beneficiaries_per_pixel_idn_downstream_dspop_benes_md5_e4d2c4.tif",
         DSPOP_SERVICE_RECHARGE_IDN_CONSERVATION_INF),
        (DIFF_FLOOD_MITIGATION_IDN_CONSERVATION_INF,
         r"D:\repositories\wwf-sipa\downstream_beneficiary_workspace\num_of_downstream_beneficiaries_per_pixel_idn_downstream_dspop_benes_md5_e4d2c4.tif",
         DSPOP_SERVICE_FLOOD_MITIGATION_IDN_CONSERVATION_INF),
        (DIFF_SEDIMENT_IDN_CONSERVATION_INF,
         r"D:\repositories\wwf-sipa\downstream_beneficiary_workspace\num_of_downstream_beneficiaries_per_pixel_idn_downstream_dspop_benes_md5_e4d2c4.tif",
         DSPOP_SERVICE_SEDIMENT_IDN_CONSERVATION_INF),
        (DIFF_RECHARGE_IDN_CONSERVATION_INF,
         r"D:\repositories\wwf-sipa\downstream_beneficiary_workspace\num_of_downstream_beneficiaries_per_pixel_idn_downstream_road2019_benes_md5_8ec2cd.tif",
         ROAD_SERVICE_RECHARGE_IDN_CONSERVATION_INF),
        (DIFF_FLOOD_MITIGATION_IDN_CONSERVATION_INF,
         r"D:\repositories\wwf-sipa\downstream_beneficiary_workspace\num_of_downstream_beneficiaries_per_pixel_idn_downstream_road2019_benes_md5_8ec2cd.tif",
         ROAD_SERVICE_FLOOD_MITIGATION_IDN_CONSERVATION_INF),
        (DIFF_SEDIMENT_IDN_CONSERVATION_INF,
         r"D:\repositories\wwf-sipa\downstream_beneficiary_workspace\num_of_downstream_beneficiaries_per_pixel_idn_downstream_road2019_benes_md5_8ec2cd.tif",
         ROAD_SERVICE_SEDIMENT_IDN_CONSERVATION_INF),
        (DIFF_RECHARGE_PH_CONSERVATION_INF,
         r"D:\repositories\wwf-sipa\downstream_beneficiary_workspace\num_of_downstream_beneficiaries_per_pixel_ph_downstream_dspop_benes_md5_028732.tif",
         DSPOP_SERVICE_RECHARGE_PH_CONSERVATION_INF),
        (DIFF_FLOOD_MITIGATION_PH_CONSERVATION_INF,
         r"D:\repositories\wwf-sipa\downstream_beneficiary_workspace\num_of_downstream_beneficiaries_per_pixel_ph_downstream_dspop_benes_md5_028732.tif",
         DSPOP_SERVICE_FLOOD_MITIGATION_PH_CONSERVATION_INF),
        (DIFF_SEDIMENT_PH_CONSERVATION_INF,
         r"D:\repositories\wwf-sipa\downstream_beneficiary_workspace\num_of_downstream_beneficiaries_per_pixel_ph_downstream_dspop_benes_md5_028732.tif",
         DSPOP_SERVICE_SEDIMENT_PH_CONSERVATION_INF),
        (DIFF_RECHARGE_PH_CONSERVATION_INF,
         r"D:\repositories\wwf-sipa\downstream_beneficiary_workspace\num_of_downstream_beneficiaries_per_pixel_ph_downstream_road2019_benes_md5_870a6c.tif",
         ROAD_SERVICE_RECHARGE_PH_CONSERVATION_INF),
        (DIFF_FLOOD_MITIGATION_PH_CONSERVATION_INF,
         r"D:\repositories\wwf-sipa\downstream_beneficiary_workspace\num_of_downstream_beneficiaries_per_pixel_ph_downstream_road2019_benes_md5_870a6c.tif",
         ROAD_SERVICE_FLOOD_MITIGATION_PH_CONSERVATION_INF),
        (DIFF_SEDIMENT_PH_CONSERVATION_INF,
         r"D:\repositories\wwf-sipa\downstream_beneficiary_workspace\num_of_downstream_beneficiaries_per_pixel_ph_downstream_road2019_benes_md5_870a6c.tif",
         ROAD_SERVICE_SEDIMENT_PH_CONSERVATION_INF),
        (DIFF_RECHARGE_IDN_RESTORATION_SSP245,
         r"D:\repositories\wwf-sipa\downstream_beneficiary_workspace\num_of_downstream_beneficiaries_per_pixel_idn_downstream_dspop_benes_md5_e4d2c4.tif",
         DSPOP_SERVICE_RECHARGE_IDN_RESTORATION_SSP245),
        (DIFF_FLOOD_MITIGATION_IDN_RESTORATION_SSP245,
         r"D:\repositories\wwf-sipa\downstream_beneficiary_workspace\num_of_downstream_beneficiaries_per_pixel_idn_downstream_dspop_benes_md5_e4d2c4.tif",
         DSPOP_SERVICE_FLOOD_MITIGATION_IDN_RESTORATION_SSP245),
        (DIFF_SEDIMENT_IDN_RESTORATION_SSP245,
         r"D:\repositories\wwf-sipa\downstream_beneficiary_workspace\num_of_downstream_beneficiaries_per_pixel_idn_downstream_dspop_benes_md5_e4d2c4.tif",
         DSPOP_SERVICE_SEDIMENT_IDN_RESTORATION_SSP245),
        (DIFF_RECHARGE_IDN_RESTORATION_SSP245,
         r"D:\repositories\wwf-sipa\downstream_beneficiary_workspace\num_of_downstream_beneficiaries_per_pixel_idn_downstream_road2019_benes_md5_8ec2cd.tif",
         ROAD_SERVICE_RECHARGE_IDN_RESTORATION_SSP245),
        (DIFF_FLOOD_MITIGATION_IDN_RESTORATION_SSP245,
         r"D:\repositories\wwf-sipa\downstream_beneficiary_workspace\num_of_downstream_beneficiaries_per_pixel_idn_downstream_road2019_benes_md5_8ec2cd.tif",
         ROAD_SERVICE_FLOOD_MITIGATION_IDN_RESTORATION_SSP245),
        (DIFF_SEDIMENT_IDN_RESTORATION_SSP245,
         r"D:\repositories\wwf-sipa\downstream_beneficiary_workspace\num_of_downstream_beneficiaries_per_pixel_idn_downstream_road2019_benes_md5_8ec2cd.tif",
         ROAD_SERVICE_SEDIMENT_IDN_RESTORATION_SSP245),
        (DIFF_RECHARGE_PH_RESTORATION_SSP245,
         r"D:\repositories\wwf-sipa\downstream_beneficiary_workspace\num_of_downstream_beneficiaries_per_pixel_ph_downstream_dspop_benes_md5_028732.tif",
         DSPOP_SERVICE_RECHARGE_PH_RESTORATION_SSP245),
        (DIFF_FLOOD_MITIGATION_PH_RESTORATION_SSP245,
         r"D:\repositories\wwf-sipa\downstream_beneficiary_workspace\num_of_downstream_beneficiaries_per_pixel_ph_downstream_dspop_benes_md5_028732.tif",
         DSPOP_SERVICE_FLOOD_MITIGATION_PH_RESTORATION_SSP245),
        (DIFF_SEDIMENT_PH_RESTORATION_SSP245,
         r"D:\repositories\wwf-sipa\downstream_beneficiary_workspace\num_of_downstream_beneficiaries_per_pixel_ph_downstream_dspop_benes_md5_028732.tif",
         DSPOP_SERVICE_SEDIMENT_PH_RESTORATION_SSP245),
        (DIFF_RECHARGE_PH_RESTORATION_SSP245,
         r"D:\repositories\wwf-sipa\downstream_beneficiary_workspace\num_of_downstream_beneficiaries_per_pixel_ph_downstream_road2019_benes_md5_870a6c.tif",
         ROAD_SERVICE_RECHARGE_PH_RESTORATION_SSP245),
        (DIFF_FLOOD_MITIGATION_PH_RESTORATION_SSP245,
         r"D:\repositories\wwf-sipa\downstream_beneficiary_workspace\num_of_downstream_beneficiaries_per_pixel_ph_downstream_road2019_benes_md5_870a6c.tif",
         ROAD_SERVICE_FLOOD_MITIGATION_PH_RESTORATION_SSP245),
        (DIFF_SEDIMENT_PH_RESTORATION_SSP245,
         r"D:\repositories\wwf-sipa\downstream_beneficiary_workspace\num_of_downstream_beneficiaries_per_pixel_ph_downstream_road2019_benes_md5_870a6c.tif",
         ROAD_SERVICE_SEDIMENT_PH_RESTORATION_SSP245),
        (DIFF_RECHARGE_IDN_CONSERVATION_INF_SSP245,
         r"D:\repositories\wwf-sipa\downstream_beneficiary_workspace\num_of_downstream_beneficiaries_per_pixel_idn_downstream_dspop_benes_md5_e4d2c4.tif",
         DSPOP_SERVICE_RECHARGE_IDN_CONSERVATION_INF_SSP245),
        (DIFF_FLOOD_MITIGATION_IDN_CONSERVATION_INF_SSP245,
         r"D:\repositories\wwf-sipa\downstream_beneficiary_workspace\num_of_downstream_beneficiaries_per_pixel_idn_downstream_dspop_benes_md5_e4d2c4.tif",
         DSPOP_SERVICE_FLOOD_MITIGATION_IDN_CONSERVATION_INF_SSP245),
        (DIFF_SEDIMENT_IDN_CONSERVATION_INF_SSP245,
         r"D:\repositories\wwf-sipa\downstream_beneficiary_workspace\num_of_downstream_beneficiaries_per_pixel_idn_downstream_dspop_benes_md5_e4d2c4.tif",
         DSPOP_SERVICE_SEDIMENT_IDN_CONSERVATION_INF_SSP245),
        (DIFF_RECHARGE_IDN_CONSERVATION_INF_SSP245,
         r"D:\repositories\wwf-sipa\downstream_beneficiary_workspace\num_of_downstream_beneficiaries_per_pixel_idn_downstream_road2019_benes_md5_8ec2cd.tif",
         ROAD_SERVICE_RECHARGE_IDN_CONSERVATION_INF_SSP245),
        (DIFF_FLOOD_MITIGATION_IDN_CONSERVATION_INF_SSP245,
         r"D:\repositories\wwf-sipa\downstream_beneficiary_workspace\num_of_downstream_beneficiaries_per_pixel_idn_downstream_road2019_benes_md5_8ec2cd.tif",
         ROAD_SERVICE_FLOOD_MITIGATION_IDN_CONSERVATION_INF_SSP245),
        (DIFF_SEDIMENT_IDN_CONSERVATION_INF_SSP245,
         r"D:\repositories\wwf-sipa\downstream_beneficiary_workspace\num_of_downstream_beneficiaries_per_pixel_idn_downstream_road2019_benes_md5_8ec2cd.tif",
         ROAD_SERVICE_SEDIMENT_IDN_CONSERVATION_INF_SSP245),
        (DIFF_RECHARGE_PH_CONSERVATION_INF_SSP245,
         r"D:\repositories\wwf-sipa\downstream_beneficiary_workspace\num_of_downstream_beneficiaries_per_pixel_ph_downstream_dspop_benes_md5_028732.tif",
         DSPOP_SERVICE_RECHARGE_PH_CONSERVATION_INF_SSP245),
        (DIFF_FLOOD_MITIGATION_PH_CONSERVATION_INF_SSP245,
         r"D:\repositories\wwf-sipa\downstream_beneficiary_workspace\num_of_downstream_beneficiaries_per_pixel_ph_downstream_dspop_benes_md5_028732.tif",
         DSPOP_SERVICE_FLOOD_MITIGATION_PH_CONSERVATION_INF_SSP245),
        (DIFF_SEDIMENT_PH_CONSERVATION_INF_SSP245,
         r"D:\repositories\wwf-sipa\downstream_beneficiary_workspace\num_of_downstream_beneficiaries_per_pixel_ph_downstream_dspop_benes_md5_028732.tif",
         DSPOP_SERVICE_SEDIMENT_PH_CONSERVATION_INF_SSP245),
        (DIFF_RECHARGE_PH_CONSERVATION_INF_SSP245,
         r"D:\repositories\wwf-sipa\downstream_beneficiary_workspace\num_of_downstream_beneficiaries_per_pixel_ph_downstream_road2019_benes_md5_870a6c.tif",
         ROAD_SERVICE_RECHARGE_PH_CONSERVATION_INF_SSP245),
        (DIFF_FLOOD_MITIGATION_PH_CONSERVATION_INF_SSP245,
         r"D:\repositories\wwf-sipa\downstream_beneficiary_workspace\num_of_downstream_beneficiaries_per_pixel_ph_downstream_road2019_benes_md5_870a6c.tif",
         ROAD_SERVICE_FLOOD_MITIGATION_PH_CONSERVATION_INF_SSP245),
        (DIFF_SEDIMENT_PH_CONSERVATION_INF_SSP245,
         r"D:\repositories\wwf-sipa\downstream_beneficiary_workspace\num_of_downstream_beneficiaries_per_pixel_ph_downstream_road2019_benes_md5_870a6c.tif",
         ROAD_SERVICE_SEDIMENT_PH_CONSERVATION_INF_SSP245)
        ]

    # prevent add and mult from doing it
    add_output_set = set([t[-1] for t in ADD_RASTER_SET])
    multiply_output_set = set([t[-1] for t in MULTIPLY_RASTER_SET])
    subtract_output_set = set([t[-1] for t in SUBTRACT_RASTER_SET])
    path_count = collections.defaultdict(int)
    for p in MULTIPLY_RASTER_SET:
        path = p[2]
        path_count[path] += 1
        if path_count[path] > 1:
            print(f'duplicate: {path}')

    print(len(add_output_set) == len(ADD_RASTER_SET))
    print(len(subtract_output_set) == len(SUBTRACT_RASTER_SET))
    print(len(multiply_output_set) == len(MULTIPLY_RASTER_SET))
    print(len(set(MULTIPLY_RASTER_SET)) == len(MULTIPLY_RASTER_SET))

    for raster_path_list_plus_target in \
            ADD_RASTER_SET+SUBTRACT_RASTER_SET+MULTIPLY_RASTER_SET:
        for p in raster_path_list_plus_target[:-1]:
            if not os.path.exists(p) and RESULTS_DIR not in p:
                print(f'input path does not exist: {p}')

    task_graph = taskgraph.TaskGraph(RESULTS_DIR, os.cpu_count()//2, 15.0)

    service_raster_path_list = []
    task_set = {}
    for raster_path_list_plus_target, op_str in (
            [(path_set, '+') for path_set in ADD_RASTER_SET] +
            [(path_set, '*') for path_set in MULTIPLY_RASTER_SET] +
            [(path_set, '-') for path_set in SUBTRACT_RASTER_SET]
            ):
        dependent_task_list = []
        target_raster_path = raster_path_list_plus_target[-1]
        input_rasters = raster_path_list_plus_target[:-1]
        for p in input_rasters:
            if p in task_set:
                dependent_task_list.append(task_set[p])
        if op_str not in ['+', '*']:
            op_task = task_graph.add_task(
                func=raster_op,
                args=(op_str, input_rasters, target_raster_path),
                target_path_list=[target_raster_path],
                dependent_task_list=dependent_task_list,
                task_name=f'calcualte {target_raster_path}')
            if target_raster_path in task_set:
                raise ValueError(f'calculating a result that we alreayd calculated {target_raster_path}')
            task_set[target_raster_path] = op_task
        if 'service' in target_raster_path:
            service_raster_path_list.append((target_raster_path, op_task))

    # ASK BCK: gte-75 gte-90 means top 25 top 10 so only 25 or 10% are selected
    # :::: call python mask_by_percentile.py D:\repositories\wwf-sipa\final_results\service_*.tif gte-75-percentile_[file_name]_gte75.tif gte-90-percentile_[file_name]_gte90.tif
    percentile_task_list = []
    for service_path, service_task in service_raster_path_list:
        percentile_task = task_graph.add_task(
            func=make_top_nth_percentile_masks,
            args=(
                service_path,
                top_percentile_list,
                os.path.join(RESULTS_DIR, 'top_{percentile}th_percentile_' + os.path.basename(service_path))),
            dependent_task_list=[service_task],
            store_result=True,
            task_name=f'percentile for {service_path}')
        percentile_task_list.append((service_path, percentile_task))
    task_graph.join()

    percentile_raster_list = []
    for service_path, percentile_task in percentile_task_list:
        local_percentile_rasters = percentile_task.get()
        LOGGER.info(f'percentile for {service_path} is {percentile_task.get()}')
        percentile_raster_list.extend(local_percentile_rasters)

    # if there are any percentile rasters that are with and without a climate ID then collapse those into a single raster
    future_climate_scenario_id = climate_list[0]
    resilient_task_list = []
    for local_percentile_raster in list(percentile_raster_list):
        if local_percentile_raster.endswith(
                f'{future_climate_scenario_id}.tif'):
            # we need to collapose into climate resilient
            base_local_percentile_raster = local_percentile_raster.replace(
                f'_{future_climate_scenario_id}', '')
            LOGGER.debug(f'*** collapse {local_percentile_raster} into {base_local_percentile_raster}')
            percentile_raster_list.remove(local_percentile_raster)
            percentile_raster_list.remove(base_local_percentile_raster)
            resilient_raster_path = os.path.join(
                CLIMATE_RESILIENT_PERCENTILES,
                os.path.basename(local_percentile_raster))
            percentile_raster_list.append(resilient_raster_path)
            # only take the mask of both
            join_mask_task = task_graph.add_task(
                func=join_mask,
                args=(
                    local_percentile_raster, base_local_percentile_raster,
                    resilient_raster_path),
                target_path_list=[resilient_raster_path],
                task_name=f'join for {resilient_raster_path}')
            resilient_task_list.append(join_mask_task)

    percentile_groups = collections.defaultdict(list)
    LOGGER.debug(f'************ THESE ARE THE PERCNTILES {percentile_raster_list}')
    for percentile_raster_path in percentile_raster_list:
        index_substring = ''
        for substring_list in [top_percentile_list, country_list, scenario_list, beneficiary_list]:
            found = False
            for substring in substring_list:
                if str(substring).lower() in percentile_raster_path.lower():
                    index_substring += f'{substring}_'
                    found = True
                    break
            if not found:
                # shortcut for doing a single country
                continue
        percentile_groups[index_substring].append(percentile_raster_path)

    LOGGER.debug(f'these are the percentile groups: {list(percentile_groups.keys())}')
    for key, percentile_raster_group in percentile_groups.items():
        service_overlap_raster_path = os.path.join(RESULTS_DIR, f'{key}service_overlap_count.tif')
        _ = task_graph.add_task(
            func=add_rasters,
            args=(percentile_raster_group, service_overlap_raster_path, gdal.GDT_Byte),
            dependent_task_list=resilient_task_list,
            target_path_list=[service_overlap_raster_path],
            task_name=f'collect service count for {key}')

    task_graph.join()
    task_graph.close()
    LOGGER.info(f'all done! results in {RESULTS_DIR}')


if __name__ == '__main__':
    main()

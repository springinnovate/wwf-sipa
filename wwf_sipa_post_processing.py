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

RESULTS_DIR = 'D:\\repositories\\wwf-sipa\\post_processing_results_updated_R_worstcase_2024_11_11'
CLIMATE_RESILIENT_PERCENTILES = os.path.join(RESULTS_DIR, 'climate_resilient_results')
MASK_SUBSET_DIR = os.path.join(RESULTS_DIR, 'mask_service_subsets')
for dir_path in [RESULTS_DIR, CLIMATE_RESILIENT_PERCENTILES, MASK_SUBSET_DIR]:
    os.makedirs(dir_path, exist_ok=True)


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
        ['near'] * 2, pixel_size, 'intersection')

    pre_cog_target = os.path.join(aligned_dir, os.path.basename(joined_mask_path))
    geoprocessing.raster_calculator(
        [(path, 1) for path in aligned_target_raster_path_list], mask_op,
        pre_cog_target, gdal.GDT_Byte, target_nodata)
    subprocess.check_call(
        f'gdal_translate {pre_cog_target} {joined_mask_path} -of COG -co BIGTIFF=YES')
    shutil.rmtree(aligned_dir)


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
        raster_path_list, aligned_target_raster_path_list, ['near'] * len(raster_path_list),
        pixel_size, 'intersection')

    def _sum_op(*array_list):
        return numpy.sum(array_list, axis=0)

    pre_cog_target = os.path.join(working_dir, os.path.basename(target_raster_path))
    geoprocessing.raster_calculator(
        [(path, 1) for path in aligned_target_raster_path_list], _sum_op, pre_cog_target,
        target_datatype, 0,
        allow_different_blocksize=True)
    subprocess.check_call(
        f'gdal_translate {pre_cog_target} {target_raster_path} -of COG -co BIGTIFF=YES')
    shutil.rmtree(working_dir)


def make_top_nth_percentile_masks(
        base_raster_path, top_percentile_list, target_raster_path_pattern):
    """Mask base by mask such that any nodata in mask is set to nodata in base."""
    ordered_top_percentile_list = list(sorted(top_percentile_list, reverse=True))
    # need to convert this to "gte" format so if top 10th percent, we get 90th percentile
    raw_percentile_list = [100 - float(x) for x in ordered_top_percentile_list]
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
        ['near'] * len(base_raster_path_list), pixel_size, 'intersection')

    raster_info = geoprocessing.get_raster_info(base_raster_path_list[0])

    nodata_list = [
        geoprocessing.get_raster_info(path)['nodata'][0]
        for path in aligned_target_raster_path_list]
    if target_nodata is None:
        target_nodata = nodata_list[0]

    def _op(*array_list):
        final_valid_mask = numpy.zeros(array_list[0].shape, dtype=bool)
        if op_str == '-':
            result = array_list[0]
            array_list = array_list[1:]
        elif op_str == '+':
            result = numpy.zeros(array_list[0].shape)
        elif op_str == '*':
            result = numpy.ones(array_list[0].shape)
            final_valid_mask[:] = True  # all vals need to be defined for mult
        for array, nodata in zip(array_list, nodata_list):
            local_valid_mask = numpy.isfinite(array) & (array != 0)
            if nodata is not None:
                local_valid_mask &= (array != nodata)
            if op_str == '*':
                # if * then all values need to be defined
                final_valid_mask &= local_valid_mask
            else:
                # otherwise we treat nodata as 0
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

    # diff x benes x services (4) x scenarios (2) x climage (2)
    country_list = ['PH', 'IDN']
    # TODO: do i need to update this vector?
    country_vector_list = [
        ('PH', './data/admin_boundaries/PH_gdam2.gpkg'),
        ('IDN', './data/admin_boundaries/IDN_adm1.gpkg'),
    ]

    scenario_list = ['restoration', 'conservation_inf']
    climate_list = ['ssp245']
    beneficiary_list = ['dspop', 'road']
    top_percentile_list = [10]

    ADMIN_POLYGONS = {
        'PH': [
            "./data/admin_boundaries/ph_visayas.gpkg",
            "./data/admin_boundaries/ph_luzon.gpkg",
            "./data/admin_boundaries/ph_mindanao.gpkg",
        ],
        'IDN': [
            "./data/admin_boundaries/idn_java.gpkg",
            "./data/admin_boundaries/idn_kalimantan.gpkg",
            "./data/admin_boundaries/idn_maluku_islands.gpkg",
            "./data/admin_boundaries/idn_nusa_tenggara.gpkg",
            "./data/admin_boundaries/idn_paupa.gpkg",
            "./data/admin_boundaries/idn_sulawesi.gpkg",
            "./data/admin_boundaries/idn_sumatra.gpkg",
        ]
    }

    # group by [operation] [country] [scenario]

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

    FLOOD_MITIGATION_IDN_BASELINE_HISTORICAL_CLIMATE = os.path.join(RESULTS_DIR, "flood_mitigation_IDN_baseline_historical_climate.tif")
    FLOOD_MITIGATION_PH_BASELINE_HISTORICAL_CLIMATE = os.path.join(RESULTS_DIR, "flood_mitigation_PH_baseline_historical_climate.tif")

    DIFF_SEDIMENT_IDN_CONSERVATION_ALL = os.path.join(RESULTS_DIR, 'diff_sediment_IDN_conservation_all.tif')
    DIFF_SEDIMENT_IDN_CONSERVATION_ALL_SSP245 = os.path.join(RESULTS_DIR, "diff_sediment_IDN_conservation_all_ssp245.tif")
    DIFF_SEDIMENT_IDN_CONSERVATION_INF = os.path.join(RESULTS_DIR, 'diff_sediment_IDN_conservation_inf.tif')
    DIFF_SEDIMENT_IDN_CONSERVATION_INF_SSP245 = os.path.join(RESULTS_DIR, "diff_sediment_IDN_conservation_inf_ssp245.tif")
    DIFF_SEDIMENT_IDN_RESTORATION = os.path.join(RESULTS_DIR, 'diff_sediment_IDN_restoration.tif')
    DIFF_SEDIMENT_IDN_RESTORATION_SSP245 = os.path.join(RESULTS_DIR, "diff_sediment_IDN_restoration_ssp245.tif")

    DIFF_SEDIMENT_PH_CONSERVATION_ALL = os.path.join(RESULTS_DIR, 'diff_sediment_PH_conservation_all.tif')
    DIFF_SEDIMENT_PH_CONSERVATION_ALL_SSP245 = os.path.join(RESULTS_DIR, "diff_sediment_PH_conservation_all_ssp245.tif")
    DIFF_SEDIMENT_PH_CONSERVATION_INF = os.path.join(RESULTS_DIR, 'diff_sediment_PH_conservation_inf.tif')
    DIFF_SEDIMENT_PH_CONSERVATION_INF_SSP245 = os.path.join(RESULTS_DIR, "diff_sediment_PH_conservation_inf_ssp245.tif")
    DIFF_SEDIMENT_PH_RESTORATION = os.path.join(RESULTS_DIR, 'diff_sediment_PH_restoration.tif')
    DIFF_SEDIMENT_PH_RESTORATION_SSP245 = os.path.join(RESULTS_DIR, "diff_sediment_PH_restoration_ssp245.tif")

    # TODO: Make "alls" here:
    DSPOP_SERVICE_FLOOD_MITIGATION_IDN_CONSERVATION_INF = os.path.join(RESULTS_DIR, "service_dspop_flood_mitigation_IDN_conservation_inf.tif")
    DSPOP_SERVICE_FLOOD_MITIGATION_IDN_RESTORATION = os.path.join(RESULTS_DIR, "service_dspop_flood_mitigation_IDN_restoration.tif")
    DSPOP_SERVICE_FLOOD_MITIGATION_PH_RESTORATION = os.path.join(RESULTS_DIR, "service_dspop_flood_mitigation_PH_restoration.tif")
    DSPOP_SERVICE_FLOOD_MITIGATION_PH_CONSERVATION_INF = os.path.join(RESULTS_DIR, "service_dspop_flood_mitigation_PH_conservation_inf.tif")

    ROAD_SERVICE_FLOOD_MITIGATION_IDN_CONSERVATION_INF = os.path.join(RESULTS_DIR, "service_road_flood_mitigation_IDN_conservation_inf.tif")
    ROAD_SERVICE_FLOOD_MITIGATION_IDN_RESTORATION = os.path.join(RESULTS_DIR, "service_road_flood_mitigation_IDN_restoration.tif")
    ROAD_SERVICE_FLOOD_MITIGATION_PH_CONSERVATION_INF = os.path.join(RESULTS_DIR, "service_road_flood_mitigation_PH_conservation_inf.tif")
    ROAD_SERVICE_FLOOD_MITIGATION_PH_RESTORATION = os.path.join(RESULTS_DIR, "service_road_flood_mitigation_PH_restoration.tif")

    DSPOP_SERVICE_RECHARGE_IDN_CONSERVATION_INF = os.path.join(RESULTS_DIR, "service_dspop_recharge_IDN_conservation_inf.tif")
    DSPOP_SERVICE_RECHARGE_IDN_RESTORATION = os.path.join(RESULTS_DIR, "service_dspop_recharge_IDN_restoration.tif")
    DSPOP_SERVICE_RECHARGE_PH_CONSERVATION_INF = os.path.join(RESULTS_DIR, "service_dspop_recharge_PH_conservation_inf.tif")
    DSPOP_SERVICE_RECHARGE_PH_RESTORATION = os.path.join(RESULTS_DIR, "service_dspop_recharge_PH_restoration.tif")

    DSPOP_SERVICE_SEDIMENT_IDN_CONSERVATION_ALL = os.path.join(RESULTS_DIR, "service_dspop_sediment_IDN_conservation_all.tif")
    DSPOP_SERVICE_SEDIMENT_IDN_CONSERVATION_INF = os.path.join(RESULTS_DIR, "service_dspop_sediment_IDN_conservation_inf.tif")
    DSPOP_SERVICE_SEDIMENT_IDN_RESTORATION = os.path.join(RESULTS_DIR, "service_dspop_sediment_IDN_restoration.tif")

    DSPOP_SERVICE_SEDIMENT_PH_CONSERVATION_ALL = os.path.join(RESULTS_DIR, "service_dspop_sediment_PH_conservation_all.tif")
    DSPOP_SERVICE_SEDIMENT_PH_CONSERVATION_INF = os.path.join(RESULTS_DIR, "service_dspop_sediment_PH_conservation_inf.tif")
    DSPOP_SERVICE_SEDIMENT_PH_RESTORATION = os.path.join(RESULTS_DIR, "service_dspop_sediment_PH_restoration.tif")

    ROAD_SERVICE_SEDIMENT_IDN_CONSERVATION_ALL = os.path.join(RESULTS_DIR, "service_road_sediment_IDN_conservation_all.tif")
    ROAD_SERVICE_SEDIMENT_IDN_CONSERVATION_INF = os.path.join(RESULTS_DIR, "service_road_sediment_IDN_conservation_inf.tif")
    ROAD_SERVICE_SEDIMENT_IDN_RESTORATION = os.path.join(RESULTS_DIR, "service_road_sediment_IDN_restoration.tif")

    ROAD_SERVICE_SEDIMENT_PH_CONSERVATION_ALL = os.path.join(RESULTS_DIR, "service_road_sediment_PH_conservation_all.tif")
    ROAD_SERVICE_SEDIMENT_PH_CONSERVATION_INF = os.path.join(RESULTS_DIR, "service_road_sediment_PH_conservation_inf.tif")
    ROAD_SERVICE_SEDIMENT_PH_RESTORATION = os.path.join(RESULTS_DIR, "service_road_sediment_PH_restoration.tif")

    # need all
    DSPOP_SERVICE_FLOOD_MITIGATION_IDN_CONSERVATION_INF_SSP245 = os.path.join(RESULTS_DIR, "service_dspop_flood_mitigation_IDN_conservation_inf_ssp245.tif")
    DSPOP_SERVICE_FLOOD_MITIGATION_IDN_RESTORATION_SSP245 = os.path.join(RESULTS_DIR, "service_dspop_flood_mitigation_IDN_restoration_ssp245.tif")

    # need all
    DSPOP_SERVICE_FLOOD_MITIGATION_PH_CONSERVATION_INF_SSP245 = os.path.join(RESULTS_DIR, "service_dspop_flood_mitigation_PH_conservation_inf_ssp245.tif")
    DSPOP_SERVICE_FLOOD_MITIGATION_PH_RESTORATION_SSP245 = os.path.join(RESULTS_DIR, "service_dspop_flood_mitigation_PH_restoration_ssp245.tif")

    # need all
    DSPOP_SERVICE_RECHARGE_IDN_CONSERVATION_INF_SSP245 = os.path.join(RESULTS_DIR, "service_dspop_recharge_IDN_conservation_inf_ssp245.tif")
    DSPOP_SERVICE_RECHARGE_IDN_RESTORATION_SSP245 = os.path.join(RESULTS_DIR, "service_dspop_recharge_IDN_restoration_ssp245.tif")

    # need all
    DSPOP_SERVICE_RECHARGE_PH_CONSERVATION_INF_SSP245 = os.path.join(RESULTS_DIR, "service_dspop_recharge_PH_conservation_inf_ssp245.tif")
    DSPOP_SERVICE_RECHARGE_PH_RESTORATION_SSP245 = os.path.join(RESULTS_DIR, "service_dspop_recharge_PH_restoration_ssp245.tif")

    DSPOP_SERVICE_SEDIMENT_IDN_CONSERVATION_ALL_SSP245 = os.path.join(RESULTS_DIR, "service_dspop_sediment_IDN_conservation_all_ssp245.tif")
    DSPOP_SERVICE_SEDIMENT_IDN_CONSERVATION_INF_SSP245 = os.path.join(RESULTS_DIR, "service_dspop_sediment_IDN_conservation_inf_ssp245.tif")
    DSPOP_SERVICE_SEDIMENT_IDN_RESTORATION_SSP245 = os.path.join(RESULTS_DIR, "service_dspop_sediment_IDN_restoration_ssp245.tif")

    DSPOP_SERVICE_SEDIMENT_PH_CONSERVATION_ALL_SSP245 = os.path.join(RESULTS_DIR, "service_dspop_sediment_PH_conservation_all_ssp245.tif")
    DSPOP_SERVICE_SEDIMENT_PH_CONSERVATION_INF_SSP245 = os.path.join(RESULTS_DIR, "service_dspop_sediment_PH_conservation_inf_ssp245.tif")
    DSPOP_SERVICE_SEDIMENT_PH_RESTORATION_SSP245 = os.path.join(RESULTS_DIR, "service_dspop_sediment_PH_restoration_ssp245.tif")

    ROAD_SERVICE_FLOOD_MITIGATION_IDN_CONSERVATION_INF_SSP245 = os.path.join(RESULTS_DIR, "service_road_flood_mitigation_IDN_conservation_inf_ssp245.tif")
    ROAD_SERVICE_FLOOD_MITIGATION_IDN_RESTORATION_SSP245 = os.path.join(RESULTS_DIR, "service_road_flood_mitigation_IDN_restoration_ssp245.tif")

    ROAD_SERVICE_FLOOD_MITIGATION_PH_CONSERVATION_INF_SSP245 = os.path.join(RESULTS_DIR, "service_road_flood_mitigation_PH_conservation_inf_ssp245.tif")
    ROAD_SERVICE_FLOOD_MITIGATION_PH_RESTORATION_SSP245 = os.path.join(RESULTS_DIR, "service_road_flood_mitigation_PH_restoration_ssp245.tif")

    ROAD_SERVICE_SEDIMENT_IDN_CONSERVATION_ALL_SSP245 = os.path.join(RESULTS_DIR, "service_road_sediment_IDN_conservation_all_ssp245.tif")
    ROAD_SERVICE_SEDIMENT_IDN_CONSERVATION_INF_SSP245 = os.path.join(RESULTS_DIR, "service_road_sediment_IDN_conservation_inf_ssp245.tif")
    ROAD_SERVICE_SEDIMENT_IDN_RESTORATION_SSP245 = os.path.join(RESULTS_DIR, "service_road_sediment_IDN_restoration_ssp245.tif")

    ROAD_SERVICE_SEDIMENT_PH_CONSERVATION_ALL_SSP245 = os.path.join(RESULTS_DIR, "service_road_sediment_PH_conservation_all_ssp245.tif")
    ROAD_SERVICE_SEDIMENT_PH_CONSERVATION_INF_SSP245 = os.path.join(RESULTS_DIR, "service_road_sediment_PH_conservation_inf_ssp245.tif")
    ROAD_SERVICE_SEDIMENT_PH_RESTORATION_SSP245 = os.path.join(RESULTS_DIR, "service_road_sediment_PH_restoration_ssp245.tif")

    DS_POP_SERVICE_CV_IDN_CONSERVATION_INF_RESULT = os.path.join(RESULTS_DIR, 'service_dspop_cv_idn_conservation_inf_result.tif')
    DS_POP_SERVICE_CV_IDN_RESTORATION_RESULT = os.path.join(RESULTS_DIR, 'service_dspop_cv_idn_restoration_result.tif')

    ROAD_SERVICE_CV_IDN_CONSERVATION_INF_RESULT = os.path.join(RESULTS_DIR, 'service_road_cv_idn_conservation_inf_result.tif')
    ROAD_SERVICE_CV_IDN_RESTORATION_RESULT = os.path.join(RESULTS_DIR, 'service_road_cv_idn_restoration_result.tif')

    DS_POP_SERVICE_CV_PH_CONSERVATION_INF_RESULT = os.path.join(RESULTS_DIR, 'service_dspop_cv_ph_conservation_inf_result.tif')
    DS_POP_SERVICE_CV_PH_RESTORATION_RESULT = os.path.join(RESULTS_DIR, 'service_dspop_cv_ph_restoration_result.tif')

    ROAD_SERVICE_CV_PH_CONSERVATION_INF_RESULT = os.path.join(RESULTS_DIR, 'service_road_cv_ph_conservation_inf_result.tif')
    ROAD_SERVICE_CV_PH_RESTORATION_RESULT = os.path.join(RESULTS_DIR, 'service_road_cv_ph_restoration_result.tif')

    # service first then beneficiary after

    # CHECK W BCK:
    # x diff between DSPOP and ROAD 'service_...' output files
    # x check that "diff" is an output to a multiply and that the filename makes sense
    ADD_RASTER_SET = [
        (r"D:\repositories\coastal_risk_reduction\workspace\IDN_baseline\value_rasters\forest_mangrove_population_less_than_2m_value_index.tif",
         r"D:\repositories\coastal_risk_reduction\workspace\IDN_baseline\value_rasters\forest_mangroves_population_less_than_2m_value_index.tif",
         r"D:\repositories\coastal_risk_reduction\workspace\IDN_baseline\value_rasters\reefs_population_less_than_2m_value_index.tif",
         r"D:\repositories\coastal_risk_reduction\workspace\IDN_baseline\value_rasters\saltmarsh_population_less_than_2m_value_index.tif",
         r"D:\repositories\coastal_risk_reduction\workspace\IDN_baseline\value_rasters\savanna_population_less_than_2m_value_index.tif",
         r"D:\repositories\coastal_risk_reduction\workspace\IDN_baseline\value_rasters\seagrass_population_less_than_2m_value_index.tif",
         r"D:\repositories\coastal_risk_reduction\workspace\IDN_baseline\value_rasters\secondary forest_population_less_than_2m_value_index.tif",
         r"D:\repositories\coastal_risk_reduction\workspace\IDN_baseline\value_rasters\woody_crop_population_less_than_2m_value_index.tif",
         DS_POP_SERVICE_CV_IDN_CONSERVATION_INF_RESULT
         ),
        (r"D:\repositories\coastal_risk_reduction\workspace\IDN_baseline\value_rasters\woody_crop_roads_within_15km_value_index.tif",
         r"D:\repositories\coastal_risk_reduction\workspace\IDN_baseline\value_rasters\forest_mangrove_roads_within_15km_value_index.tif",
         r"D:\repositories\coastal_risk_reduction\workspace\IDN_baseline\value_rasters\forest_mangroves_roads_within_15km_value_index.tif",
         r"D:\repositories\coastal_risk_reduction\workspace\IDN_baseline\value_rasters\reefs_roads_within_15km_value_index.tif",
         r"D:\repositories\coastal_risk_reduction\workspace\IDN_baseline\value_rasters\saltmarsh_roads_within_15km_value_index.tif",
         r"D:\repositories\coastal_risk_reduction\workspace\IDN_baseline\value_rasters\savanna_roads_within_15km_value_index.tif",
         r"D:\repositories\coastal_risk_reduction\workspace\IDN_baseline\value_rasters\seagrass_roads_within_15km_value_index.tif",
         r"D:\repositories\coastal_risk_reduction\workspace\IDN_baseline\value_rasters\secondary forest_roads_within_15km_value_index.tif",
         ROAD_SERVICE_CV_IDN_CONSERVATION_INF_RESULT
         ),
        (r"D:\repositories\coastal_risk_reduction\workspace\IDN_restoration\value_rasters\forest_mangrove_population_less_than_2m_value_index.tif",
         r"D:\repositories\coastal_risk_reduction\workspace\IDN_restoration\value_rasters\forest_mangroves_population_less_than_2m_value_index.tif",
         r"D:\repositories\coastal_risk_reduction\workspace\IDN_restoration\value_rasters\reefs_population_less_than_2m_value_index.tif",
         r"D:\repositories\coastal_risk_reduction\workspace\IDN_restoration\value_rasters\saltmarsh_population_less_than_2m_value_index.tif",
         r"D:\repositories\coastal_risk_reduction\workspace\IDN_restoration\value_rasters\savanna_population_less_than_2m_value_index.tif",
         r"D:\repositories\coastal_risk_reduction\workspace\IDN_restoration\value_rasters\seagrass_population_less_than_2m_value_index.tif",
         r"D:\repositories\coastal_risk_reduction\workspace\IDN_restoration\value_rasters\secondary forest_population_less_than_2m_value_index.tif",
         r"D:\repositories\coastal_risk_reduction\workspace\IDN_restoration\value_rasters\woody_crop_population_less_than_2m_value_index.tif",
         DS_POP_SERVICE_CV_IDN_RESTORATION_RESULT
         ),
        (r"D:\repositories\coastal_risk_reduction\workspace\IDN_restoration\value_rasters\woody_crop_roads_within_15km_value_index.tif",
         r"D:\repositories\coastal_risk_reduction\workspace\IDN_restoration\value_rasters\forest_mangrove_roads_within_15km_value_index.tif",
         r"D:\repositories\coastal_risk_reduction\workspace\IDN_restoration\value_rasters\forest_mangroves_roads_within_15km_value_index.tif",
         r"D:\repositories\coastal_risk_reduction\workspace\IDN_restoration\value_rasters\reefs_roads_within_15km_value_index.tif",
         r"D:\repositories\coastal_risk_reduction\workspace\IDN_restoration\value_rasters\saltmarsh_roads_within_15km_value_index.tif",
         r"D:\repositories\coastal_risk_reduction\workspace\IDN_restoration\value_rasters\savanna_roads_within_15km_value_index.tif",
         r"D:\repositories\coastal_risk_reduction\workspace\IDN_restoration\value_rasters\seagrass_roads_within_15km_value_index.tif",
         r"D:\repositories\coastal_risk_reduction\workspace\IDN_restoration\value_rasters\secondary forest_roads_within_15km_value_index.tif",
         ROAD_SERVICE_CV_IDN_RESTORATION_RESULT
         ),
        (r"D:\repositories\coastal_risk_reduction\workspace\ph_baseline\value_rasters\brush_population_less_than_2m_value_index.tif",
         r"D:\repositories\coastal_risk_reduction\workspace\ph_baseline\value_rasters\grassland_population_less_than_2m_value_index.tif",
         r"D:\repositories\coastal_risk_reduction\workspace\ph_baseline\value_rasters\perennial_crop_population_less_than_2m_value_index.tif",
         r"D:\repositories\coastal_risk_reduction\workspace\ph_baseline\value_rasters\forest_mangroves_population_less_than_2m_value_index.tif",
         r"D:\repositories\coastal_risk_reduction\workspace\ph_baseline\value_rasters\reefs_population_less_than_2m_value_index.tif",
         r"D:\repositories\coastal_risk_reduction\workspace\ph_baseline\value_rasters\saltmarsh_population_less_than_2m_value_index.tif",
         r"D:\repositories\coastal_risk_reduction\workspace\ph_baseline\value_rasters\seagrass_population_less_than_2m_value_index.tif",
         r"D:\repositories\coastal_risk_reduction\workspace\ph_baseline\value_rasters\secondary forest_population_less_than_2m_value_index.tif",
         DS_POP_SERVICE_CV_PH_CONSERVATION_INF_RESULT
         ),
        (r"D:\repositories\coastal_risk_reduction\workspace\ph_baseline\value_rasters\brush_roads_within_15km_value_index.tif",
         r"D:\repositories\coastal_risk_reduction\workspace\ph_baseline\value_rasters\grassland_roads_within_15km_value_index.tif",
         r"D:\repositories\coastal_risk_reduction\workspace\ph_baseline\value_rasters\perennial_crop_roads_within_15km_value_index.tif",
         r"D:\repositories\coastal_risk_reduction\workspace\ph_baseline\value_rasters\forest_mangroves_roads_within_15km_value_index.tif",
         r"D:\repositories\coastal_risk_reduction\workspace\ph_baseline\value_rasters\reefs_roads_within_15km_value_index.tif",
         r"D:\repositories\coastal_risk_reduction\workspace\ph_baseline\value_rasters\saltmarsh_roads_within_15km_value_index.tif",
         r"D:\repositories\coastal_risk_reduction\workspace\ph_baseline\value_rasters\seagrass_roads_within_15km_value_index.tif",
         r"D:\repositories\coastal_risk_reduction\workspace\ph_baseline\value_rasters\secondary forest_roads_within_15km_value_index.tif",
         ROAD_SERVICE_CV_PH_CONSERVATION_INF_RESULT
         ),
        (r"D:\repositories\coastal_risk_reduction\workspace\ph_restoration\value_rasters\brush_population_less_than_2m_value_index.tif",
         r"D:\repositories\coastal_risk_reduction\workspace\ph_restoration\value_rasters\grassland_population_less_than_2m_value_index.tif",
         r"D:\repositories\coastal_risk_reduction\workspace\ph_restoration\value_rasters\perennial_crop_population_less_than_2m_value_index.tif",
         r"D:\repositories\coastal_risk_reduction\workspace\ph_restoration\value_rasters\forest_mangroves_population_less_than_2m_value_index.tif",
         r"D:\repositories\coastal_risk_reduction\workspace\ph_restoration\value_rasters\reefs_population_less_than_2m_value_index.tif",
         r"D:\repositories\coastal_risk_reduction\workspace\ph_restoration\value_rasters\saltmarsh_population_less_than_2m_value_index.tif",
         r"D:\repositories\coastal_risk_reduction\workspace\ph_restoration\value_rasters\seagrass_population_less_than_2m_value_index.tif",
         r"D:\repositories\coastal_risk_reduction\workspace\ph_restoration\value_rasters\secondary forest_population_less_than_2m_value_index.tif",
         DS_POP_SERVICE_CV_PH_RESTORATION_RESULT
         ),
        (r"D:\repositories\coastal_risk_reduction\workspace\ph_restoration\value_rasters\brush_roads_within_15km_value_index.tif",
         r"D:\repositories\coastal_risk_reduction\workspace\ph_restoration\value_rasters\grassland_roads_within_15km_value_index.tif",
         r"D:\repositories\coastal_risk_reduction\workspace\ph_restoration\value_rasters\perennial_crop_roads_within_15km_value_index.tif",
         r"D:\repositories\coastal_risk_reduction\workspace\ph_restoration\value_rasters\forest_mangroves_roads_within_15km_value_index.tif",
         r"D:\repositories\coastal_risk_reduction\workspace\ph_restoration\value_rasters\reefs_roads_within_15km_value_index.tif",
         r"D:\repositories\coastal_risk_reduction\workspace\ph_restoration\value_rasters\saltmarsh_roads_within_15km_value_index.tif",
         r"D:\repositories\coastal_risk_reduction\workspace\ph_restoration\value_rasters\seagrass_roads_within_15km_value_index.tif",
         r"D:\repositories\coastal_risk_reduction\workspace\ph_restoration\value_rasters\secondary forest_roads_within_15km_value_index.tif",
         ROAD_SERVICE_CV_PH_RESTORATION_RESULT
         ),
    ]

    # for the template, DIFF_SEDIMENT_IDN_CONSERVATION_ALL should look like DIFF_SEDIMENT_IDN_CONSERVATION_INF
    # we subtract the bad one minus the good one for sediment retention and quickflow, but the good one minus the bad one for recharge
    # and in this case restoration is "good" compared to baseline and infra or worstcase is "bad" compared to baseline
    SUBTRACT_RASTER_SET = [
        (
            r"D:\repositories\ndr_sdr_global\wwf_IDN_baseline_historical_climate\stitched_sed_export_wwf_IDN_baseline_historical_climate.tif",
            r"D:\repositories\ndr_sdr_global\wwf_IDN_restoration_historical_climate\stitched_sed_export_wwf_IDN_restoration_historical_climate.tif",
            DIFF_SEDIMENT_IDN_RESTORATION
        ),
        (
            r"D:\repositories\ndr_sdr_global\wwf_IDN_baseline_ssp245_climate\stitched_sed_export_wwf_IDN_baseline_ssp245_climate.tif",
            r"D:\repositories\ndr_sdr_global\wwf_IDN_restoration_ssp245_climate\stitched_sed_export_wwf_IDN_restoration_ssp245_climate.tif",
            DIFF_SEDIMENT_IDN_RESTORATION_SSP245
        ),
        (
            r"D:\repositories\ndr_sdr_global\wwf_IDN_infra_historical_climate\stitched_sed_export_wwf_IDN_infra_historical_climate.tif",
            r"D:\repositories\ndr_sdr_global\wwf_IDN_baseline_historical_climate\stitched_sed_export_wwf_IDN_baseline_historical_climate.tif",
            DIFF_SEDIMENT_IDN_CONSERVATION_INF
        ),
        (
            r"D:\repositories\ndr_sdr_global\wwf_IDN_infra_ssp245_climate\stitched_sed_export_wwf_IDN_infra_ssp245_climate.tif",
            r"D:\repositories\ndr_sdr_global\wwf_IDN_baseline_ssp245_climate\stitched_sed_export_wwf_IDN_baseline_ssp245_climate.tif",
            DIFF_SEDIMENT_IDN_CONSERVATION_INF_SSP245
        ),
        (
            r"D:\repositories\ndr_sdr_global\wwf_IDN_worstcase_historical_climate\stitched_sed_export_wwf_IDN_worstcase_historical_climate.tif",
            r"D:\repositories\ndr_sdr_global\wwf_IDN_baseline_historical_climate\stitched_sed_export_wwf_IDN_baseline_historical_climate.tif",
            DIFF_SEDIMENT_IDN_CONSERVATION_ALL
        ),
        (
            # TODO -- likely all conservation_all_ssp245s are wrong, they need to compare their respective 245 scenarios
            #r"D:\repositories\ndr_sdr_global\wwf_IDN_worstcase_historical_climate\stitched_sed_export_wwf_IDN_worstcase_historical_climate.tif",
            r"D:\repositories\ndr_sdr_global\wwf_IDN_worstcase_historical_climate\stitched_sed_export_wwf_IDN_worstcase_ssp245_climate.tif",
            r"D:\repositories\ndr_sdr_global\wwf_IDN_baseline_ssp245_climate\stitched_sed_export_wwf_IDN_baseline_ssp245_climate.tif",
            DIFF_SEDIMENT_IDN_CONSERVATION_ALL_SSP245
        ),
        (
            r"D:\repositories\ndr_sdr_global\wwf_PH_baseline_historical_climate\stitched_sed_export_wwf_PH_baseline_historical_climate.tif",
            r"D:\repositories\ndr_sdr_global\wwf_PH_restoration_historical_climate\stitched_sed_export_wwf_PH_restoration_historical_climate.tif",
            DIFF_SEDIMENT_PH_RESTORATION
        ),
        (
            r"D:\repositories\ndr_sdr_global\wwf_PH_baseline_ssp245_climate\stitched_sed_export_wwf_PH_baseline_ssp245_climate.tif",
            r"D:\repositories\ndr_sdr_global\wwf_PH_restoration_ssp245_climate\stitched_sed_export_wwf_PH_restoration_ssp245_climate.tif",
            DIFF_SEDIMENT_PH_RESTORATION_SSP245
        ),
        (
            r"D:\repositories\ndr_sdr_global\wwf_PH_infra_historical_climate\stitched_sed_export_wwf_PH_infra_historical_climate.tif",
            r"D:\repositories\ndr_sdr_global\wwf_PH_baseline_historical_climate\stitched_sed_export_wwf_PH_baseline_historical_climate.tif",
            DIFF_SEDIMENT_PH_CONSERVATION_INF
        ),
        (
            r"D:\repositories\ndr_sdr_global\wwf_PH_infra_ssp245_climate\stitched_sed_export_wwf_PH_infra_ssp245_climate.tif",
            r"D:\repositories\ndr_sdr_global\wwf_PH_baseline_ssp245_climate\stitched_sed_export_wwf_PH_baseline_ssp245_climate.tif",
            DIFF_SEDIMENT_PH_CONSERVATION_INF_SSP245
        ),
        (
            r"D:\repositories\ndr_sdr_global\wwf_PH_worstcase_historical_climate\stitched_sed_export_wwf_PH_worstcase_historical_climate.tif",
            r"D:\repositories\ndr_sdr_global\wwf_PH_baseline_historical_climate\stitched_sed_export_wwf_PH_baseline_historical_climate.tif",
            DIFF_SEDIMENT_PH_CONSERVATION_ALL
        ),
        (
            r"D:\repositories\ndr_sdr_global\wwf_PH_worstcase_historical_climate\stitched_sed_export_wwf_PH_worstcase_historical_climate.tif",
            r"D:\repositories\ndr_sdr_global\wwf_PH_baseline_ssp245_climate\stitched_sed_export_wwf_PH_baseline_ssp245_climate.tif",
            DIFF_SEDIMENT_PH_CONSERVATION_ALL_SSP245
        ),

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
        # THIS NEEDS TO BE ADDED AFTER ALL CONSERVATION_INF - FOLLOW THE SAME PATTERN FOR RECHARGE
        # -- recharge is "good" so we subtract worstcase from baseline
        #(r"D:\repositories\swy_global\workspace_swy_wwf_IDN_baseline_historical_climate\B_wwf_IDN_baseline_historical_climate.tif",
        # r"D:\repositories\swy_global\workspace_swy_wwf_IDN_worstcase_historical_climate\B_wwf_IDN_worstcase_historical_climate.tif",
         #DIFF_RECHARGE_IDN_CONSERVATION_ALL),
        (r"D:\repositories\swy_global\workspace_swy_wwf_IDN_infra_historical_climate\QF_wwf_IDN_infra_historical_climate.tif",
         r"D:\repositories\swy_global\workspace_swy_wwf_IDN_baseline_historical_climate\QF_wwf_IDN_baseline_historical_climate.tif",
         DIFF_QUICKFLOW_IDN_CONSERVATION_INF),
        # THIS NEEDS TO BE ADDED AFTER ALL CONSERVATION_INF - FOLLOW THE SAME PATTERN FOR QUICKFLOW
        # -- quickflow is "bad" so we subtract baseline from worstcase
        # (r"D:\repositories\swy_global\workspace_swy_wwf_IDN_worstcase_historical_climate\QF_wwf_IDN_worstcase_historical_climate.tif",
        # r"D:\repositories\swy_global\workspace_swy_wwf_IDN_baseline_historical_climate\QF_wwf_IDN_baseline_historical_climate.tif",
        # DIFF_QUICKFLOW_IDN_CONSERVATION_ALL),
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
        # The following missing variables are because we changed the name from "flood mitigation", to 'flood' so it looks like 'sediment'
        (r"D:\repositories\wwf-sipa\idn_downstream_flood_risk.tif",
         r"D:\repositories\swy_global\workspace_swy_wwf_IDN_baseline_historical_climate\QF_wwf_IDN_baseline_historical_climate.tif",
         FLOOD_IDN_BASELINE_HISTORICAL_CLIMATE),
        # this used to be called "DIFF_FLOOD_MITIGATION' but it's not mitigation, it's just diff flood, changing so it looks like "DIFF_SEDIMENT...."
        (r"D:\repositories\wwf-sipa\idn_downstream_flood_risk.tif",
         DIFF_QUICKFLOW_IDN_CONSERVATION_INF,
         DIFF_FLOOD_IDN_CONSERVATION_INF),
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
        (DIFF_FLOOD_MITIGATION_PH_CONSERVATION_INF_SSP245,
         r"D:\repositories\wwf-sipa\downstream_beneficiary_workspace\num_of_downstream_beneficiaries_per_pixel_ph_downstream_road2019_benes_md5_870a6c.tif",
         ROAD_SERVICE_FLOOD_MITIGATION_PH_CONSERVATION_INF_SSP245),
        (DIFF_SEDIMENT_PH_CONSERVATION_INF_SSP245,
         r"D:\repositories\wwf-sipa\downstream_beneficiary_workspace\num_of_downstream_beneficiaries_per_pixel_ph_downstream_road2019_benes_md5_870a6c.tif",
         ROAD_SERVICE_SEDIMENT_PH_CONSERVATION_INF_SSP245),
        (DIFF_SEDIMENT_IDN_CONSERVATION_ALL,
         r"D:\repositories\wwf-sipa\downstream_beneficiary_workspace\num_of_downstream_beneficiaries_per_pixel_idn_downstream_dspop_benes_md5_e4d2c4.tif",
         DSPOP_SERVICE_SEDIMENT_IDN_CONSERVATION_ALL),
        (DIFF_SEDIMENT_IDN_CONSERVATION_ALL,
         r"D:\repositories\wwf-sipa\downstream_beneficiary_workspace\num_of_downstream_beneficiaries_per_pixel_idn_downstream_road2019_benes_md5_8ec2cd.tif",
         ROAD_SERVICE_SEDIMENT_IDN_CONSERVATION_ALL),
        (DIFF_SEDIMENT_PH_CONSERVATION_ALL,
         r"D:\repositories\wwf-sipa\downstream_beneficiary_workspace\num_of_downstream_beneficiaries_per_pixel_ph_downstream_dspop_benes_md5_028732.tif",
         DSPOP_SERVICE_SEDIMENT_PH_CONSERVATION_ALL),
        (DIFF_SEDIMENT_PH_CONSERVATION_ALL,
         r"D:\repositories\wwf-sipa\downstream_beneficiary_workspace\num_of_downstream_beneficiaries_per_pixel_ph_downstream_road2019_benes_md5_870a6c.tif",
         ROAD_SERVICE_SEDIMENT_PH_CONSERVATION_ALL),
        (DIFF_SEDIMENT_IDN_CONSERVATION_ALL_SSP245,
         r"D:\repositories\wwf-sipa\downstream_beneficiary_workspace\num_of_downstream_beneficiaries_per_pixel_idn_downstream_dspop_benes_md5_e4d2c4.tif",
         DSPOP_SERVICE_SEDIMENT_IDN_CONSERVATION_ALL_SSP245),
        (DIFF_SEDIMENT_IDN_CONSERVATION_ALL_SSP245,
         r"D:\repositories\wwf-sipa\downstream_beneficiary_workspace\num_of_downstream_beneficiaries_per_pixel_idn_downstream_road2019_benes_md5_8ec2cd.tif",
         ROAD_SERVICE_SEDIMENT_IDN_CONSERVATION_ALL_SSP245),
        (DIFF_SEDIMENT_PH_CONSERVATION_ALL_SSP245,
         r"D:\repositories\wwf-sipa\downstream_beneficiary_workspace\num_of_downstream_beneficiaries_per_pixel_ph_downstream_dspop_benes_md5_028732.tif",
         DSPOP_SERVICE_SEDIMENT_PH_CONSERVATION_ALL_SSP245),
        (DIFF_SEDIMENT_PH_CONSERVATION_ALL_SSP245,
         r"D:\repositories\wwf-sipa\downstream_beneficiary_workspace\num_of_downstream_beneficiaries_per_pixel_ph_downstream_road2019_benes_md5_870a6c.tif",
         ROAD_SERVICE_SEDIMENT_PH_CONSERVATION_ALL_SSP245)
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

    task_graph = taskgraph.TaskGraph(RESULTS_DIR, os.cpu_count()//2+2, 15.0)

    service_raster_path_list = []
    task_set = {}
    for raster_path_list_plus_target, op_str in (
            [(path_set, '-') for path_set in SUBTRACT_RASTER_SET] +
            [(path_set, '+') for path_set in ADD_RASTER_SET] +
            [(path_set, '*') for path_set in MULTIPLY_RASTER_SET]):
        dependent_task_list = []
        target_raster_path = raster_path_list_plus_target[-1]
        input_rasters = raster_path_list_plus_target[:-1]
        for p in input_rasters:
            if p in task_set:
                dependent_task_list.append(task_set[p])
        op_task = task_graph.add_task(
            func=raster_op,
            args=(op_str, input_rasters, target_raster_path),
            target_path_list=[target_raster_path],
            dependent_task_list=dependent_task_list,
            task_name=f'calcualte {target_raster_path}')
        if target_raster_path in task_set:
            raise ValueError(f'calculating a result that we already calculated {target_raster_path}')
        task_set[target_raster_path] = op_task
        if 'service' in target_raster_path:
            service_raster_path_list.append((target_raster_path, op_task))

    percentile_task_list = []
    for service_path, service_task in service_raster_path_list:
        # for ADMIN_POLYGONS
        #   1) mask out service path for each polygon
        #   2) get the percentile for that service path
        #   3) add all the sub percentiles back together into one raster
        for country_code in ADMIN_POLYGONS:
            if f'{country_code.lower()}_' in service_path.lower():
                # this sets `country_code` to be the current country code
                break
        # country_code is the right country code now
        per_admin_task_list = []
        for admin_vector_path in ADMIN_POLYGONS[country_code]:
            # mask service_path to admin_vector_path
            admin_basename = os.path.splitext(os.path.basename(admin_vector_path))[0]
            masked_service_path = os.path.join(
                MASK_SUBSET_DIR, f'{admin_basename}_{os.path.basename(service_path)}')
            mask_task = task_graph.add_task(
                func=geoprocessing.mask_raster,
                args=(
                    (service_path, 1), admin_vector_path, masked_service_path),
                kwargs={
                    'working_dir': MASK_SUBSET_DIR,
                    'allow_different_blocksize': True},
                target_path_list=[masked_service_path],
                dependent_task_list=[service_task],
                task_name=f'mask {service_path} by {admin_vector_path}')

            percentile_task = task_graph.add_task(
                func=make_top_nth_percentile_masks,
                args=(
                    masked_service_path,
                    top_percentile_list,
                    os.path.join(RESULTS_DIR, 'top_{percentile}th_percentile_' + os.path.basename(masked_service_path))),
                dependent_task_list=[mask_task],
                store_result=True,
                task_name=f'percentile for {masked_service_path}')
            per_admin_task_list.append(percentile_task)
            # right here -> get this aggregated together into one path
        percentile_task_list.append((service_path, per_admin_task_list))

    task_graph.join()

    percentile_raster_list = []
    for service_path, per_admin_percentile_task_list in percentile_task_list:
        percentile_sets = collections.defaultdict(list)
        for local_admin_percentile_task in per_admin_percentile_task_list:
            # gather into each percentile
            for local_admin_service_path in local_admin_percentile_task.get():
                for percentile_value in top_percentile_list:
                    if f'top_{percentile_value}' in local_admin_service_path:
                        percentile_sets[percentile_value].append(local_admin_service_path)
        # add the sub-islands together
        for percentile_value in top_percentile_list:
            target_percentile_sum = os.path.join(
                RESULTS_DIR, f'top_{percentile_value}th_percentile_'+os.path.basename(service_path))
            task_graph.add_task(
                func=raster_op,
                args=('+', percentile_sets[percentile_value], target_percentile_sum),
                target_path_list=[target_percentile_sum],
                task_name=(
                    f're-sum the percentiles to {target_percentile_sum} for '
                    f'percentile {percentile_value} for these rasters: '
                    f'{percentile_sets[percentile_value]} on the original service raster of {service_path}'))
            percentile_raster_list.append(target_percentile_sum)
    task_graph.join()

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
    overlap_file_list = open('overlap.txt', 'w')
    for key, percentile_raster_group in percentile_groups.items():
        service_overlap_raster_path = os.path.join(RESULTS_DIR, f'{key}service_overlap_count.tif')
        _ = task_graph.add_task(
            func=add_rasters,
            args=(percentile_raster_group, service_overlap_raster_path, gdal.GDT_Byte),
            dependent_task_list=resilient_task_list,
            target_path_list=[service_overlap_raster_path],
            task_name=f'collect service count for {key}')
        overlap_file_list.write(f'{service_overlap_raster_path}:\n\t')
        overlap_file_list.write('\n\t'.join(percentile_raster_group))
        overlap_file_list.write('\n')
    overlap_file_list.close()

    task_graph.join()

    # build the stats by polygon
    for percentile_value in top_percentile_list:
        for country_id, country_aggregate_vector in country_vector_list:
            for scenario_id in scenario_list:
                target_vector = os.path.join(
                    RESULTS_DIR,
                    f'{percentile_value}_{country_id}_{scenario_id}_service_overlap_count.gpkg')
                base_raster_list = [
                    f'{RESULTS_DIR}/{percentile_value}_{country_id}_{scenario_id}_{beneficiary_id}_service_overlap_count.tif'
                    for beneficiary_id in beneficiary_list]
                task_graph.add_task(
                    func=zonal_stats,
                    args=(base_raster_list, country_aggregate_vector, target_vector),
                    target_path_list=[target_vector],
                    task_name=f'stats on {target_vector}')
    task_graph.join()
    task_graph.close()
    LOGGER.info(f'all done! results in {RESULTS_DIR}')


def local_zonal_stats(prefix, raster_path_list, aggregate_vector_path):
    working_dir = tempfile.mkdtemp(
        prefix='zonal_stats_', dir=os.path.dirname(aggregate_vector_path))
    summed_dir = os.path.join(
        RESULTS_DIR, 'summed_services')
    os.makedirs(summed_dir, exist_ok=True)
    fixed_raster_path = os.path.join(
        summed_dir, f'{prefix}.tif')
    sum_zero_to_nodata(raster_path_list, fixed_raster_path)
    stat_dict = geoprocessing.zonal_statistics(
        (fixed_raster_path, 1), aggregate_vector_path,
        working_dir=working_dir,
        clean_working_dir=True,
        polygons_might_overlap=False)
    return stat_dict


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

    working_dir = tempfile.mkdtemp(
        prefix='zero_to_nodata_', dir=os.path.dirname(target_raster_path))
    pre_cog_target = os.path.join(working_dir, os.path.basename(target_raster_path))

    geoprocessing.single_thread_raster_calculator(
        [(path, 1) for path in base_raster_path_list], _op, pre_cog_target,
        raster_info['datatype'], global_nodata)

    subprocess.check_call(
        f'gdal_translate {pre_cog_target} {target_raster_path} -of COG -co BIGTIFF=YES')
    shutil.rmtree(working_dir)


def zonal_stats(raster_path_list, aggregate_vector_path, target_vector_path):
    """Do zonal stats over base raster in each polygon of the vector."""
    basename = os.path.basename(os.path.splitext(target_vector_path)[0])
    stat_dict = local_zonal_stats(basename, raster_path_list, aggregate_vector_path)

    source_ds = ogr.Open(aggregate_vector_path, 0)
    source_layer = source_ds.GetLayer()
    driver = ogr.GetDriverByName("GPKG")

    # Create the target Geopackage
    if os.path.exists(target_vector_path):
        os.remove(target_vector_path)
    target_ds = driver.CreateDataSource(target_vector_path)

    # Create the target layer with the same schema as the source layer
    target_layer = target_ds.CreateLayer(basename, geom_type=source_layer.GetGeomType())
    target_layer.CreateFields(source_layer.schema)

    # Add two new floating point fields
    target_layer.CreateField(
        ogr.FieldDefn("proportional_service_area", ogr.OFTReal))
    target_layer.CreateField(
        ogr.FieldDefn("service_intensity", ogr.OFTReal))

    # Copy features from source layer to target layer and populate new fields
    source_layer.ResetReading()
    for feature in source_layer:
        new_feature = ogr.Feature(target_layer.GetLayerDefn())
        new_feature.SetFrom(feature)
        fid = feature.GetFID()
        try:
            proportional_service_area = stat_dict[fid]['count']/(stat_dict[fid]['count']+stat_dict[fid]['nodata_count'])
            service_intensity = stat_dict[fid]['sum']/stat_dict[fid]['count']
        except ZeroDivisionError:
            proportional_service_area = 0
            service_intensity = 0
        # Set values for the new fields (optional)
        new_feature.SetField(
            "proportional_service_area", proportional_service_area)
        new_feature.SetField(
            "service_intensity", service_intensity)
        target_layer.CreateFeature(new_feature)

    target_layer = None
    target_ds = None
    source_ds = None
    target_ds = None


if __name__ == '__main__':
    main()

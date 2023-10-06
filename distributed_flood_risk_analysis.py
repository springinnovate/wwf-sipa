"""Distributed flood risk analysis."""
import argparse
import hashlib
import os
import logging
import shutil
import sys
import tempfile

from osgeo import gdal
from ecoshard.geoprocessing import routing
from ecoshard import geoprocessing
from ecoshard import taskgraph
import numpy

NODATA = -1
GLOBAL_WORKSPACE_DIR = 'flood_risk_workspace'

logging.basicConfig(
    level=logging.DEBUG,
    stream=sys.stdout,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(levelname)s %(name)s'
        ' [%(funcName)s:%(lineno)d] %(message)s'))
LOGGER = logging.getLogger(os.path.splitext(os.path.basename(__file__))[0])
LOGGER.setLevel(logging.DEBUG)
logging.getLogger('ecoshard.fetch_data').setLevel(logging.INFO)


def div_op(num_array, den_array):
    result = numpy.full(num_array.shape, NODATA, dtype=float)
    valid_mask = (num_array > 0) & (den_array > 0)
    result[valid_mask] = num_array[valid_mask] / den_array[valid_mask]
    return result


def get_fid_list(downstream_value_sum_raster_path):
    vector = gdal.OpenEx(downstream_value_sum_raster_path, gdal.OF_VECTOR)
    layer = vector.GetLayer()
    fid_list = [feature.GetFID() for feature in layer]
    return fid_list


def sum_by_coverage(value_raster_path, mask_raster_path):
    running_sum = 0
    value_nodata = geoprocessing.get_raster_info(
        value_raster_path)['nodata'][0]
    for _, (value_array, mask_array) in geoprocessing.iterblocks(
            [(value_raster_path, 1), (mask_raster_path, 1)], skip_sparse=True):
        valid_mask = mask_array > 0
        if value_nodata is not None:
            valid_mask &= value_array != value_nodata
        running_sum += numpy.sum(value_array[valid_mask])
    return running_sum


def mask_raster(base_raster_path, mask_raster_path, target_raster_path):
    nodata = geoprocessing.get_raster_info(base_raster_path)['nodata'][0]

    def _mask_op(base_array, mask_array):
        result = base_array
        result[mask_array == 0] = nodata
        return result

    geoprocessing.raster_calculator(
        [(base_raster_path, 1), (mask_raster_path, 1)], _mask_op,
        target_raster_path, gdal.GDT_Float32, nodata)


def logical_and_masks(raster_path_list, target_raster_path):
    nodata_list = [
        geoprocessing.get_raster_info(path)['nodata'][0]
        for path in raster_path_list]
    nodata_target = -1

    LOGGER.debug(f'in (logical_and_masks): {raster_path_list}, {target_raster_path}')
    for path in raster_path_list:
        LOGGER.debug(f'{path} info: {geoprocessing.get_raster_info(path)}')

    def _logical_and(*array_list):
        n_arrays = len(array_list)
        overlap_count = numpy.zeros(array_list[0].shape, dtype=int)
        nodata_count = numpy.zeros(overlap_count.shape, dtype=int)
        for nodata, array in zip(nodata_list, array_list):
            if nodata is not None:
                valid_mask = (array != nodata)
                nodata_count += ~valid_mask
            else:
                valid_mask = numpy.ones(overlap_count.shape, dtype=bool)
                nodata_count += 1
            overlap_count += (valid_mask & (array > 0)).astype(int)
        # only nodata where they were all nodata
        result = (overlap_count == n_arrays).astype(int)
        result[nodata_count == n_arrays] = nodata_target
        return result

    geoprocessing.raster_calculator(
        [(path, 1) for path in raster_path_list], _logical_and,
        target_raster_path, gdal.GDT_Int32, nodata_target)


def rasterize_from_base_raster(
        task_graph, base_raster_path, base_vector_path, rasterize_kwargs,
        target_raster_path, dependent_task_list=[],
        additional_mask_raster_path=None):

    if additional_mask_raster_path is None:
        rasterized_raster_path = target_raster_path
    else:
        rasterized_raster_path = os.path.join(
            os.path.dirname(target_raster_path),
            f'pre_masked_{os.path.basename(target_raster_path)}')

    last_task = task_graph.add_task(
        func=geoprocessing.new_raster_from_base,
        args=(
            base_raster_path,
            rasterized_raster_path,
            gdal.GDT_Byte, [0]),
        target_path_list=[rasterized_raster_path],
        dependent_task_list=dependent_task_list,
        task_name=(
            f'create a new raster rasterizing {rasterized_raster_path}'))

    last_task = task_graph.add_task(
        func=geoprocessing.rasterize,
        args=(base_vector_path, rasterized_raster_path),
        kwargs=rasterize_kwargs,
        dependent_task_list=[last_task]+dependent_task_list,
        target_path_list=[rasterized_raster_path],
        task_name=f'rasterize {base_vector_path} to {rasterized_raster_path}')

    if additional_mask_raster_path:
        LOGGER.debug(
            f'********* logical ANDing {rasterized_raster_path} and '
            f'{additional_mask_raster_path}')
        last_task = task_graph.add_task(
            func=logical_and_masks,
            args=(
                [rasterized_raster_path, additional_mask_raster_path],
                target_raster_path),
            target_path_list=[target_raster_path],
            dependent_task_list=[last_task]+dependent_task_list,
            task_name=f'logical and between {rasterized_raster_path}, {additional_mask_raster_path}'
            )

    return last_task


def warp_and_rescale(
        base_raster_path, target_pixel_size, target_bb, target_projection_wkt,
        target_raster_path):
    """Warp a raster so units are consistent with a different pixel size."""
    working_dir = tempfile.mkdtemp(dir=os.path.dirname(target_raster_path))
    warped_raster_path = os.path.join(working_dir, 'warped.tif')

    geoprocessing.warp_raster(
        base_raster_path,
        target_pixel_size,
        warped_raster_path,
        'bilinear',
        target_bb=target_bb,
        target_projection_wkt=target_projection_wkt)
    warped_raster_info = geoprocessing.get_raster_info(warped_raster_path)

    test_base_, base_pixel_area = \
        geoprocessing.get_pixel_area_in_target_projection(
            base_raster_path, warped_raster_info['projection_wkt'])

    test_val, target_pixel_area = \
        geoprocessing.get_pixel_area_in_target_projection(
            warped_raster_path, warped_raster_info['projection_wkt'])

    scale_factor = target_pixel_area / base_pixel_area
    target_nodata = warped_raster_info['nodata'][0]

    if scale_factor != 1:
        def _scale_by_factor(array):
            result = array.copy().astype(float)
            if target_nodata is not None:
                nodata_mask = array != target_nodata
                result[nodata_mask] = array[nodata_mask] * scale_factor
            else:
                result *= scale_factor
            return result

        geoprocessing.raster_calculator(
            [(warped_raster_path, 1)], _scale_by_factor, target_raster_path,
            gdal.GDT_Float32, target_nodata)
    else:
        shutil.copyfile(warped_raster_path, target_raster_path)

    shutil.rmtree(working_dir)


def _sum_all_op(raster_path_list, target_raster):

    nodata_list = [
        geoprocessing.get_raster_info(path)['nodata'][0]
        for path in raster_path_list]
    local_nodata = -1

    def _sum_op(*array_list):
        result = numpy.zeros(array_list[0].shape)
        total_valid_mask = numpy.zeros(result.shape, dtype=bool)
        for array, nodata in zip(array_list, nodata_list):
            if nodata is not None:
                valid_mask = array != nodata
            else:
                valid_mask = numpy.ones(array.shape, dtype=bool)
            result[valid_mask] += array[valid_mask]
            total_valid_mask |= valid_mask
        result[~total_valid_mask] = local_nodata
        return result

    geoprocessing.raster_calculator(
        [(path, 1) for path in raster_path_list], _sum_op,
        target_raster, gdal.GDT_Float32, local_nodata)


def process_dem(
        task_graph, base_dem_path, aoi_path, target_pixel_size,
        workspace_dir, flow_dir_mfd_raster_path, outlet_raster_path):
    """Clip, clean, and route the dem.

    Args:
        task_graph (taskgraph): taskgraph to schedule
        base_dem_path (str): path to DEM raster
        aoi_path (str): path to AOI vector
        target_pixel_size (float): size of target raster in projected units
             of the aoi_path
        workspace_dir (str): directory that is safe to create intermediate
            and final files.

    Returns:
        task that will .get() the flow_direction_raster and outlet raster path
    """
    # clip and align the dem to the aoi_path file
    # pitfill the DEM
    clipped_dem_raster_path = os.path.join(
        workspace_dir, 'clipped_dem.tif')
    if geoprocessing.get_gis_type(aoi_path) == geoprocessing.RASTER_TYPE:
        aoi_info = geoprocessing.get_raster_info(aoi_path)
    else:
        aoi_info = geoprocessing.get_vector_info(aoi_path)

    clip_raster_task = task_graph.add_task(
        func=geoprocessing.warp_raster,
        args=(
            base_dem_path, (target_pixel_size, -target_pixel_size),
            clipped_dem_raster_path, 'bilinear'),
        kwargs={
            'target_bb': aoi_info['bounding_box'],
            'target_projection_wkt': aoi_info['projection_wkt'],
            },
        target_path_list=[clipped_dem_raster_path],
        task_name=f'clip base_dem_path {clipped_dem_raster_path}')

    filled_dem_raster_path = os.path.join(
        workspace_dir, 'filled_dem.tif')
    fill_pits_task = task_graph.add_task(
        func=routing.fill_pits,
        args=(
            (clipped_dem_raster_path, 1), filled_dem_raster_path),
        kwargs={
            'working_dir': workspace_dir,
            'max_pixel_fill_count': -1},
        dependent_task_list=[clip_raster_task],
        target_path_list=[filled_dem_raster_path],
        task_name=f'fill dem pits to {filled_dem_raster_path}')
    # route the DEM
    flow_dir_mfd_task = task_graph.add_task(
        func=routing.flow_dir_mfd,
        args=(
            (filled_dem_raster_path, 1), flow_dir_mfd_raster_path),
        kwargs={'working_dir': workspace_dir},
        dependent_task_list=[fill_pits_task],
        target_path_list=[flow_dir_mfd_raster_path],
        task_name=f'calc flow dir for {flow_dir_mfd_raster_path}')

    # calculate the number of downstream value pixels for any pixel on
    # the raster
    outlet_vector_path = os.path.join(workspace_dir, 'outlet_points.gpkg')
    outlet_detection_task = task_graph.add_task(
        func=routing.detect_outlets,
        args=(
            (flow_dir_mfd_raster_path, 1), 'mfd', outlet_vector_path),
        dependent_task_list=[flow_dir_mfd_task],
        target_path_list=[outlet_vector_path],
        ignore_path_list=[outlet_vector_path],
        task_name=f'detect outlets {outlet_vector_path}')

    rasterize_kwargs = {
        'burn_values': [1], 'option_list': ['ALL_TOUCHED=TRUE']}
    rasterized_outlet_task = rasterize_from_base_raster(
        task_graph, flow_dir_mfd_raster_path, outlet_vector_path,
        rasterize_kwargs, outlet_raster_path, dependent_task_list=[
            outlet_detection_task])

    return rasterized_outlet_task


def get_tuple_hash(t):
    # Convert the tuple to a string representation
    tuple_str = str(t)

    # Create a hash object
    hash_obj = hashlib.md5()

    # Calculate the hash of the tuple string
    hash_obj.update(tuple_str.encode('utf-8'))

    # Get the hexadecimal representation of the hash
    hash_str = hash_obj.hexdigest()

    return hash_str


def main():
    """Entrypoint."""
    parser = argparse.ArgumentParser(
        description='Distributed flood risk analysis.')
    parser.add_argument('dem_path', help='Path to DEM')
    parser.add_argument(
        'flood_risk_path', help=(
            'Path to flood risk measured in 0-1 risk / yr of '
            'flooding on a pixel.'))
    parser.add_argument('aoi_path', help='Path to AOI vector.')
    parser.add_argument(
        '--pixel_size', type=float, help='Target pixel size')
    parser.add_argument(
        '--target_raster_path', help='Path to desired output path.')
    parser.add_argument(
        '--file_prefix', default='',
        help='Added to intermediate files to avoid collision.')
    args = parser.parse_args()

    os.makedirs(GLOBAL_WORKSPACE_DIR, exist_ok=True)
    task_graph = taskgraph.TaskGraph(GLOBAL_WORKSPACE_DIR, -1)

    flow_dir_hash = 'dem_workspace_'+get_tuple_hash((
        args.dem_path, args.aoi_path, args.pixel_size))

    dem_workspace_dir = os.path.join(GLOBAL_WORKSPACE_DIR, flow_dir_hash)
    os.makedirs(dem_workspace_dir, exist_ok=True)

    # calculate flow direction and warp/align to AOI
    flow_dir_mfd_raster_path = os.path.join(
        GLOBAL_WORKSPACE_DIR, f'{args.file_prefix}flow_dir_mfd.tif')
    outlet_raster_path = os.path.join(
        GLOBAL_WORKSPACE_DIR, f'{args.file_prefix}outlet.tif')
    flow_dir_task = process_dem(
        task_graph, args.dem_path,
        args.aoi_path,
        args.pixel_size,
        dem_workspace_dir,
        flow_dir_mfd_raster_path, outlet_raster_path)

    # warp and clip flood risk path to AOI
    aoi_info = geoprocessing.get_vector_info(args.aoi_path)
    local_flood_risk_raster_path = os.path.join(
        GLOBAL_WORKSPACE_DIR, os.path.basename(args.flood_risk_path))
    warp_and_rescale_flood_risk_task = task_graph.add_task(
        func=warp_and_rescale,
        args=(
            args.flood_risk_path,
            (args.pixel_size, -args.pixel_size),
            aoi_info['bounding_box'],
            aoi_info['projection_wkt'],
            local_flood_risk_raster_path),
        target_path_list=[local_flood_risk_raster_path],
        task_name=f'clip local beneficiary {local_flood_risk_raster_path}')

    # calculate flow accumulation from flood direction
    flow_accumulation_raster_path = os.path.join(
        GLOBAL_WORKSPACE_DIR,
        f'{args.file_prefix}flow_accumulation_mfd.tif')
    flow_accum_task = task_graph.add_task(
        func=routing.flow_accumulation_mfd,
        args=(
            (flow_dir_mfd_raster_path, 1), flow_accumulation_raster_path),
        dependent_task_list=[flow_dir_task],
        target_path_list=[flow_accumulation_raster_path],
        task_name=f'flow accumulation for {flow_accumulation_raster_path}')
    # calculate weighted flood risk: flood risk / flow accumulation
    weighted_flood_risk_raster_path = os.path.join(
        GLOBAL_WORKSPACE_DIR, f'{args.file_prefix}weighted_flood_risk.tif')
    weighted_flood_risk_task = task_graph.add_task(
        func=geoprocessing.raster_calculator,
        args=(
            [(local_flood_risk_raster_path, 1),
             (flow_accumulation_raster_path, 1)], div_op,
            weighted_flood_risk_raster_path,
            gdal.GDT_Float32, NODATA),
        dependent_task_list=[
            flow_accum_task, warp_and_rescale_flood_risk_task],
        target_path_list=[weighted_flood_risk_raster_path],
        task_name=(
            f'calc weighted flood risk {weighted_flood_risk_raster_path}'))

    _ = task_graph.add_task(
        func=routing.distance_to_channel_mfd,
        args=(
            (flow_dir_mfd_raster_path, 1), (outlet_raster_path, 1),
            args.target_raster_path),
        kwargs={
            'weight_raster_path_band': (weighted_flood_risk_raster_path, 1)
            },
        dependent_task_list=[weighted_flood_risk_task, flow_dir_task],
        task_name=f'create downstream flood risk at {args.target_raster_path}')

    task_graph.join()
    task_graph.close()


if __name__ == '__main__':
    main()

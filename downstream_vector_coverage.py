"""Calculate downstream vector coverage.


How many road pixels are downstream of a mask of an area... Input is maskzl, roads, dem, output is road pixel count n

There's another technique where you pass a mask and it smears downstream and then masks out another layer then adds that together to get a single number

Inputs
Dem
Road vector current broken up by primary, secondary, tertiary ph says primary etc ind calls it collector artery
Population map landscan
Masks ar no



"""
import argparse
import os
import logging
import shutil
import sys
import tempfile

from osgeo import gdal
from osgeo import ogr
from ecoshard.geoprocessing import routing
from ecoshard import geoprocessing
from ecoshard import taskgraph
import numpy

logging.basicConfig(
    level=logging.DEBUG,
    stream=sys.stdout,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(levelname)s %(name)s'
        ' [%(funcName)s:%(lineno)d] %(message)s'))
LOGGER = logging.getLogger(os.path.splitext(os.path.basename(__file__))[0])
LOGGER.setLevel(logging.DEBUG)
logging.getLogger('ecoshard.fetch_data').setLevel(logging.INFO)


def copy_vector_to_downstream_value(
        input_vector_path, fid_to_value_map, output_vector_path):
    input_vector = ogr.Open(input_vector_path)
    input_layer = input_vector.GetLayer()

    # Create a new output vector
    output_driver = ogr.GetDriverByName('GPKG')
    output_vector = output_driver.CreateDataSource(output_vector_path)
    output_layer = output_vector.CreateLayer(
        os.path.splitext(os.path.basename(output_vector_path))[0],
        geom_type=input_layer.GetGeomType())

    # Create a new field called "downstream_value"
    downstream_value_field = ogr.FieldDefn("downstream_value", ogr.OFTInteger)
    output_layer.CreateField(downstream_value_field)

    # Copy feature geometry and set "downstream_value" field
    output_layer.StartTransaction()
    for feature in input_layer:
        output_feature = ogr.Feature(output_layer.GetLayerDefn())
        output_feature.SetGeometry(feature.GetGeometryRef())
        output_feature.SetField(
            "downstream_value", fid_to_value_map[feature.GetFID()].get())
        output_layer.CreateFeature(output_feature)
    output_layer.CommitTransaction()

    input_layer = None
    output_layer = None
    input_vector = None
    output_vector = None


def return_a_string(string_val):
    """Returns `string_val`."""
    return string_val


def get_fid_list(downstream_value_sum_raster_path):
    vector = gdal.OpenEx(downstream_value_sum_raster_path, gdal.OF_VECTOR)
    layer = vector.GetLayer()
    fid_list = [feature.GetFID() for feature in layer]
    return fid_list


def sum_by_coverage(value_raster_path, mask_raster_path):
    running_sum = 0
    for value_array, mask_array in geoprocessing.iterblocks(
            [(value_raster_path, 1), (mask_raster_path, 1)], skip_sparse=True):
        running_sum += numpy.sum(value_array[mask_array > 0])
    return running_sum


def rasterize_from_base_raster(
        task_graph, base_raster_path, base_vector_path, rasterize_kwargs,
        target_raster_path, dependent_task_list=[]):
    new_raster_task = task_graph.add_task(
        func=geoprocessing.new_raster_from_base,
        args=(
            base_raster_path,
            target_raster_path,
            gdal.GDT_Byte, [0]),
        target_path_list=[target_raster_path],
        task_name=(
            f'create a new raster for outlets {target_raster_path}'))

    value_raster_task = task_graph.add_task(
        func=geoprocessing.rasterize,
        args=(base_vector_path, target_raster_path),
        kwargs=rasterize_kwargs,
        dependent_task_list=dependent_task_list+[new_raster_task],
        target_path_list=[target_raster_path],
        task_name=f'rasterize {base_vector_path} to {target_raster_path}')

    return value_raster_task


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
            target_raster_path, warped_raster_info['projection_wkt'])

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
        workspace_dir):
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
        task that will .get() the flow_direction_raster path
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
        task_name=f'clip {clipped_dem_raster_path}')

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
    flow_dir_mfd_raster_path = os.path.join(
        workspace_dir, 'mfd_flow_dir.tif')
    flow_dir_mfd_task = task_graph.add_task(
        func=routing.flow_dir_mfd,
        args=(
            (filled_dem_raster_path, 1), flow_dir_mfd_raster_path),
        kwargs={'working_dir': workspace_dir},
        dependent_task_list=[fill_pits_task],
        target_path_list=[flow_dir_mfd_raster_path],
        task_name=f'calc flow dir for {flow_dir_mfd_raster_path}')

    flow_dir_path_task = task_graph.add_task(
        func=return_a_string,
        args=(flow_dir_mfd_raster_path,),
        dependent_task_list=[flow_dir_mfd_task],
        store_result=True,
        task_name=f'return {flow_dir_mfd_raster_path}')

    return flow_dir_path_task


def main():
    """Entrypoint."""
    parser = argparse.ArgumentParser(
        description='Downstream beneficiary analysis.')
    parser.add_argument('dem_path', help='Path to DEM file')
    parser.add_argument('aoi_path', help='Path to AOI file')
    parser.add_argument(
        'target_pixel_size', type=float,
        help='Target pixel size in AOI projected units')
    parser.add_argument(
        '--vector_or_raster_value_paths', nargs='+', required=True,
        help='Path to vector or raster value file')
    parser.add_argument(
        '--sum_of_downstream_aoi_path',
        help='Path to a vector to aggregate results by for downstream sums')
    parser.add_argument(
        '--output_suffix', default='',
        help='Additional string to append to output filenames.')
    parser.add_argument('--debug', action='store_true', help='debug mode')
    args = parser.parse_args()

    non_existent_files = [
        file for file in args.vector_or_raster_value_paths
        if not os.path.isfile(file)]

    if non_existent_files:
        raise FileNotFoundError(
            "The following files do not exist: "
            f"{', '.join(non_existent_files)}")

    if args.output_suffix and not args.output_suffix.startswith('_'):
        args.output_suffix = '_' + args.output_suffix

    target_dir = os.path.splitext(
        os.path.basename(args.dem_path))[0]

    workspace_dir = os.path.join(
        f'downstream_beneficiary_workspace{args.output_suffix}',
        target_dir)
    os.makedirs(workspace_dir, exist_ok=True)

    task_graph = taskgraph.TaskGraph(workspace_dir, min(
        os.cpu_count(), len(args.vector_or_raster_value_paths))
        if not args.debug else -1, 5)

    flow_dir_task = process_dem(
        task_graph, args.dem_path, args.aoi_path, args.target_pixel_size,
        workspace_dir)

    aoi_info = geoprocessing.get_vector_info(args.aoi_path)
    value_raster_list = []
    for index, vector_or_raster_value_path in enumerate(
            args.vector_or_raster_value_paths):
        local_value_raster_path = os.path.join(
            workspace_dir,
            f'local_value_raster_{index}{args.output_suffix}.tif')
        value_raster_list.append(local_value_raster_path)
        if geoprocessing.get_gis_type(vector_or_raster_value_path) == \
                geoprocessing.VECTOR_TYPE:
            LOGGER.debug(
                f'**** processing vector {vector_or_raster_value_path}')

            vector_path = vector_or_raster_value_path
            reprojected_vector_path = os.path.join(
                workspace_dir,
                f'reprojected_{os.path.basename(vector_path)}')
            reproject_task = task_graph.add_task(
                func=geoprocessing.reproject_vector,
                args=(
                    vector_path, aoi_info['projection_wkt'],
                    reprojected_vector_path),
                target_path_list=[reprojected_vector_path],
                ignore_path_list=[reprojected_vector_path],
                task_name=f'reproject {reprojected_vector_path}')

            new_raster_task = task_graph.add_task(
                func=geoprocessing.new_raster_from_base,
                args=(
                    flow_dir_task.get(),
                    local_value_raster_path,
                    gdal.GDT_Byte, [0]),
                target_path_list=[local_value_raster_path],
                task_name=(
                    f'create a new raster for rasterization '
                    f'{local_value_raster_path}'))
            value_raster_task = task_graph.add_task(
                func=geoprocessing.rasterize,
                args=(reprojected_vector_path, local_value_raster_path),
                kwargs={'burn_values': [1]},
                dependent_task_list=[new_raster_task, reproject_task],
                target_path_list=[local_value_raster_path],
                task_name=(
                    f'rasterize {reprojected_vector_path} to {local_value_raster_path}'))
        else:
            # clip and reproject value raster to aoi's projection
            aoi_info = geoprocessing.get_vector_info(args.aoi_path)
            raster_path = vector_or_raster_value_path
            LOGGER.debug(
                f'**** processing raster {raster_path}')
            warp_and_rescale_raster_task = task_graph.add_task(
                func=warp_and_rescale,
                args=(
                    raster_path,
                    (args.target_pixel_size, -args.target_pixel_size),
                    aoi_info['bounding_box'],
                    aoi_info['projection_wkt'],
                    local_value_raster_path),
                target_path_list=[local_value_raster_path],
                task_name=f'clip {local_value_raster_path}')

    value_raster_path = os.path.join(
        workspace_dir, f'value_raster{args.output_suffix}.tif')

    task_graph.join()
    sum_value_list_task = task_graph.add_task(
        func=_sum_all_op,
        args=(value_raster_list, value_raster_path),
        target_path_list=[value_raster_path],
        task_name=f'sum all to {value_raster_path}')

    # calculate the number of downstream value pixels for any pixel on
    # the raster
    outlet_vector_path = os.path.join(
        workspace_dir, f'outlet_points{args.output_suffix}.gpkg')
    outlet_detection_task = task_graph.add_task(
        func=routing.detect_outlets,
        args=(
            (flow_dir_task.get(), 1), 'mfd', outlet_vector_path),
        target_path_list=[outlet_vector_path],
        ignore_path_list=[outlet_vector_path],
        task_name=f'detect outlets {outlet_vector_path}')

    outlet_raster_path = os.path.join(
        workspace_dir, f'outlet_raster{args.output_suffix}.tif')
    rasterize_kwargs = {'burn_values': [1], 'option_list': ['ALL_TOUCHED=TRUE']}
    rasterized_outlet_task = rasterize_from_base_raster(
        task_graph, flow_dir_task.get(), outlet_vector_path,
        rasterize_kwargs, outlet_raster_path, dependent_task_list=[
            outlet_detection_task])

    downstream_value_sum_raster_path = os.path.join(
        workspace_dir, f'downstream_value_sum{args.output_suffix}.tif')
    task_graph.add_task(
        func=routing.distance_to_channel_mfd,
        args=(
            (flow_dir_task.get(), 1), (outlet_raster_path, 1),
            downstream_value_sum_raster_path),
        kwargs={
            'weight_raster_path_band': (value_raster_path, 1)},
        target_path_list=[downstream_value_sum_raster_path],
        dependent_task_list=[rasterized_outlet_task, sum_value_list_task],
        task_name='value accumulation')

    # TODO: check if there's a flag to do sum of downstream aoi
    if args.sum_of_downstream_aoi_path is not None:
        reprojected_downstream_aoi_path = os.path.join(
            workspace_dir,
            f'reprojected_{os.path.basename(args.sum_of_downstream_aoi_path)}')
        reproject_task = task_graph.add_task(
            func=geoprocessing.reproject_vector,
            args=(
                args.sum_of_downstream_aoi_path, aoi_info['projection_wkt'],
                reprojected_downstream_aoi_path),
            target_path_list=[reprojected_downstream_aoi_path],
            ignore_path_list=[reprojected_downstream_aoi_path],
            task_name=f'reproject {reprojected_downstream_aoi_path}')
        reproject_task.join()
        fid_list = get_fid_list(reprojected_downstream_aoi_path)
        result_by_fid = {}
        for fid in fid_list:
            fid_mask_path = os.path.join(workspace_dir, f'{fid}_mask.tif')
            fid_rasterize_kwargs = {
                'burn_values': [1],
                'option_list': ['ALL_TOUCHED=TRUE', f'WHERE="FID={fid}"']}
            fid_rasterize_task = rasterize_from_base_raster(
                task_graph, flow_dir_task.get(),
                reprojected_downstream_aoi_path,
                fid_rasterize_kwargs, fid_mask_path)

            downstream_coverage_raster_path = os.path.join(
                workspace_dir, f'downstream_coverage_{fid}.tif')
            downstream_coverage_task = task_graph.add_task(
                func=routing.distance_to_channel_mfd,
                args=(
                    (flow_dir_task.get(), 1), (outlet_raster_path, 1),
                    downstream_coverage_raster_path),
                kwargs={
                    'weight_raster_path_band': (fid_mask_path, 1)},
                target_path_list=[
                    downstream_coverage_raster_path, rasterized_outlet_task],
                dependent_task_list=[fid_rasterize_task],
                task_name='value accumulation')

            sum_by_coverage_task = task_graph.add_task(
                func=sum_by_coverage,
                args=(value_raster_path, downstream_coverage_raster_path),
                store_result=True,
                dependent_task_list=[
                    sum_value_list_task, downstream_coverage_task],
                task_name=f'sum coverage for fid {fid}')

            result_by_fid[fid] = sum_by_coverage_task

        downstream_sum_vector_path = os.path.join(
            workspace_dir,
            f'downstream_value_sum_by_aoi{args.output_suffix}.gpkg')
        task_graph.join()
        copy_vector_to_downstream_value(
            reprojected_downstream_aoi_path, result_by_fid,
            downstream_sum_vector_path)

    task_graph.join()
    task_graph.close()


if __name__ == '__main__':
    main()

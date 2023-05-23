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

from ecoshard.geoprocessing import routing
from ecoshard import geoprocessing
from ecoshard import taskgraph
from osgeo import gdal


def return_a_string(string_val):
    """Returns `string_val`."""
    return string_val


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
    if geoprocessing.get_gis_type(aoi_path) == geoprocessing.geoprocessing.RASTER_TYPE:
        aoi_info = geoprocessing.get_raster_info(aoi_path)
    else:
        aoi_info = geoprocessing.get_vector_info(aoi_path)

    clip_raster_task = task_graph.add_task(
        func=geoprocessing.warp_raster,
        args=(
            base_dem_path, target_pixel_size, clipped_dem_raster_path,
            'bilinear'),
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
            (base_dem_path, 1), filled_dem_raster_path),
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
    parser.add_argument(
        'vector_or_raster_value_path',
        help='Path to vector or raster value file')
    parser.add_argument('aoi_path', help='Path to AOI file')
    parser.add_argument(
        'target_pixel_size', type=float,
        help='Target pixel size in AOI projected units')
    parser.add_argument(
        '--sum_of_downstream_aoi', action='store_true',
        help='Optional argument')
    parser.add_argument(
        '--output_suffix', default='',
        help='Additional string to append to output filenames.')
    args = parser.parse_args()

    if args.output_suffix and not args.output_suffix.startswith('_'):
        args.output_suffix = '_' + args.output_suffix

    target_dir = (
        f'{os.path.splitext(os.path.basename(args.dem_path))[0]}_'
        f'''{os.path.splitext(os.path.basename(
            args.vector_or_raster_value_path))[0]}''')

    workspace_dir = os.path.join(
        f'downstream_beneficiary_workspace{args.output_suffix}',
        target_dir)
    os.makedirs(workspace_dir, exist_ok=True)

    task_graph = taskgraph.TaskGraph(workspace_dir, 2)

    flow_dir_task = process_dem(
        task_graph, args.dem_path, args.aoi_path, args.target_pixel_size,
        workspace_dir)

    value_raster_path = os.path.join(
        workspace_dir, f'value_raster{args.output_suffix}.tif')
    if geoprocessing.get_gis_type(args.vector_or_raster_value_path) == \
            geoprocessing.geoprocessing.VECTOR_TYPE:
        # rasterize the vector onto a raster of the correct shape/resolution
        new_raster_task = task_graph.add_task(
            func=geoprocessing.create_raster_from_vector_extents,
            args=(
                args.aoi_path, value_raster_path, args.target_pixel_size,
                gdal.GDT_Byte, 0),
            target_path_list=[value_raster_path],
            task_name=(
                f'create a new raster for rasterization {value_raster_path}'))

        # rasterize the vector
        # TODO: Do I need to project vector path to aoi's projection?
        vector_path = args.vector_or_raster_value_path
        value_raster_task = task_graph.add_task(
            func=geoprocessing.rasterize,
            args=(vector_path, value_raster_path),
            kwargs={'burn_values': [1]},
            dependent_task_list=[new_raster_task],
            target_path_list=[value_raster_path],
            task_name=f'rasterize {vector_path} to {value_raster_path}')
    else:
        # clip and reproject value raster to aoi's projection
        aoi_info = geoprocessing.get_vector_info(args.aoi_path)
        base_value_raster_path = args.vector_or_raster_value_path
        value_raster_task = task_graph.add_task(
            func=geoprocessing.warp_raster,
            args=(
                base_value_raster_path, args.target_pixel_size,
                value_raster_path, 'bilinear'),
            kwargs={
                'target_bb': aoi_info['bounding_box'],
                'target_projection_wkt': aoi_info['projection_wkt'],
                },
            target_path_list=[value_raster_path],
            task_name=f'clip {value_raster_path}')

    # TODO: check if there's a flag to do sum of downstream aoi

    # TODO: calclate number of downstream value pixels for any pixel on
    #   the raster

    # flow accum the vector on the routed DEM
    flow_dir_mfd_raster_path = flow_dir_task.get()
    downstream_value_sum_raster_path = os.path.join(
        workspace_dir, f'downstream_value_sum{args.output_suffix}.tif')
    task_graph.add_task(
        func=routing.flow_accumulation_mfd,
        args=(
            (flow_dir_mfd_raster_path, 1), downstream_value_sum_raster_path),
        kwargs={
            'weight_raster_path_band': (value_raster_path, 1)},
        target_path_list=[downstream_value_sum_raster_path],
        dependent_task_list=[value_raster_task],
        task_name='flow accumulation')

    task_graph.close()
    task_graph.join()


if __name__ == '__main__':
    main()

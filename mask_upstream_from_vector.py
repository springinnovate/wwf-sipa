import argparse
import os
import logging
import sys
from ecoshard import taskgraph
from ecoshard import geoprocessing
from ecoshard.geoprocessing import routing
from osgeo import gdal

logging.basicConfig(
    level=logging.DEBUG,
    stream=sys.stdout,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(levelname)s %(name)s'
        ' [%(funcName)s:%(lineno)d] %(message)s'))
LOGGER = logging.getLogger(os.path.splitext(os.path.basename(__file__))[0])
LOGGER.setLevel(logging.DEBUG)
logging.getLogger('ecoshard.fetch_data').setLevel(logging.INFO)


def main():
    parser = argparse.ArgumentParser(description=(
        "Mask upstream components of a raster given a downstream vector."))
    parser.add_argument(
        "--dem_path", type=str, required=True,
        help="Path to the DEM file")
    parser.add_argument(
        "--vector_path", type=str, required=True,
        help="Path to the vector file")
    parser.add_argument(
        "--raster_path", type=str, required=True,
        help="Path to the arbitrary raster file")
    parser.add_argument(
        '--working_dir', type=str, required=True,
        help="Path to working directory.")
    parser.add_argument(
        '--buffer_window', type=float, required=True,
        help="Search an area around this many units of the raster path")
    args = parser.parse_args()
    intermediate_dir = os.path.join(args.working_dir, 'intermediate_files')
    os.makedirs(intermediate_dir, exist_ok=True)
    task_graph = taskgraph.TaskGraph(
        args.working_dir, 4,
        parallel_mode='process',
        reporting_interval=10.0)

    # temp workspace
    # clip the DEM to an intersection of the raster path and the buffer window
    # which is centered on the centroid of the vector_path
    # var bounding_box

    vector_info = geoprocessing.get_vector_info(args.vector_path)
    dem_info = geoprocessing.get_raster_info(args.dem_path)
    raster_info = geoprocessing.get_raster_info(args.raster_path)

    if not (vector_info['projection_wkt'] ==
            dem_info['projection_wkt'] ==
            raster_info['projection_wkt']):
        raise RuntimeError(
            'vector dem and raster are not same projections: ' +
            vector_info['projection_wkt'] + '\n' +
            dem_info['projection_wkt'] + '\n' +
            raster_info['projection_wkt'])

    # warp the dem and the raster path to this bounding box

    # bounding box order is [minx, miny, maxx, maxy]
    buffered_vector_bounding_box = [
        vector_info['bounding_box'][i] +
        args.buffer_window*s for i, s in enumerate([-1, -1, 1, 1])]
    bounding_box_list = [
        raster_info['bounding_box'], buffered_vector_bounding_box]
    bounding_box = geoprocessing.merge_bounding_box_list(
        bounding_box_list, 'intersection')
    LOGGER.debug(bounding_box)

    clipped_dem_path = os.path.join(
        intermediate_dir, 'clipped_dem.tif')
    clip_dem_task = task_graph.add_task(
        func=geoprocessing.warp_raster,
        args=(
            args.dem_path, dem_info['pixel_size'], clipped_dem_path,
            'near'),
        kwargs={'target_bb': bounding_box},
        target_path_list=[clipped_dem_path],
        task_name=f'clip {clipped_dem_path}')

    # fill pits in the dem
    filled_dem_path = os.path.join(intermediate_dir, 'filled_dem.tif')
    fill_dem_task = task_graph.add_task(
        func=routing.fill_pits,
        args=((clipped_dem_path, 1), filled_dem_path),
        kwargs={'working_dir': intermediate_dir},
        dependent_task_list=[clip_dem_task],
        target_path_list=[filled_dem_path],
        task_name='fill dem')

    # mfd route the dem
    mfd_flow_dir_path = os.path.join(intermediate_dir, 'mfd_flow_dir.tif')
    flow_dir_task = task_graph.add_task(
        func=routing.flow_dir_mfd,
        args=((filled_dem_path, 1), mfd_flow_dir_path),
        kwargs={'working_dir': intermediate_dir},
        dependent_task_list=[fill_dem_task],
        target_path_list=[mfd_flow_dir_path],
        task_name='flow dir mfd')

    # raster to mark the distance to the "channel"
    rasterized_vector_path = os.path.join(
        intermediate_dir, 'rasterized_vector.tif')
    new_raster_task = task_graph.add_task(
        func=geoprocessing.new_raster_from_base,
        args=(
            clipped_dem_path, rasterized_vector_path,
            gdal.GDT_Int32, [-1]),
        dependent_task_list=[clip_dem_task],
        ignore_path_list=[rasterized_vector_path],
        task_name='downstream intersection')
    rasterized_vector_task = task_graph.add_task(
        func=geoprocessing.rasterize,
        args=(args.vector_path, rasterized_vector_path),
        kwargs={'burn_values': [1]},
        dependent_task_list=[new_raster_task],
        target_path_list=[rasterized_vector_path],
        task_name=f'rasterize {rasterized_vector_path}')

    downstream_intersection_mask_path = os.path.join(
        intermediate_dir, 'downstream_intersection.tif')
    distance_to_channel_task = task_graph.add_task(
        func=routing.distance_to_channel_mfd,
        args=((mfd_flow_dir_path, 1), (rasterized_vector_path, 1),
              downstream_intersection_mask_path),
        dependent_task_list=[flow_dir_task, rasterized_vector_task],
        target_path_list=[downstream_intersection_mask_path],
        task_name='distance to channel')

    clipped_raster_path = os.path.join(
        intermediate_dir, 'clipped_'+os.path.basename(args.raster_path))
    clip_raster_task = task_graph.add_task(
        func=geoprocessing.warp_raster,
        args=(
            args.raster_path, dem_info['pixel_size'], clipped_raster_path,
            'near'),
        kwargs={'target_bb': bounding_box},
        target_path_list=[clipped_raster_path],
        task_name=f'clip {clipped_raster_path}')

    upstream_masked_raster_path = os.path.join(
        args.working_dir,
        os.path.basename(os.path.normpath(args.working_dir))+'.tif')
    task_graph.add_task(
        func=geoprocessing.raster_calculator,
        args=(
            [(clipped_raster_path, 1), (downstream_intersection_mask_path, 1),
             (raster_info['nodata'][0], 'raw')],
            _mask_op, upstream_masked_raster_path, raster_info['datatype'],
            raster_info['nodata'][0]),
        dependent_task_list=[clip_raster_task, distance_to_channel_task],
        target_path_list=[upstream_masked_raster_path],
        task_name='mask upstraem raster')
    # mask out the clipped raster_path as the result

    task_graph.join()
    task_graph.close()


def _mask_op(base_array, mask_array, nodata):
    result = base_array.copy()
    result[mask_array <= 0] = nodata
    return result


if __name__ == '__main__':
    main()

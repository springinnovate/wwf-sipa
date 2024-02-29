import argparse
import os
import logging
import sys
from ecoshard import taskgraph
from ecoshard import geoprocessing

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

    os.makedirs(args.working_dir, exist_ok=True)
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
    # def warp_raster(
    #     base_raster_path, target_pixel_size, target_raster_path,
    #     resample_method, target_bb=None, base_projection_wkt=None,
    #     target_projection_wkt=None, n_threads=None, vector_mask_options=None,
    #     gdal_warp_options=None, working_dir=None,
    #     output_type=gdal.GDT_Unknown,
    #     raster_driver_creation_tuple=DEFAULT_GTIFF_CREATION_TUPLE_OPTIONS,
    #     osr_axis_mapping_strategy=DEFAULT_OSR_AXIS_MAPPING_STRATEGY):

    # fill pits in the dem
    # def fill_pits(
    #     dem_raster_path_band, target_filled_dem_raster_path,
    #     working_dir=None,
    #     long long max_pixel_fill_count=-1,
    #     single_outlet_tuple=None,
    #     raster_driver_creation_tuple=DEFAULT_GTIFF_CREATION_TUPLE_OPTIONS):

    # D8 route the dem
    # def flow_dir_d8(
    #     dem_raster_path_band, target_flow_dir_path,
    #     working_dir=None,
    #     raster_driver_creation_tuple=DEFAULT_GTIFF_CREATION_TUPLE_OPTIONS):

    # raster to mark the distance to the "channel"
    # def distance_to_channel_d8(
    #     flow_dir_d8_raster_path_band, channel_raster_path_band,
    #     target_distance_to_channel_raster_path,
    #     weight_raster_path_band=None,
    #     raster_driver_creation_tuple=DEFAULT_GTIFF_CREATION_TUPLE_OPTIONS):

    # mask out the clipped raster_path as the result


if __name__ == '__main__':
    main()

import glob
import logging
LOG_LEVEL = logging.DEBUG
LOG_FORMAT = (
    '%(asctime)s (%(relativeCreated)d) %(levelname)s %(name)s'
    ' [%(funcName)s:%(lineno)d] %(message)s')
logging.basicConfig(
    level=LOG_LEVEL,
    format=LOG_FORMAT)

logging.getLogger('taskgraph').setLevel(logging.DEBUG)
LOGGER = logging.getLogger(__name__)

import argparse
import os
import numpy
from ecoshard import geoprocessing
from ecoshard import taskgraph
from osgeo import gdal


OUTPUT_DIR = 'area_of_polygon_working_dir'
PIXEL_SIZE = (0.0002777777777777777778, 0.0002777777777777777778)


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


def main():
    """Entry point."""
    parser = argparse.ArgumentParser(description='projected area of polygon')
    parser.add_argument('vector_path_pattern', help='path/pattern to vectors')
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    task_graph = taskgraph.TaskGraph(OUTPUT_DIR, os.cpu_count(), 15.0)

    count_task_list = []
    for vector_path in glob.glob(args.vector_path_pattern):
        raster_path = os.path.join(
            OUTPUT_DIR, os.path.splitext(os.path.basename(vector_path))[0])
        rasterize_task = task_graph.add_task(
            func=geoprocessing.create_raster_from_vector_extents,
            args=(vector_path, raster_path, PIXEL_SIZE, gdal.GDT_Byte, 0),
            target_path_list=[raster_path],
            task_name=f'rasterize {raster_path}')
        count_task = task_graph.add_task(
            func=count_valid_pixels,
            args=(raster_path,),
            dependent_task_list=[rasterize_task],
            store_result=True,
            task_name=f'count {raster_path}')
        count_task_list.append((
            os.path.splitext(os.path.basename(raster_path))[0], count_task))
    with open('area_table.csv', 'w') as table:
        table.write('file,pixel count,area ha\n')
        for filename, count_task in count_task_list:
            pixel_count, area_ha = count_task.get()
            table.write(f'{filename},{pixel_count},{count_task}\n')


if __name__ == '__main__':
    main()

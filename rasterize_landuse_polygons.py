"""Rasterize the landuse polygons onto a single raster."""
import glob
import logging
import os


from ecoshard import geoprocessing
from ecoshard import taskgraph
from osgeo import gdal
import numpy

logging.basicConfig(
    level=logging.DEBUG,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(levelname)s %(name)s'
        ' [%(funcName)s:%(lineno)d] %(message)s'))
logging.getLogger('taskgraph').setLevel(logging.INFO)
LOGGER = logging.getLogger(__name__)

TARGET_PIXEL_SIZE = 10.0


def get_all_field_values(shapefile_path, field_id):
    """Return all values in field_id for all features in ``shapefile_path``."""
    landcover_id_set = set()
    vector = gdal.OpenEx(shapefile_path, gdal.OF_VECTOR)
    layer = vector.GetLayer()
    for feature in layer:
        landcover_id_set.add(feature.GetField(landcover_field))
    return landcover_id_set


def main():
    path_to_shapefiles = './data/land_use_polygons/*'
    path_to_target_rasters = './data/landcover_rasters/'
    task_graph = taskgraph.TaskGraph('.', 4, 15.0)
    os.makedirs(path_to_target_rasters, exist_ok=True)
    landcover_field = 'AGG12'
    landcover_id_set = set()
    field_value_task_list = []
    for shapefile_path in glob.glob(path_to_shapefiles):
        basename = os.path.basename(os.path.splitext(shapefile_path)[0])
        task = task_graph.add_task(
            func=get_all_field_values,
            args=(shapefile_path, landcover_field),
            store_result=True,
            task_name=f'{landcover_field} values for {basename}')
        field_value_task_list.append(task)

        LOGGER.info(f'processing {shapefile_path}')
        vector_info = geoprocessing.get_vector_info(shapefile_path)
        xwidth = numpy.subtract(*[vector_info['bounding_box'][i] for i in (2, 0)])
        ywidth = numpy.subtract(*[vector_info['bounding_box'][i] for i in (3, 1)])
        n_cols = int(xwidth / TARGET_PIXEL_SIZE)
        n_rows = int(ywidth / TARGET_PIXEL_SIZE)
        LOGGER.info(f'expected raster size for {basename} is ({n_cols}x{n_rows})')

        target_raster_path = os.path.join(path_to_target_rasters, f'{basename}_lulc.tif')
        geoprocessing.create_raster_from_vector_extents(
            shapefile_path, target_raster_path, (TARGET_PIXEL_SIZE, -TARGET_PIXEL_SIZE),
            gdal.GDT_Byte, 128)

    for task in field_value_task_list:
        landcover_id_set |= task.get()
    LOGGER.info(f'landcover set: {landcover_id_set}')


if __name__ == '__main__':
    main()

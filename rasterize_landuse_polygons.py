"""Rasterize the landuse polygons onto a single raster."""
import glob
import logging
import os

from ecoshard import geoprocessing
from osgeo import gdal
import numpy

logging.basicConfig(
    level=logging.DEBUG,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(levelname)s %(name)s'
        ' [%(funcName)s:%(lineno)d] %(message)s'))
logging.getLogger('taskgraph').setLevel(logging.INFO)
LOGGER = logging.getLogger(__name__)

TARGET_PIXEL_SIZE = 3.0


def main():
    path_to_shapefiles = './data/land_use_polygons/*'
    landcover_field = 'AGG12'
    landcover_id_set = set()
    for shapefile_path in glob.glob(path_to_shapefiles):
        LOGGER.info(f'processing {shapefile_path}')
        basename = os.path.basename(os.path.splitext(shapefile_path)[0])
        vector_info = geoprocessing.get_vector_info(shapefile_path)
        xwidth = numpy.subtract(vector_info['bounding_box'][i] for i in (2, 0))
        ywidth = numpy.subtract(vector_info['bounding_box'][i] for i in (3, 1))
        n_cols = int(xwidth / TARGET_PIXEL_SIZE)
        n_rows = int(ywidth / TARGET_PIXEL_SIZE)
        LOGGER.info(f'expected raster size for {basename} is ({n_cols}x{n_rows})')
        # vector = gdal.OpenEx(shapefile_path, gdal.OF_VECTOR)
        # layer = vector.GetLayer()
        # for feature in layer:
        #     landcover_id_set.add(feature.GetField(landcover_field))
    # LOGGER.info(f'landcover set: {landcover_id_set}')


if __name__ == '__main__':
    main()

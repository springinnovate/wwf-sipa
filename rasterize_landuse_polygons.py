"""Rasterize the landuse polygons onto a single raster."""
import glob
import logging

from ecoshard import geoprocessing
from osgeo import gdal

logging.basicConfig(
    level=logging.DEBUG,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(levelname)s %(name)s'
        ' [%(funcName)s:%(lineno)d] %(message)s'))
logging.getLogger('taskgraph').setLevel(logging.INFO)
LOGGER = logging.getLogger(__name__)


def main():
    path_to_shapefiles = './data/land_use_polygons/*'
    landcover_field = 'AGG12'
    landcover_id_set = set()
    for shapefile_path in glob.glob(path_to_shapefiles):
        LOGGER.info(f'processing {shapefile_path}')
        vector = gdal.OpenEx(shapefile_path, gdal.OF_VECTOR)
        layer = vector.GetLayer()
        for feature in layer:
            landcover_id_set.add(feature.GetField(landcover_field))
    LOGGER.info(f'landcover set: {landcover_id_set}')


if __name__ == '__main__':
    main()

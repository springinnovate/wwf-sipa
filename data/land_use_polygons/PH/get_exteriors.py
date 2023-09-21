"""Get Exteriors of Polyons."""
import glob
import logging
import numpy
from ecoshard import geoprocessing

from osgeo import gdal
import geopandas

logging.basicConfig(
    level=logging.DEBUG,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(levelname)s %(name)s'
        ' [%(pathname)s.%(funcName)s:%(lineno)d] %(message)s'))
LOGGER = logging.getLogger(__name__)

if __name__ == '__main__':
    raster_path = 'D:/repositories/wwf-sipa/data/landcover_rasters/ph_baseline_lulc_md5_7f29da.tif'
    raster_info = geoprocessing.get_raster_info(raster_path)
    nodata = raster_info['nodata'][0]

    def _mask_op(array):
        result = numpy.ones(array.shape, dtype=bool)
        result[array == nodata] = 0
        return result

    local_mask_raster = 'ph_mask.tif'

    geoprocessing.single_thread_raster_calculator(
        [(raster_path, 1)], _mask_op, local_mask_raster, gdal.GDT_Byte, 0)
    LOGGER.info(f'dibe')

import sys
import os
from ecoshard import geoprocessing
import glob
import logging


logging.basicConfig(
    level=logging.DEBUG,
    stream=sys.stdout,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(levelname)s %(name)s'
        ' [%(funcName)s:%(lineno)d] %(message)s'))
LOGGER = logging.getLogger(os.path.splitext(os.path.basename(__file__))[0])
LOGGER.setLevel(logging.DEBUG)

for path_band in [(path, 1) for path in glob.glob('*.tif')]:
    clean_path = f"{path_band[0].split('_md5')[0]}_clean.tif"
    raster_info = geoprocessing.get_raster_info(path_band[0])

    geoprocessing.raster_calculator(
        [path_band], lambda x: x, clean_path, raster_info['datatype'],
        raster_info['nodata'][0])


gdal_translate diff_infra_baseline_usle_wwf_PH_clean.tif COG_diff_infra_baseline_usle_wwf_PH_clean.tif -of COG -co COMPRESS=LZW

gdal_translate diff_future_infra_baseline_sed_export_wwf_PH_clean.tif COG_diff_future_infra_baseline_sed_export_wwf_PH_clean.tif -of COG -co COMPRESS=LZW
gdal_translate diff_future_infra_baseline_usle_wwf_PH_clean.tif COG_diff_future_infra_baseline_usle_wwf_PH_clean.tif -of COG -co COMPRESS=LZW
gdal_translate diff_infra_baseline_sed_export_wwf_PH_clean.tif COG_diff_infra_baseline_sed_export_wwf_PH_clean.tif -of COG -co COMPRESS=LZW
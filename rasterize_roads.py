"""Rasterize roads onto rasters."""
import argparse
import os
import logging
import sys
import shutil
import tempfile

from osgeo import osr
from osgeo import ogr
from osgeo import gdal
from ecoshard import geoprocessing
from ecoshard import taskgraph
from ecoshard.geoprocessing.geoprocessing_core import DEFAULT_GTIFF_CREATION_TUPLE_OPTIONS
from ecoshard.geoprocessing.geoprocessing_core import DEFAULT_OSR_AXIS_MAPPING_STRATEGY


RASTER_CREATE_OPTIONS = DEFAULT_GTIFF_CREATION_TUPLE_OPTIONS[1]

logging.basicConfig(
    level=logging.DEBUG,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(levelname)s %(name)s'
        ' [%(pathname)s.%(funcName)s:%(lineno)d] %(message)s'),
    stream=sys.stdout)
LOGGER = logging.getLogger(__name__)
logging.getLogger('ecoshard.taskgraph').setLevel(logging.INFO)


WORKSPACE_DIR = '_workspace_auto_riparian_buffers'


def main():
    """Entry point."""
    parser = argparse.ArgumentParser(description='Rasterize Roads')
    parser.add_argument('base_raster_path', help='Path to base raster.')
    parser.add_argument('road_vector_path', help='Path to road vector.')
    parser.add_argument(
        'buffer_width', type=float, help='how wide of the road effect in m')
    parser.add_argument(
        'road_lulc_val', type=int, help='integer value to rasterize')
    args = parser.parse_args()

    os.makedirs(WORKSPACE_DIR, exist_ok=True)

    road_basename = os.path.splitext(
        os.path.basename(args.road_vector_path))[0]
    target_road_raster_path = (
        f'%s_{road_basename}_{args.buffer_width}%s' % os.path.splitext(
            args.base_raster_path))
    LOGGER.info(
        f'creating local copy of raster at {target_road_raster_path} so '
        f'original is not destroyed')
    if os.path.exists(target_road_raster_path):
        raise ValueError(
            f'error: {target_road_raster_path} already exists, quitting so it '
            f'does not overwrite.')
    gtiff_driver = gdal.GetDriverByName('GTiff')
    base_raster = gdal.OpenEx(args.base_raster_path, gdal.OF_RASTER)
    gtiff_driver.CreateCopy(
        target_road_raster_path, base_raster, options=RASTER_CREATE_OPTIONS)

    working_dir = tempfile.mkdtemp(dir=os.path.dirname(__file__))
    local_road_vector_path = os.path.join(
        working_dir,
        f'{os.path.basename(os.path.splitext(args.road_vector_path)[0])}.gpkg')
    LOGGER.info(
        f'creating buffered/projected version of road vector at '
        f'{local_road_vector_path}')
    raster_info = geoprocessing.get_raster_info(args.base_raster_path)
    pixel_size = raster_info['pixel_size'][0]

    # Create a coordinate transformation
    raster_sr = osr.SpatialReference(raster_info['projection_wkt'])
    raster_sr.SetAxisMappingStrategy(DEFAULT_OSR_AXIS_MAPPING_STRATEGY)
    original_road_vector = ogr.Open(args.road_vector_path)
    original_road_layer = original_road_vector.GetLayer()
    road_sr = original_road_layer.GetSpatialRef()
    road_sr.SetAxisMappingStrategy(DEFAULT_OSR_AXIS_MAPPING_STRATEGY)
    road_to_raster_trans = osr.CreateCoordinateTransformation(
        road_sr, raster_sr)

    gpkg_driver = gdal.GetDriverByName('GPKG')
    road_vector = gpkg_driver.Create(
        local_road_vector_path, 0, 0, 0, gdal.GDT_Unknown)
    road_layer = road_vector.CreateLayer(
        road_basename, raster_sr, ogr.wkbPolygon)
    road_layer.StartTransaction()
    for feature in original_road_layer:
        geom = feature.GetGeometryRef()
        error_code = geom.Transform(road_to_raster_trans)
        if error_code != 0:  # error
            LOGGER.info(f'geom transform error on {feature}')
            continue
        geom = geom.Buffer(args.buffer_width)
        geom = geom.Simplify(pixel_size/2)

        # Copy original_datasource's feature and set as new shapes feature
        target_feature = ogr.Feature(road_layer.GetLayerDefn())
        target_feature.SetGeometry(geom)
        road_layer.CreateFeature(target_feature)
    road_layer.CommitTransaction()
    road_vector.FlushCache()
    road_layer = None
    road_vector = None
    original_road_layer = None
    original_road_vector = None

    geoprocessing.rasterize(
        local_road_vector_path, target_road_raster_path,
        burn_values=[args.road_lulc_val])
    shutil.rmtree(working_dir, ignore_errors=True)
    LOGGER.info(f'rasteriziation of {target_road_raster_path} complete')


if __name__ == '__main__':
    main()

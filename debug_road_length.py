import itertools
import logging
import math
import os
import sys
import time

from ecoshard.geoprocessing.geoprocessing_core import DEFAULT_OSR_AXIS_MAPPING_STRATEGY
from ecoshard.geoprocessing import routing
from ecoshard import geoprocessing
from ecoshard import taskgraph
from osgeo import gdal
from osgeo import ogr
from osgeo import osr
import collections
import numpy
import pandas
ELLIPSOID_EPSG = 6933

IDN_ROAD_VECTOR_PATH = r"./data\infrastructure_polygons\IDN_All_Roads_Merged.gpkg"
IDN_PROVINCE_VECTOR_PATH = r"./data\admin_boundaries\IDN_adm1.gpkg"

VALID_PROVINCE_NAMES = None  # ['National_Capital_Region', 'Region_IV-A']
VALID_COUNTRY_ID = None  # ['PH']
logging.basicConfig(
    level=logging.DEBUG,
    stream=sys.stdout,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(levelname)s %(name)s'
        ' [%(pathname)s.%(funcName)s:%(lineno)d] %(message)s'))
LOGGER = logging.getLogger(__name__)
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
logging.getLogger('PIL').setLevel(logging.ERROR)
logging.getLogger('ecoshard.taskgraph').setLevel(logging.INFO)
logging.getLogger('ecoshard.geoprocessing').setLevel(logging.INFO)


def clip_and_calculate_length_in_km(
        poly_vector_path, line_vector_path, fid_value, epsg_projection):
    poly_vector = gdal.OpenEx(poly_vector_path, gdal.OF_VECTOR)
    poly_layer = poly_vector.GetLayer()
    poly_layer.SetAttributeFilter(f"FID = '{fid_value}'")

    line_vector = gdal.OpenEx(line_vector_path, gdal.OF_VECTOR)
    line_layer = line_vector.GetLayer()

    # Check if spatial references are the same
    poly_srs = poly_layer.GetSpatialRef()
    line_srs = line_layer.GetSpatialRef()

    if not poly_srs.IsSame(line_srs):
        # Create a transformed copy of line_layer in the spatial reference of poly_layer

        # Define the coordinate transformation
        transformed_line_mem = ogr.GetDriverByName('Memory').CreateDataSource('transformed_temp')
        transformed_line_layer = transformed_line_mem.CreateLayer('transformed_lines', srs=poly_srs)

        # Define the coordinate transformation
        coord_transform = osr.CoordinateTransformation(line_srs, poly_srs)

        # Transform each feature
        line_feature = line_layer.GetNextFeature()
        while line_feature:
            geom = line_feature.GetGeometryRef()
            geom.Transform(coord_transform)
            new_feature = ogr.Feature(transformed_line_layer.GetLayerDefn())
            new_feature.SetGeometry(geom)
            transformed_line_layer.CreateFeature(new_feature)
            line_feature = line_layer.GetNextFeature()

        # Reset the line_layer reference to point to the transformed layer
        line_layer = transformed_line_layer

    clipped_lines_mem = ogr.GetDriverByName('Memory').CreateDataSource('temp')
    clipped_lines_layer = clipped_lines_mem.CreateLayer(
        'clipped_roads', srs=poly_srs)

    target_projection = osr.SpatialReference()
    target_projection.ImportFromEPSG(epsg_projection)
    target_projection.SetAxisMappingStrategy(DEFAULT_OSR_AXIS_MAPPING_STRATEGY)

    line_layer.Clip(poly_layer, clipped_lines_layer)
    line_projection = clipped_lines_layer.GetSpatialRef()

    if not line_projection.IsProjected():
        transform = osr.CreateCoordinateTransformation(
            line_layer.GetSpatialRef(), target_projection)
    else:
        transform = None
    total_length = 0
    for line_feature in clipped_lines_layer:
        line_geometry = line_feature.GetGeometryRef()
        if transform is not None:
            err_code = line_geometry.Transform(transform)
            if err_code != 0:
                raise RuntimeError(f'{line_geometry.ExportToWkt()}')
        total_length += line_geometry.Length() / 1000  # convert to km

    return total_length

def calculate_length_in_km_with_raster(
        mask_raster_path, line_vector_path, epsg_projection):
    local_time = time.time()
    temp_raster_path = f'%s_{epsg_projection}_mask_{local_time}%s' % os.path.splitext(
        mask_raster_path)
    geoprocessing.single_thread_raster_calculator(
        [(mask_raster_path, 1)], lambda a: (a > 0).astype(numpy.uint8),
        temp_raster_path,
        gdal.GDT_Byte, None,
        calc_raster_stats=False)

    mask_raster = gdal.OpenEx(temp_raster_path, gdal.OF_RASTER)
    mask_band = mask_raster.GetRasterBand(1)
    mask_projection = osr.SpatialReference(mask_raster.GetProjection())
    mask_projection.SetAxisMappingStrategy(DEFAULT_OSR_AXIS_MAPPING_STRATEGY)
    raster_mem = ogr.GetDriverByName('Memory').CreateDataSource('temp')
    raster_layer = raster_mem.CreateLayer(
        'raster', srs=mask_projection,
        geom_type=ogr.wkbPolygon)
    raster_field = ogr.FieldDefn("value", ogr.OFTInteger)
    raster_layer.CreateField(raster_field)

    # Convert raster to polygons
    LOGGER.debug(f'converting {mask_raster_path} to polygon')
    start_time = time.time()
    gdal.Polygonize(
        mask_band, None, raster_layer, 0, [], callback=None)
    raster_layer.SetAttributeFilter("value = 1")
    LOGGER.debug(
        f'done converting {mask_raster_path} to polygon in '
        f'{time.time()-start_time:.2f}s')

    line_vector = gdal.OpenEx(line_vector_path, gdal.OF_VECTOR)
    line_layer = line_vector.GetLayer()
    line_srs = line_layer.GetSpatialRef()

    if not mask_projection.IsSame(line_srs):
        # Create a transformed copy of line_layer in the spatial reference of poly_layer
        transformed_line_mem = ogr.GetDriverByName('Memory').CreateDataSource('clipped_roads_PRE' + os.path.basename(os.path.splitext(mask_raster_path)[0])+'.gpkg')
        transformed_line_layer = transformed_line_mem.CreateLayer('transformed_lines', srs=mask_projection, geom_type=ogr.wkbLineString)

        # Define the coordinate transformation
        coord_transform = osr.CoordinateTransformation(line_srs, mask_projection)
        line_srs.SetAxisMappingStrategy(DEFAULT_OSR_AXIS_MAPPING_STRATEGY)

        # Transform each feature
        line_feature = line_layer.GetNextFeature()
        while line_feature:
            geom = line_feature.GetGeometryRef()
            geom.Transform(coord_transform)
            new_feature = ogr.Feature(transformed_line_layer.GetLayerDefn())
            new_feature.SetGeometry(geom)
            transformed_line_layer.CreateFeature(new_feature)
            line_feature = line_layer.GetNextFeature()

        # Reset the line_layer reference to point to the transformed layer
        line_layer = transformed_line_layer

    target_projection = osr.SpatialReference()
    target_projection.ImportFromEPSG(epsg_projection)
    target_projection.SetAxisMappingStrategy(DEFAULT_OSR_AXIS_MAPPING_STRATEGY)
    transform = osr.CreateCoordinateTransformation(
        mask_projection, target_projection)

    output_gpkg_path = os.path.join(
        os.path.dirname(mask_raster_path),
        f'clipped_roads_{os.path.basename(os.path.splitext(mask_raster_path)[0])}.gpkg')
    gpkg_driver = ogr.GetDriverByName('GPKG')
    if os.path.exists(output_gpkg_path):
        gpkg_driver.DeleteDataSource(output_gpkg_path)
    clipped_lines_ds = gpkg_driver.CreateDataSource(output_gpkg_path)
    clipped_lines_layer = clipped_lines_ds.CreateLayer(
        'clipped_roads', srs=line_layer.GetSpatialRef(),
        geom_type=ogr.wkbLineString)

    LOGGER.debug(f'clipping {line_vector_path} to polygon')
    start_time = time.time()

    line_layer.Clip(
        raster_layer, clipped_lines_layer)
    LOGGER.debug(
        f'done clippping {clipped_lines_layer} to polygon in '
        f'{time.time()-start_time:.2f}s')

    total_length = 0
    clipped_lines_layer.ResetReading()
    for index, line_feature in enumerate(clipped_lines_layer):
        line_geometry = line_feature.GetGeometryRef()
        try:
            line_geometry.Transform(transform)
        except:
            LOGGER.exception(f'{line_geometry.ExportToWkt()}')

        total_length += line_geometry.Length()

    total_length_km = total_length / 1000

    mask_raster = None
    mask_band = None
    try:
        os.remove(temp_raster_path)
    except:
        pass
    return total_length_km


def main():
    # raw road length
    province_fid = 1
    total_length = clip_and_calculate_length_in_km(
        IDN_PROVINCE_VECTOR_PATH,
        IDN_ROAD_VECTOR_PATH,
        province_fid, ELLIPSOID_EPSG)
    print(f'TOTAL LENGTH: {total_length}')
    local_downstream_coverage_raster_path = './province_dependence_workspace_2024_12_25/province_masks/Aceh_conservation_all_top10_service_downstream_local_coverage.tif'

    in_raster_length = calculate_length_in_km_with_raster(
        local_downstream_coverage_raster_path, IDN_ROAD_VECTOR_PATH,
        ELLIPSOID_EPSG)
    print(f'TOTAL LENGTH: {total_length}\nIN RASTER LENGTH: {in_raster_length}')


if __name__ == '__main__':
    main()

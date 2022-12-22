"""Rasterize the landuse polygons onto a single raster."""
import argparse
import glob
import logging
import os
import subprocess
import sys

from ecoshard import geoprocessing
from ecoshard import taskgraph
from osgeo import gdal
from osgeo import ogr
from osgeo import osr
import numpy

logging.basicConfig(
    level=logging.DEBUG,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(levelname)s %(name)s'
        ' [%(funcName)s:%(lineno)d] %(message)s'))
logging.getLogger('taskgraph').setLevel(logging.INFO)
LOGGER = logging.getLogger(__name__)

TARGET_PIXEL_SIZE = 30.0
CODE_ID = 'id'


def _get_centroid(vector_path):
    """Return x/y centroid of the entire vector."""
    vector = ogr.Open(vector_path)
    layer = vector.GetLayer()
    point_cloud = ogr.Geometry(type=ogr.wkbMultiPoint)
    for feature in layer:
        geom = feature.GetGeometryRef()
        point_cloud.AddGeometry(geom.Centroid())
        geom = None
        feature = None
    layer = None
    vector = None
    centroid = point_cloud.Centroid()
    return centroid


def simplify_poly(base_vector_path, target_vector_path, tol, description_field_id, code_id, description_to_id_map):
    """Simplify base to target."""
    vector = ogr.Open(base_vector_path)
    layer = vector.GetLayer()
    vector_srs = layer.GetSpatialRef()
    if not vector_srs.IsProjected():
        LOGGER.info(
            f'{base_vector_path} is not projected, creating local UTM '
            f'projection based off of centroid')
        centroid = _get_centroid(base_vector_path)
        LOGGER.debug(centroid)
        utm_epsg = geoprocessing.get_utm_zone(centroid.GetX(), centroid.GetY())
        LOGGER.debug(utm_epsg)

        target_srs = osr.SpatialReference()
        target_srs.ImportFromEPSG(utm_epsg)
    else:
        target_srs = vector_srs
    geoprocessing.reproject_vector(
        base_vector_path, target_srs.ExportToWkt(), target_vector_path,
        driver_name='GPKG',
        copy_fields=True,
        geometry_type=ogr.wkbMultiPolygon,
        simplify_tol=TARGET_PIXEL_SIZE/2)

    vector = ogr.Open(target_vector_path, 1)
    layer = vector.GetLayer()
    id_field = ogr.FieldDefn(code_id, ogr.OFTInteger)
    layer.CreateField(id_field)

    layer.StartTransaction()
    for feature in layer:
        feature.SetField(
            code_id, description_to_id_map[feature.GetField(description_field_id)])
        layer.SetFeature(feature)
    layer.CommitTransaction()


def rasterize_id_by_value(vector_path, raster_path, field_id):
    """Rasterize a subset of vector onto raster.

    Args:
        vector_path (str): path to existing vector
        raster_path (str): path to existing raster
        field_id (str): field index in `vector_path` to reference
        field_value (str): field value in the field index of vector to reference
        rasterize_val (int): value to rasterize that matches features with given field value

    Returns:
        None
    """
    geoprocessing.rasterize(
        vector_path, raster_path,
        option_list=["MERGE_ALG=REPLACE", "ALL_TOUCHED=TRUE", f"ATTRIBUTE={field_id}"])


def get_all_field_values(shapefile_path, field_id):
    """Return all values in field_id for all features in ``shapefile_path``."""
    landcover_id_set = set()
    vector = gdal.OpenEx(shapefile_path, gdal.OF_VECTOR)
    layer = vector.GetLayer()
    for feature in layer:
        landcover_id_set.add(feature.GetField(field_id))
    return landcover_id_set


def main():
    parser = argparse.ArgumentParser(description='Rasterize landuse polygons')
    parser.add_argument(
        'vector_path', help='Path to vector(s) to rasterize.')
    parser.add_argument('landcover_field', help='Field in vector that describes the unique landcover')
    parser.add_argument('--tolerance', help='desired resolution of raster in meters (defaults to 30)', type=float, default=30)
    parser.add_argument('--single_raster_mode_name', help='if passed, give name of target raster and generate a single raster rather than separate rasters')
    args = parser.parse_args()

    simplified_vector_dir = './data/simplified_vectors'
    path_to_target_rasters = './data/landcover_rasters/'
    path_to_template_table = './data/biophysical_template.csv'

    os.makedirs(path_to_target_rasters, exist_ok=True)
    os.makedirs(simplified_vector_dir, exist_ok=True)

    task_graph = taskgraph.TaskGraph('.', 4, 15.0)
    landcover_id_set = set()
    field_value_task_list = []
    vector_path_list = glob.glob(args.vector_path)
    for vector_path in vector_path_list:
        basename = os.path.basename(os.path.splitext(vector_path)[0])
        task = task_graph.add_task(
            func=get_all_field_values,
            args=(vector_path, args.landcover_field),
            store_result=True,
            task_name=f'{args.landcover_field} values for {basename}')
        field_value_task_list.append(task)

    for task in field_value_task_list:
        landcover_id_set |= task.get()
    description_to_landcover = {
        description: (i + 1) for (i, description) in enumerate(sorted(landcover_id_set))
    }
    LOGGER.info(f'landcover set: {landcover_id_set}')
    if not os.path.exists(path_to_template_table):
        with open(path_to_template_table, 'w') as table_file:
            table_file.write('lulc_id,lulc_description\n')
            for field_description, field_id in description_to_landcover.items():
                table_file.write(f'{field_id},{field_description}\n')
    vector_info_path_list = []
    for vector_path in vector_path_list:
        LOGGER.info(f'processing {vector_path}')
        basename = os.path.basename(os.path.splitext(vector_path)[0])
        simplified_vector_path = os.path.join(
            simplified_vector_dir, f'{basename}_simple.gpkg')

        simplify_task = task_graph.add_task(
            func=simplify_poly,
            args=(vector_path, simplified_vector_path, TARGET_PIXEL_SIZE/2,
                  args.landcover_field, CODE_ID, description_to_landcover),
            ignore_path_list=[vector_path],
            target_path_list=[simplified_vector_path],
            task_name=f'simplifying {simplified_vector_path}')
        simplify_task.join()

        vector_info = geoprocessing.get_vector_info(simplified_vector_path)
        if args.single_raster_mode_name is not None:
            vector_info_path_list.append(vector_info)
        else:
            xwidth = numpy.subtract(*[vector_info['bounding_box'][i] for i in (2, 0)])
            ywidth = numpy.subtract(*[vector_info['bounding_box'][i] for i in (3, 1)])
            n_cols = int(xwidth / TARGET_PIXEL_SIZE)
            n_rows = int(ywidth / TARGET_PIXEL_SIZE)
            LOGGER.info(f'expected raster size for {basename} is ({n_cols}x{n_rows})')

            target_raster_path = os.path.join(path_to_target_rasters, f'{basename}_lulc.tif')
            if not os.path.exists(target_raster_path):
                geoprocessing.create_raster_from_vector_extents(
                    simplified_vector_path, target_raster_path,
                    (TARGET_PIXEL_SIZE, -TARGET_PIXEL_SIZE), gdal.GDT_Byte, 128)

            task_graph.add_task(
                func=rasterize_id_by_value,
                args=(simplified_vector_path, target_raster_path, CODE_ID),
                dependent_task_list=[simplify_task],
                task_name=(
                    f'rasterizing {simplified_vector_path} to '
                    f'{os.path.basename(target_raster_path)}'))

    if args.single_raster_mode_name is not None:
        # create global bounding box
        single_bounding_box = geoprocessing.merge_bounding_box_list(
            [info['bounding_box'] for info in vector_path_list], 'union')
        xwidth = numpy.subtract(*[single_bounding_box[i] for i in (2, 0)])
        ywidth = numpy.subtract(*[single_bounding_box[i] for i in (3, 1)])
        n_cols = int(xwidth / TARGET_PIXEL_SIZE)
        n_rows = int(ywidth / TARGET_PIXEL_SIZE)
        LOGGER.info(f'expected raster size for {basename} is ({n_cols}x{n_rows})')

        target_raster_path = os.path.join(
            path_to_target_rasters, f'{args.single_raster_mode_name}_lulc.tif')
        if not os.path.exists(target_raster_path):
            geoprocessing.create_raster_from_vector_extents(
                simplified_vector_path, target_raster_path,
                (TARGET_PIXEL_SIZE, -TARGET_PIXEL_SIZE), gdal.GDT_Byte, 128)

        task_graph.add_task(
            func=rasterize_id_by_value,
            args=(simplified_vector_path, target_raster_path, CODE_ID),
            dependent_task_list=[simplify_task],
            task_name=(
                f'rasterizing {simplified_vector_path} to '
                f'{os.path.basename(target_raster_path)}'))

    task_graph.close()
    task_graph.join()


if __name__ == '__main__':
    main()

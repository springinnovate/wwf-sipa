"""Rasterize the landuse polygons onto a single raster."""
import glob
import logging
import os
import subprocess

from ecoshard import geoprocessing
from ecoshard import taskgraph
from osgeo import gdal
from osgeo import ogr
import numpy

logging.basicConfig(
    level=logging.DEBUG,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(levelname)s %(name)s'
        ' [%(funcName)s:%(lineno)d] %(message)s'))
logging.getLogger('taskgraph').setLevel(logging.INFO)
LOGGER = logging.getLogger(__name__)

TARGET_PIXEL_SIZE = 10.0
CODE_ID = 'id'

def simplify_poly(base_vector_path, target_vector_path, tol, description_field_id, code_id, description_to_id_map):
    """Simplify base to target."""
    subprocess.run(
        f'ogr2ogr -simplify {tol} -f GPKG -overwrite {target_vector_path} {base_vector_path}',
        shell=True, check=True)
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
    path_to_shapefiles = './data/land_use_polygons/*'
    simplified_vector_dir = './data/simplified_vectors'
    path_to_target_rasters = './data/landcover_rasters/'
    path_to_template_table = './data/biophysical_template.csv'
    os.makedirs(path_to_target_rasters, exist_ok=True)
    os.makedirs(simplified_vector_dir, exist_ok=True)

    task_graph = taskgraph.TaskGraph('.', 4, 15.0)
    landcover_field = 'AGG12'
    landcover_id_set = set()
    field_value_task_list = []
    shapefile_to_raster_map = {}
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
        if not os.path.exists(target_raster_path):
            geoprocessing.create_raster_from_vector_extents(
                shapefile_path, target_raster_path, (TARGET_PIXEL_SIZE, -TARGET_PIXEL_SIZE),
                gdal.GDT_Byte, 128)
        shapefile_to_raster_map[shapefile_path] = target_raster_path

    for task in field_value_task_list:
        landcover_id_set |= task.get()
    description_to_landcover = {
        description: i+1 for (i, description) in enumerate(sorted(landcover_id_set))
    }
    LOGGER.info(f'landcover set: {landcover_id_set}')
    if not os.path.exists(path_to_template_table):
        with open(path_to_template_table, 'w') as table_file:
            table_file.write('lulc_id,lulc_description\n')
            for field_description, field_id in description_to_landcover.items():
                table_file.write(f'{field_id},{field_description}\n')

    for vector_path, raster_path in shapefile_to_raster_map.items():
        basename = os.path.basename(os.path.splitext(vector_path)[0])
        simplified_vector_path = os.path.join(simplified_vector_dir, f'{basename}_simple.gpkg')
        simplify_task = task_graph.add_task(
            func=simplify_poly,
            args=(vector_path, simplified_vector_path, TARGET_PIXEL_SIZE/2, landcover_field, CODE_ID, description_to_landcover),
            ignore_path_list=[vector_path, simplified_vector_path],
            target_path_list=[simplified_vector_path],
            task_name=f'simplifying {simplified_vector_path}')

        task_graph.add_task(
            func=rasterize_id_by_value,
            args=(vector_path, raster_path, CODE_ID),
            dependent_task_list=[simplify_task],
            task_name=f'rasterizing {vector_path} to  {os.path.basename(raster_path)}')

    task_graph.close()
    task_graph.join()


if __name__ == '__main__':
    main()

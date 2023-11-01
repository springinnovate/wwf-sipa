import argparse
import logging
import sys
import tempfile
import os
import shutil
from ecoshard import geoprocessing
from osgeo import ogr
logging.basicConfig(
    level=logging.DEBUG,
    stream=sys.stdout,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(levelname)s %(name)s'
        ' [%(pathname)s.%(funcName)s:%(lineno)d] %(message)s'))
LOGGER = logging.getLogger(__name__)


def zonal_stats(raster_path, vector_path, table_path):
    """Do zonal stats over base raster in each polygon of the vector."""
    working_dir = tempfile.mkdtemp(
        prefix='zonal_stats_', dir=os.path.dirname(table_path))
    LOGGER.info(f'processing {raster_path}')
    stat_dict = geoprocessing.zonal_statistics(
        (raster_path, 1), vector_path,
        working_dir=working_dir,
        clean_working_dir=True,
        polygons_might_overlap=False)

    source_ds = ogr.Open("vector_path", 0)
    source_layer = source_ds.GetLayer()

    # Create the target Geopackage
    driver = ogr.GetDriverByName("GPKG")
    target_ds = driver.CreateDataSource("copy.gpkg")

    # Create the target layer with the same schema as the source layer
    target_layer = target_ds.CreateLayer("layer_name", geom_type=source_layer.GetGeomType())
    target_layer.CreateFields(source_layer.schema)

    # Add two new floating point fields
    new_field1 = ogr.FieldDefn("new_field1", ogr.OFTReal)
    new_field2 = ogr.FieldDefn("new_field2", ogr.OFTReal)
    target_layer.CreateField(new_field1)
    target_layer.CreateField(new_field2)

    # Copy features from source layer to target layer and populate new fields
    for feature in source_layer:
        new_feature = ogr.Feature(target_layer.GetLayerDefn())
        new_feature.SetFrom(feature)

        # Set values for the new fields (optional)
        new_feature.SetField("new_field1", 0.0)
        new_feature.SetField("new_field2", 0.0)

        target_layer.CreateFeature(new_feature)

    # Cleanup
    source_ds = None
    target_ds = None



    stat_list = ['count', 'max', 'min', 'nodata_count', 'sum']
    LOGGER.info(f'*********** building table at {table_path}')
    with open(table_path, 'w') as table_file:
        table_file.write(f'{raster_path}\n{vector_path}\n')
        table_file.write('fid,')
        table_file.write(f'{",".join(stat_list)},mean\n')
        for fid, stats in stat_dict.items():
            table_file.write(f'{fid},')
            for stat_id in stat_list:
                table_file.write(f'{stats[stat_id]},')
            if stats['count'] > 0:
                table_file.write(f'{stats["sum"]/stats["count"]}')
            else:
                table_file.write('NaN')
            table_file.write('\n')
    shutil.rmtree(working_dir)
    LOGGER.info(f'all done, table at {table_path}')


def main():
    LOGGER.debug('starting')
    parser = argparse.ArgumentParser(description='Global CV analysis')
    parser.add_argument('raster_path', help='Raster to aggregate up')
    parser.add_argument('vector_path', help='Vector to aggregate across')
    args = parser.parse_args()

    # do zonal stats
    # Area of coverage / Area of the unit – proportional area 50% is important to at least 1 service – proportional area
    # Average of the non-zeros - intensity

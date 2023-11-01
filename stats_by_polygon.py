import argparse
import logging
import sys
import tempfile
import os
import shutil
from ecoshard import geoprocessing
from ecoshard import taskgraph

from osgeo import ogr
logging.basicConfig(
    level=logging.DEBUG,
    stream=sys.stdout,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(levelname)s %(name)s'
        ' [%(pathname)s.%(funcName)s:%(lineno)d] %(message)s'))
LOGGER = logging.getLogger(__name__)


aggregate_vector = './data/admin_boundaries/IDN_gdam3.gpkg'

rasters_to_process = {
    'IDN_10th_percentile_service_conservation_dspop': 'idn_emergency_data/10_IDN_conservation_inf_dspop__service_overlap_count.tif',
    'IDN_10th_percentile_service_conservation_road': 'idn_emergency_data/10_IDN_conservation_inf_road__service_overlap_count.tif',
    'IDN_10th_percentile_service_restoration_dspop': 'idn_emergency_data/10_IDN_restoration_dspop__service_overlap_count.tif',
    'IDN_10th_percentile_service_restoration_road': 'idn_emergency_data/10_IDN_restoration_road__service_overlap_count.tif',
}


def local_zonal_stats(raster_path):
    working_dir = tempfile.mkdtemp(
        prefix='zonal_stats_', dir=os.path.dirname(__file__))
    fixed_raster_path = os.path.join(
            working_dir, os.path.basename(raster_path))
    zero_to_nodata(raster_path, fixed_raster_path)
    stat_dict = geoprocessing.zonal_statistics(
        (fixed_raster_path, 1), aggregate_vector,
        working_dir=working_dir,
        clean_working_dir=True,
        polygons_might_overlap=False)
    shutil.rmtree(working_dir)
    return stat_dict


def zero_to_nodata(base_raster_path, target_raster_path):
    raster_info = geoprocessing.get_raster_info(base_raster_path)
    nodata = raster_info['nodata'][0]

    def _op(base_array):
        result = base_array.copy()
        result[result == 0] = nodata
        return result

    geoprocessing.raster_calculator(
        [(base_raster_path, 1)], _op, target_raster_path,
        raster_info['datatype'], nodata)


def zonal_stats():
    """Do zonal stats over base raster in each polygon of the vector."""
    task_graph = taskgraph.TaskGraph(os.path.dirname(__file__), len(rasters_to_process), 10.0)

    zonal_results = {}
    for key, raster_path in rasters_to_process.items():
        LOGGER.info(f'processing {key}')
        zonal_stats_task = task_graph.add_task(
            func=local_zonal_stats,
            args=(raster_path),
            store_result=True,
            task_name=f'stats for {key}')
        zonal_results[key] = zonal_stats_task
    task_graph.join()
    task_graph.close()

    source_ds = ogr.Open(aggregate_vector, 0)
    source_layer = source_ds.GetLayer()
    driver = ogr.GetDriverByName("GPKG")
    for key, task in zonal_results.items():
        stat_dict = task.get()

        # Create the target Geopackage
        target_vector_path = f"{key}.gpkg"
        if os.path.exists(target_vector_path):
            os.remove(target_vector_path)
        target_ds = driver.CreateDataSource(f"{key}.gpkg")

        # Create the target layer with the same schema as the source layer
        target_layer = target_ds.CreateLayer(key, geom_type=source_layer.GetGeomType())
        target_layer.CreateFields(source_layer.schema)

        # Add two new floating point fields
        target_layer.CreateField(
            ogr.FieldDefn("proportional_service_area", ogr.OFTReal))
        target_layer.CreateField(
            ogr.FieldDefn("service_intensity", ogr.OFTReal))

        # Copy features from source layer to target layer and populate new fields
        source_layer.ResetReading()
        for feature in source_layer:
            new_feature = ogr.Feature(target_layer.GetLayerDefn())
            new_feature.SetFrom(feature)
            fid = feature.GetFID()
            try:
                proportional_service_area = stat_dict[fid]['count']/(stat_dict[fid]['count']+stat_dict[fid]['nodata_count'])
                service_intensity = stat_dict[fid]['sum']/stat_dict[fid]['count']
            except ZeroDivisionError:
                proportional_service_area = 0
                service_intensity = 0
            # Set values for the new fields (optional)
            new_feature.SetField(
                "proportional_service_area", proportional_service_area)
            new_feature.SetField(
                "service_intensity", service_intensity)
            target_layer.CreateFeature(new_feature)

        target_layer = None
        target_ds = None
    # Cleanup
    source_ds = None
    target_ds = None


if __name__ == '__main__':
    zonal_stats()
import argparse
import logging
import sys
import tempfile
import os
import shutil
from ecoshard import geoprocessing
from ecoshard import taskgraph
import numpy

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
    'IDN_25th_percentile_service_conservation': ('idn_emergency_data/25_IDN_conservation_inf_dspop__service_overlap_count.tif', 'idn_emergency_data/25_IDN_conservation_inf_road__service_overlap_count.tif'),
    'IDN_25th_percentile_service_restoration': ('idn_emergency_data/25_IDN_restoration_dspop__service_overlap_count.tif', 'idn_emergency_data/25_IDN_restoration_road__service_overlap_count.tif'),
}


def local_zonal_stats(prefix, raster_path_list):
    working_dir = tempfile.mkdtemp(
        prefix='zonal_stats_', dir=os.path.dirname(__file__))
    fixed_raster_path = os.path.join(
            working_dir, f'{prefix}.tif')
    sum_zero_to_nodata(raster_path_list, fixed_raster_path)
    stat_dict = geoprocessing.zonal_statistics(
        (fixed_raster_path, 1), aggregate_vector,
        working_dir=working_dir,
        clean_working_dir=True,
        polygons_might_overlap=False)
    shutil.rmtree(working_dir)
    return stat_dict


def sum_zero_to_nodata(base_raster_path_list, target_raster_path):
    raster_info = geoprocessing.get_raster_info(base_raster_path_list[0])
    global_nodata = raster_info['nodata'][0]
    if global_nodata is None:
        global_nodata = 0

    nodata_list = [
        geoprocessing.get_raster_info(path)['nodata'][0]
        for path in base_raster_path_list]

    def _op(*base_array_list):
        result = numpy.zeros(base_array_list[0].shape)
        running_valid_mask = numpy.zeros(result.shape, dtype=bool)
        for base_array, local_nodata in zip(base_array_list, nodata_list):
            valid_mask = base_array != 0
            if local_nodata is not None:
                valid_mask = valid_mask & (base_array != local_nodata)
            result[valid_mask] += base_array[valid_mask]
            running_valid_mask |= valid_mask
        result[result == 0] = global_nodata
        return result

    geoprocessing.raster_calculator(
        [(path, 1) for path in base_raster_path_list], _op, target_raster_path,
        raster_info['datatype'], global_nodata,
        allow_different_blocksize=True)


def zonal_stats():
    """Do zonal stats over base raster in each polygon of the vector."""
    task_graph = taskgraph.TaskGraph(os.path.dirname(__file__), len(rasters_to_process), 10.0)

    zonal_results = {}
    for key, raster_path_list in rasters_to_process.items():
        LOGGER.info(f'processing {key}')
        zonal_stats_task = task_graph.add_task(
            func=local_zonal_stats,
            args=(key, raster_path_list,),
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
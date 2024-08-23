import logging
import numpy
import os
import sys

import pandas
import geopandas
from ecoshard import geoprocessing
from ecoshard import taskgraph
from osgeo import gdal

logging.basicConfig(
    level=logging.DEBUG,
    stream=sys.stdout,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(levelname)s %(name)s'
        ' [%(funcName)s:%(lineno)d] %(message)s'))
logging.getLogger('taskgraph').setLevel(logging.WARNING)
LOGGER = logging.getLogger(__name__)

SERVICE_INTENSITY_WORKSPACE = 'service_intensity_workspace'
os.makedirs(SERVICE_INTENSITY_WORKSPACE, exist_ok=True)


def avg_valid_pixels(array_a, array_b):
    result = numpy.zeros(array_a.shape)
    valid_mask = numpy.ones(array_a.shape, dtype=bool)
    avg_mask = numpy.zeros(array_a.shape)
    for array in [array_a, array_b]:
        local_valid_mask = (array > 0)
        avg_mask += local_valid_mask
        valid_mask &= local_valid_mask
        result[local_valid_mask] += array[local_valid_mask]
    result[valid_mask] /= avg_mask[valid_mask]
    return result


def main():

    task_graph = taskgraph.TaskGraph(SERVICE_INTENSITY_WORKSPACE, -1)
    muni_vector_path = r"D:\repositories\wwf-sipa\data\admin_boundaries\IDN_gdam3.gpkg"

    cons_dspop_path = r"D:\repositories\wwf-sipa\post_processing_results_no_road_recharge\10_IDN_conservation_inf_dspop_service_overlap_count.tif"
    cons_road_path = r"D:\repositories\wwf-sipa\post_processing_results_no_road_recharge\10_IDN_conservation_inf_road_service_overlap_count.tif"
    avg_cons_overlap_path = os.path.join(SERVICE_INTENSITY_WORKSPACE, 'avg_cons_overlaps.tif')

    task_graph.add_task(
        func=geoprocessing.raster_calculator,
        args=(
            [(cons_dspop_path, 1), (cons_road_path, 1)], avg_valid_pixels, avg_cons_overlap_path,
            gdal.GDT_Float32, 0),
        kwargs={'skip_sparse': True, 'allow_different_blocksize': True},
        target_path_list=[avg_cons_overlap_path])

    cons_stats = task_graph.add_task(
        func=geoprocessing.zonal_statistics,
        args=((avg_cons_overlap_path, 1), muni_vector_path),
        kwargs={'polygons_might_overlap': False},
        store_result=True)

    rest_dspop_path = r"D:\repositories\wwf-sipa\post_processing_results_no_road_recharge\10_IDN_restoration_dspop_service_overlap_count.tif"
    rest_road_path = r"D:\repositories\wwf-sipa\post_processing_results_no_road_recharge\10_IDN_restoration_road_service_overlap_count.tif"
    avg_rest_overlap_path = os.path.join(SERVICE_INTENSITY_WORKSPACE, 'avg_rest_overlaps.tif')

    task_graph.add_task(
        func=geoprocessing.raster_calculator,
        args=(
            [(rest_dspop_path, 1), (rest_road_path, 1)], avg_valid_pixels, avg_rest_overlap_path,
            gdal.GDT_Float32, 0),
        kwargs={'skip_sparse': True, 'allow_different_blocksize': True},
        target_path_list=[avg_rest_overlap_path])

    rest_stats = task_graph.add_task(
        func=geoprocessing.zonal_statistics,
        args=((avg_rest_overlap_path, 1), muni_vector_path),
        kwargs={'polygons_might_overlap': False},
        store_result=True)

    for type_name, stats in [('restoration', rest_stats.get()), ('conservation', cons_stats.get())]:
        muni_vector = gdal.OpenEx(muni_vector_path, gdal.OF_VECTOR)
        muni_layer = muni_vector.GetLayer()

        spatial_ref_wkt = muni_layer.GetSpatialRef().ExportToWkt()

        schema = muni_layer.GetLayerDefn()
        columns = [schema.GetFieldDefn(i).GetName() for i in range(schema.GetFieldCount())]
        columns.extend(['service_intensity', 'proportional_service_area'])
        muni_zonal_gdf = geopandas.GeoDataFrame(columns=columns)
        muni_zonal_gdf['geometry'] = None
        gdf_list = []
        for muni_feature in muni_layer:
            # Convert the feature to a GeoDataFrame
            geom = muni_feature.GetGeometryRef().Clone()
            geom_wkt = geom.ExportToWkt()
            feature_dict = {field: muni_feature.GetField(field) for field in columns if field not in ['service_intensity', 'proportional_service_area']}
            feature_dict['geometry'] = geopandas.GeoSeries.from_wkt([geom_wkt])[0]

            # Add additional fields
            fid = muni_feature.GetFID()
            try:
                feature_dict['service_intensity'] = stats[fid]['sum']/stats[fid]['count']
            except ZeroDivisionError:
                feature_dict['service_intensity'] = 0
            try:
                feature_dict['proportional_service_area'] = stats[fid]['count']/(stats[fid]['count']+stats[fid]['nodata_count'])
            except ZeroDivisionError:
                feature_dict['proportional_service_area'] = 0

            # Append the feature to the GeoDataFrame
            feature_gdf = geopandas.GeoDataFrame([feature_dict], columns=columns + ['geometry'])
            gdf_list.append(feature_gdf)
            #muni_zonal_gdf = muni_zonal_gdf.append(feature_dict, ignore_index=True)

        muni_zonal_gdf = pandas.concat(gdf_list, ignore_index=True)

        muni_zonal_gdf.set_crs(spatial_ref_wkt, inplace=True)
        # Save the GeoDataFrame as a GeoPackage
        muni_zonal_gdf.to_file(f'muni_{type_name}_zonal.gpkg', driver='GPKG')


if __name__ == '__main__':
    main()
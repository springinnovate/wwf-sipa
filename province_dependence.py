"""What are the upstream/downstream dependancies between provinces?"""
import logging
import os
import sys
import time

from ecoshard.geoprocessing import routing
from ecoshard import geoprocessing
from ecoshard import taskgraph
from osgeo import gdal
from osgeo import ogr
from osgeo import osr
import numpy

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


IDN_PROViNCE_VECTOR_PATH = r"D:\repositories\wwf-sipa\data\admin_boundaries\IDN_adm1.gpkg"
PH_PROViNCE_VECTOR_PATH = r"D:\repositories\wwf-sipa\data\admin_boundaries\PH_adm1.gpkg"

IDN_DEM_PATH = r"D:\repositories\wwf-sipa\data\idn_dem.tif"
PH_DEM_PATH = r"D:\repositories\wwf-sipa\data\ph_dem.tif"

PH_EPSG_PROJECTION = 3121
IDN_EPSG_PROJECTION = 23830

IDN_POP_RASTER_PATH = r"D:\repositories\wwf-sipa\data\pop\idn_ppp_2020.tif"
PH_POP_RASTER_PATH = r"D:\repositories\wwf-sipa\data\pop\phl_ppp_2020.tif"

PH_ROAD_VECTOR_PATH = r"D:\repositories\wwf-sipa\data\infrastructure_polygons\PH_All_Roads_Merged.gpkg"
IDN_ROAD_VECTOR_PATH = r"D:\repositories\wwf-sipa\data\infrastructure_polygons\IDN_All_Roads_Merged.gpkg"

WORKSPACE_DIR = 'province_depedence_workspace'
MASK_DIR = os.path.join(WORKSPACE_DIR, 'province_masks')
SERVICE_DIR = os.path.join(WORKSPACE_DIR, 'masked_services')
for dir_path in [WORKSPACE_DIR, MASK_DIR, SERVICE_DIR]:
    os.makedirs(dir_path, exist_ok=True)


def basefilename(path):
    return os.path.basename(os.path.splitext(path)[0])


def mask_raster(base_raster_path, mask_raster_path, target_raster_path):
    def _mask_raster(array, mask_array):
        result = numpy.zeros(array.shape)
        valid_mask = mask_array > 0
        result[valid_mask] = array[valid_mask]
        return result

    geoprocessing.raster_calculator(
        [(base_raster_path, 1), (mask_raster_path, 1)], _mask_raster,
        target_raster_path, gdal.GDT_Float32, None, skip_sparse=True,
        raster_driver_creation_tuple=('GTIFF', (
            'TILED=YES', 'BIGTIFF=YES', 'COMPRESS=LZW',
            'BLOCKXSIZE=256', 'BLOCKYSIZE=256', 'SPARSE_OK=TRUE')))


def calculate_sum_over_mask(base_raster_path, mask_raster_path):
    running_sum = 0
    for base_array, mask_array in zip(
            geoprocessing.iterblocks(
                (base_raster_path, 1), skip_sparse=True),
            geoprocessing.iterblocks(
                (mask_raster_path, 1), skip_sparse=True)):
        running_sum += base_raster_path[
            (mask_raster_path > 0) & (base_raster_path > 0)]
    return running_sum


def calculate_pixel_area_km2(base_raster_path, target_epsg):
    source_raster = gdal.OpenEx(base_raster_path, gdal.OF_RASTER)
    target_srs = osr.SpatialReference()
    target_srs.ImportFromEPSG(target_epsg)

    LOGGER.debug(f'about to warp {base_raster_path} to epsg:{target_epsg}')
    reprojected_raster = gdal.Warp(
        '',
        source_raster,
        format='MEM',
        dstSRS=target_srs)
    reprojected_band = reprojected_raster.GetRasterBand(1)
    nodata = reprojected_band.GetNoDataValue()
    LOGGER.debug(f'nodata value is {nodata}')
    LOGGER.debug(f'read warped {base_raster_path}')
    reprojected_data = reprojected_band.ReadAsArray()
    reprojected_geotransform = reprojected_raster.GetGeoTransform()
    reprojected_pixel_width, reprojected_pixel_height = (
        reprojected_geotransform[1], abs(reprojected_geotransform[5]))

    # Calculate the area of pixels with values > 1
    pixel_area = reprojected_pixel_width * reprojected_pixel_height
    LOGGER.debug('sum it up')
    count = ((reprojected_data > 0) & (reprojected_data != nodata)).sum()
    total_area = count * pixel_area / 1e6  # covert to km2

    LOGGER.debug(f'total area of {base_raster_path}: {total_area}km2')
    return total_area


def clip_and_calculate_length_in_km(
        poly_vector_path, line_vector_path, fid_value, epsg_projection):
    poly_vector = gdal.OpenEx(poly_vector_path, gdal.GA_VECTOR)
    poly_layer = poly_vector.GetLayer()
    poly_layer.SetAttributeFilter(f"FID = '{fid_value}'")
    poly_feature = poly_layer.GetNextFeature()
    poly_geometry = poly_feature.GetGeometryRef()

    line_vector = gdal.OpenEx(line_vector_path, gdal.GA_VECTOR)
    line_layer = line_vector.GetLayer()

    clipped_roads_mem = ogr.GetDriverByName('Memory').CreateDataSource('temp')
    clipped_roads_layer = clipped_roads_mem.CreateLayer('clipped_roads')

    target_projection = osr.SpatialReference()
    target_projection.ImportFromEPSG(epsg_projection)

    line_layer.Clip(poly_geometry, clipped_roads_layer)
    transform = osr.CoordinateTransformation(
        clipped_roads_layer.GetSpatialRef(), target_projection)
    total_length = 0
    for line_feature in clipped_roads_layer:
        geometry = line_feature.GetGeometryRef()
        geometry.Transform(transform)
        total_length += geometry.Length() / 1000  # convert to km

    poly_vector = None
    line_vector = None
    clipped_roads_mem = None

    return total_length


def calculate_length_in_km_with_raster(mask_raster_path, line_vector_path, epsg_projection):
    line_ds = ogr.Open(line_vector_path)
    line_layer = line_ds.GetLayer()

    mask_raster = gdal.OpenEx(
        mask_raster_path, gdal.GA_RASTER | gdal.OF_ReadOnly)
    mask_band = mask_raster.GetRasterBand()
    mask_band.WriteArray(mask_band.ReadAsArray() > 0)
    projection = mask_raster.GetProjection()

    raster_mem = ogr.GetDriverByName('Memory').CreateDataSource('temp')
    raster_layer = raster_mem.CreateLayer(
        'raster', srs=ogr.osr.SpatialReference(projection),
        geom_type=ogr.wkbPolygon)
    raster_field = ogr.FieldDefn("value", ogr.OFTInteger)
    raster_layer.CreateField(raster_field)

    # Convert raster to polygons
    LOGGER.debug(f'converting {mask_raster_path} to polygon')
    start_time = time.time()
    # Polygonize(Band srcBand, Band maskBand, Layer outLayer, int iPixValField, char ** options=None, GDALProgressFunc callback=0, void * callback_data=None) -> int"""
    gdal.Polygonize(
        mask_raster.GetRasterBand(1), None, raster_layer, 0, [], callback=None)
    LOGGER.debug(
        f'done converting {mask_raster_path} to polygon in '
        f'{time.time()-start_time:.2f}s')

    target_projection = osr.SpatialReference()
    target_projection.ImportFromWkt(epsg_projection)
    transform = osr.CoordinateTransformation(projection, target_projection)

    total_length = 0
    for line_feature in line_layer:
        line_geometry = line_feature.GetGeometryRef()
        line_geometry.Transform(transform)
        for raster_feature in raster_layer:
            if raster_feature.GetValue('value') <= 0:
                continue
            raster_geometry = raster_feature.GetGeometryRef()
            raster_geometry.Transform(transform)
            if line_geometry.Intersects(raster_geometry):
                clipped_geometry = line_geometry.Intersection(raster_geometry)
                total_length += clipped_geometry.Length()

    total_length_km = total_length / 1000
    return total_length_km


SCENARIO_LIST = ['restoration', 'conservation']

TOP10_SERVICE_COVERAGE_RASTERS = {
    ('PH', 'restoration'): r"D:\repositories\wwf-sipa\fig_generator_dir\overlap_rasters\overlap_combos_top_10_PH_restoration_each ecosystem service.tif",
    ('IDN', 'restoration'): r"D:\repositories\wwf-sipa\fig_generator_dir\overlap_rasters\overlap_combos_top_10_IDN_restoration_each ecosystem service.tif",
    ('PH', 'conservation'): r"D:\repositories\wwf-sipa\fig_generator_dir\overlap_rasters\overlap_combos_top_10_PH_conservation_each ecosystem service.tif",
    ('IDN', 'conservation'): r"D:\repositories\wwf-sipa\fig_generator_dir\overlap_rasters\overlap_combos_top_10_IDN_conservation_each ecosystem service.tif",
}


def main():
    task_graph = taskgraph.TaskGraph(WORKSPACE_DIR, -1)

    for (country_id,
         dem_path,
         province_vector_path,
         province_name_key,
         epsg_projection,
         pop_raster_path,
         road_vector_path) in [
            ('IDN',
             IDN_DEM_PATH,
             IDN_PROViNCE_VECTOR_PATH,
             'NAME_1',
             IDN_EPSG_PROJECTION,
             IDN_POP_RASTER_PATH,
             IDN_ROAD_VECTOR_PATH),
            ('PH',
             PH_DEM_PATH,
             PH_PROViNCE_VECTOR_PATH,
             'ADM1_EN',
             PH_EPSG_PROJECTION,
             PH_POP_RASTER_PATH,
             PH_ROAD_VECTOR_PATH)]:
        flow_dir_path = os.path.join(WORKSPACE_DIR, basefilename(dem_path))
        routing_task = task_graph.add_task(
            func=routing.flow_dir_mfd,
            args=((dem_path, 1), flow_dir_path),
            kwargs={
                'working_dir': WORKSPACE_DIR,
                'raster_driver_creation_tuple': ('GTIFF', (
                    'TILED=YES', 'BIGTIFF=YES', 'COMPRESS=LZW',
                    'BLOCKXSIZE=256', 'BLOCKYSIZE=256', 'SPARSE_OK=TRUE'))
            },
            target_path_list=[flow_dir_path],
            task_name=f'flow dir {flow_dir_path}')

        current_projection_wkt = geoprocessing.get_vector_info(
            province_vector_path)['projection_wkt']
        dem_pixel_size = abs(
            geoprocessing.get_raster_info(dem_path)['pixel_size'][0])
        simplified_vector_path = os.path.join(
            WORKSPACE_DIR, f'simple_{basefilename(province_vector_path)}.tif')
        simplify_vector_task = task_graph.add_task(
            func=geoprocessing.reproject_vector,
            args=(
                province_vector_path, current_projection_wkt,
                simplified_vector_path),
            kwargs={
                'simplify_tol': dem_pixel_size,
            },
            target_path_list=[simplified_vector_path],
            ignore_path_list=[province_vector_path, simplified_vector_path],
            task_name=f'simplify {province_vector_path}')

        simplify_vector_task.join()
        vector = gdal.OpenEx(
            simplified_vector_path, gdal.OF_RASTER | gdal.GA_ReadOnly)
        layer = vector.getLayer()
        for feature in layer:
            province_fid = feature.GetFID()
            province_name = feature.GetValue(province_name_key)
            province_mask_path = os.path.join(MASK_DIR, f'{province_name}.tif')

            new_raster_task = task_graph.add_task(
                func=geoprocessing.new_raster_from_base,
                args=(dem_path, province_mask_path, gdal.GDT_Float32, [0]),
                target_path_list=[province_mask_path],
                ignore_path_list=[province_mask_path],
                kwargs={
                    'raster_driver_creation_tuple': ('GTIFF', (
                        'TILED=YES', 'BIGTIFF=YES', 'COMPRESS=LZW',
                        'BLOCKXSIZE=256', 'BLOCKYSIZE=256', 'SPARSE_OK=TRUE'))
                },
                task_name=f'make mask for {province_mask_path}')

            rasterize_province_task = task_graph.add_task(
                func=geoprocessing.rasterize,
                args=(
                    simplified_vector_path,
                    province_mask_path),
                kwargs={
                    'burn_values': None,
                    'where_clause': f"FID = '{province_fid}'",
                    'raster_driver_creation_tuple': ('GTIFF', (
                        'TILED=YES', 'BIGTIFF=YES', 'COMPRESS=LZW',
                        'BLOCKXSIZE=256', 'BLOCKYSIZE=256', 'SPARSE_OK=TRUE'))
                },
                dependent_task_list=[
                    simplify_vector_task, routing_task, new_raster_task],
                target_path_list=[province_mask_path],
                task_name=f'masking {province_name}')

            # calculate area of province
            province_area_task = task_graph.add_task(
                func=calculate_pixel_area_km2,
                args=(province_mask_path, epsg_projection),
                dependent_task_list=[rasterize_province_task],
                store_result=True,
                task_name=f'calculate area of {province_name}')

            # # of people in province
            pop_count_task = task_graph.add_task(
                func=calculate_sum_over_mask,
                args=(province_mask_path, pop_raster_path),
                dependent_task_list=[rasterize_province_task],
                store_result=True,
                task_name=f'calculate pope in {province_name}')

            # length of roads in km
            length_of_roads_task = task_graph.add_task(
                func=clip_and_calculate_length_in_km,
                args=(
                    province_vector_path, road_vector_path,
                    province_fid, epsg_projection),
                ignore_path_list=[province_vector_path, road_vector_path],
                store_result=True,
                task_name=f'road length for {province_fid}')

            for scenario in SCENARIO_LIST:

                # TODO: check what's here i was doing it global before so files etc are wrong

                # mask TOP10_SERVICE to the current province
                masked_service_raster_path = os.path.join(
                    SERVICE_DIR,
                    f'{country_id}_{province_name}_{scenario}_top10.tif')
                mask_service_task = task_graph.add_task(
                    func=mask_raster,
                    args=(
                        TOP10_SERVICE_COVERAGE_RASTERS[(country_id, scenario)],
                        province_mask_path,
                        masked_service_raster_path),
                    dependent_task_list=[rasterize_province_task],
                    target_path_list=[masked_service_raster_path],
                    task_name=f'masking service {province_name} {scenario}')

                # area of the mask of top 10 service in the province
                service_area_task = task_graph.add_task(
                    func=calculate_pixel_area_km2,
                    args=(masked_service_raster_path, epsg_projection),
                    dependent_task_list=[mask_service_task],
                    store_result=True,
                    task_name=f'calculate area {masked_service_raster_path}')

                # downstream areas of top 10 service
                global_downstream_coverage_raster_path = os.path.join(
                    MASK_DIR, f'{province_name}_{scenario}_top10_service_downstream_global_coverage.tif')
                downstream_coverage_task = task_graph.add_task(
                    func=routing.flow_accumulation_mfd,
                    args=(
                        flow_dir_path,
                        global_downstream_coverage_raster_path),
                    kwargs={
                        'weight_raster_path_band': (
                            masked_service_raster_path, 1),
                        'raster_driver_creation_tuple': ('GTIFF', (
                            'TILED=YES', 'BIGTIFF=YES', 'COMPRESS=LZW',
                            'BLOCKXSIZE=256', 'BLOCKYSIZE=256',
                            'SPARSE_OK=TRUE'))
                    },
                    dependent_task_list=[rasterize_province_task, routing_task],
                    target_path_list=[global_downstream_coverage_raster_path],
                    task_name=f'flow accum of service {scenario} down from {province_name}')
                local_downstream_coverage_raster_path = os.path.join(
                    MASK_DIR,
                    f'{province_name}_{scenario}_top10_service_downstream_'
                    'local_coverage.tif')
                local_mask_service_task = task_graph.add_task(
                    func=mask_raster,
                    args=(
                        global_downstream_coverage_raster_path,
                        province_mask_path,
                        local_downstream_coverage_raster_path),
                    dependent_task_list=[
                        downstream_coverage_task, rasterize_province_task],
                    target_path_list=[local_downstream_coverage_raster_path],
                    task_name=f'masking service {province_name} {scenario}')
                local_downstream_service_area_task = task_graph.add_task(
                    func=calculate_pixel_area_km2,
                    args=(local_downstream_coverage_raster_path, epsg_projection),
                    dependent_task_list=[local_mask_service_task],
                    store_result=True,
                    task_name=f'calculate area {local_downstream_coverage_raster_path}')
                # calculate area of total downstream top 10% areas
                global_downstream_service_area_task = task_graph.add_task(
                    func=calculate_pixel_area_km2,
                    args=(global_downstream_coverage_raster_path, epsg_projection),
                    dependent_task_list=[downstream_coverage_task],
                    store_result=True,
                    task_name=f'calculate area {global_downstream_coverage_raster_path}')

                # calculate number of people on the ds mask in the province
                local_ds_service_pop_count_task = task_graph.add_task(
                    func=calculate_sum_over_mask,
                    args=(local_downstream_coverage_raster_path, pop_raster_path),
                    dependent_task_list=[local_downstream_service_area_task],
                    store_result=True,
                    task_name=f'calculate people in {local_downstream_coverage_raster_path}')

                # calculate km of roads on the ds mask in the province
                local_ds_length_of_roads_task = task_graph.add_task(
                    func=calculate_length_in_km_with_raster,
                    args=(
                        local_downstream_coverage_raster_path, road_vector_path,
                        epsg_projection),
                    ignore_path_list=[road_vector_path],
                    store_result=True,
                    task_name=f'road length for {province_fid} {scenario}')

                result_dict = {
                    'province name': province_name,
                    'province_area': province_area_task.get(),
                    'pop count': pop_count_task.get(),
                    'length of roads km': length_of_roads_task.get(),
                    r'top 10% service area': service_area_task.get(),
                    'top 10% service area downstream': global_downstream_service_area_task.get(),
                    'pop count in top 10% local downstream service': local_ds_service_pop_count_task.get(),
                    'length of roads in km in top 10% local downstream service': local_ds_length_of_roads_task.get(),
                }
                LOGGER.debug(result_dict)
                return

    task_graph.join()
    task_graph.close()


if __name__ == '__main__':
    main()

"""What are the upstream/downstream dependancies between provinces?"""
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

logging.basicConfig(
    level=logging.DEBUG,
    stream=sys.stdout,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(levelname)s %(name)s'
        ' [%(pathname)s.%(funcName)s:%(lineno)d] %(message)s'))
LOGGER = logging.getLogger(__name__)
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
logging.getLogger('PIL').setLevel(logging.ERROR)
logging.getLogger('ecoshard.taskgraph').setLevel(logging.DEBUG)
logging.getLogger('ecoshard.geoprocessing').setLevel(logging.INFO)

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

WORKSPACE_DIR = 'province_dependence_workspace'
MASK_DIR = os.path.join(WORKSPACE_DIR, 'province_masks')
SERVICE_DIR = os.path.join(WORKSPACE_DIR, 'masked_services')
ALIGNED_DIR = os.path.join(WORKSPACE_DIR, 'aligned_rasters')
DOWNSTREAM_COVERAGE_DIR = os.path.join(WORKSPACE_DIR, 'downstream_rasters')
DEM_DIR = os.path.join(WORKSPACE_DIR, 'filled_dems')
for dir_path in [
        WORKSPACE_DIR, MASK_DIR, SERVICE_DIR, ALIGNED_DIR,
        DOWNSTREAM_COVERAGE_DIR, DEM_DIR]:
    os.makedirs(dir_path, exist_ok=True)


def area_of_pixel(pixel_size, center_lat):
    """Calculate m^2 area of a wgs84 square pixel.

    Adapted from: https://gis.stackexchange.com/a/127327/2397

    Args:
        pixel_size (float): length of side of pixel in degrees.
        center_lat (float): latitude of the center of the pixel. Note this
            value +/- half the `pixel-size` must not exceed 90/-90 degrees
            latitude or an invalid area will be calculated.

    Returns:
        Area of square pixel of side length `pixel_size` centered at
        `center_lat` in m^2.

    """
    a = 6378137  # meters
    b = 6356752.3142  # meters
    e = math.sqrt(1 - (b/a)**2)
    area_list = []
    for f in [center_lat+pixel_size/2, center_lat-pixel_size/2]:
        zm = 1 - e*math.sin(math.radians(f))
        zp = 1 + e*math.sin(math.radians(f))
        area_list.append(
            math.pi * b**2 * (
                math.log(zp/zm) / (2*e) +
                math.sin(math.radians(f)) / (zp*zm)))
    return abs(pixel_size / 360. * (area_list[0] - area_list[1]))


def mask_op(mask_array, value_array):
    """Mask out value to 0 if mask array is not 1."""
    result = numpy.copy(value_array)
    result[mask_array > 0] = 0.0
    return result


def calculate_mask_area_km2(base_mask_raster_path):
    """Calculate area of mask==1."""
    base_raster_info = geoprocessing.get_raster_info(
        base_mask_raster_path)

    base_srs = osr.SpatialReference()
    base_srs.ImportFromWkt(base_raster_info['projection_wkt'])
    if base_srs.IsProjected():
        # convert m^2 of pixel size to km2
        pixel_conversion = numpy.array([[
            abs(base_raster_info['pixel_size'][0] *
                base_raster_info['pixel_size'][1])]]) / 1e6
    else:
        # create 1D array of pixel size vs. lat
        n_rows = base_raster_info['raster_size'][1]
        pixel_height = abs(base_raster_info['geotransform'][5])
        # the / 2 is to get in the center of the pixel
        miny = base_raster_info['bounding_box'][1] + pixel_height/2
        maxy = base_raster_info['bounding_box'][3] - pixel_height/2
        lat_vals = numpy.linspace(maxy, miny, n_rows)

        pixel_conversion = 1.0 / 1e6 * numpy.array([
            [area_of_pixel(pixel_height, lat_val)] for lat_val in lat_vals])

    nodata = base_raster_info['nodata'][0]
    area_raster_path = 'tmp_area_mask.tif'
    geoprocessing.raster_calculator(
        [(base_mask_raster_path, 1), pixel_conversion], mask_op,
        area_raster_path, gdal.GDT_Float32, nodata)

    area_sum = 0.0
    for _, area_block in geoprocessing.iterblocks((area_raster_path, 1)):
        area_sum += numpy.sum(area_block)
    return area_sum


def _make_logger_callback(message):
    """Build a timed logger callback that prints ``message`` replaced.

    Args:
        message (string): a string that expects 2 placement %% variables,
            first for % complete from ``df_complete``, second from
            ``p_progress_arg[0]``.

    Return:
        Function with signature:
            logger_callback(df_complete, psz_message, p_progress_arg)

    """
    def logger_callback(df_complete, _, p_progress_arg):
        """Argument names come from the GDAL API for callbacks."""
        try:
            current_time = time.time()
            if ((current_time - logger_callback.last_time) > 5.0 or
                    (df_complete == 1.0 and
                     logger_callback.total_time >= 5.0)):
                # In some multiprocess applications I was encountering a
                # ``p_progress_arg`` of None. This is unexpected and I suspect
                # was an issue for some kind of GDAL race condition. So I'm
                # guarding against it here and reporting an appropriate log
                # if it occurs.
                if p_progress_arg:
                    LOGGER.info(message, df_complete * 100, p_progress_arg[0])
                else:
                    LOGGER.info(message, df_complete * 100, '')
                logger_callback.last_time = current_time
                logger_callback.total_time += current_time
        except AttributeError:
            logger_callback.last_time = time.time()
            logger_callback.total_time = 0.0
        except Exception:
            LOGGER.exception("Unhandled error occurred while logging "
                             "progress.  df_complete: %s, p_progress_arg: %s",
                             df_complete, p_progress_arg)

    return logger_callback


def basefilename(path):
    return os.path.basename(os.path.splitext(path)[0])


def rasterize(vector_path, fid, base_raster_path, target_raster_path):
    geoprocessing.new_raster_from_base(
        base_raster_path, target_raster_path, gdal.GDT_Float32, [0])
    geoprocessing.rasterize(
        vector_path, target_raster_path,
        burn_values=[1], where_clause=f"FID = '{fid}'")


def mask_raster(base_raster_path, mask_raster_path, target_raster_path):
    def _mask_raster(array, mask_array):
        result = numpy.zeros(array.shape)
        valid_mask = mask_array > 0
        result[valid_mask] = array[valid_mask]
        return result

    geoprocessing.single_thread_raster_calculator(
        [(base_raster_path, 1), (mask_raster_path, 1)], _mask_raster,
        target_raster_path, gdal.GDT_Float32, None,
        raster_driver_creation_tuple=('GTIFF', (
            'TILED=YES', 'BIGTIFF=YES', 'COMPRESS=LZW',
            'BLOCKXSIZE=256', 'BLOCKYSIZE=256', 'SPARSE_OK=TRUE')))


def calculate_sum_over_mask(base_raster_path, mask_raster_path):
    running_sum = 0
    mask_raster = gdal.OpenEx(
        mask_raster_path, gdal.OF_RASTER | gdal.GA_ReadOnly)
    mask_band = mask_raster.GetRasterBand(1)
    for offset_dict, base_array in geoprocessing.iterblocks(
            (base_raster_path, 1), skip_sparse=True):
        mask_array = mask_band.ReadAsArray(**offset_dict)
        running_sum += numpy.sum(
            base_array[(mask_array > 0) & (base_array > 0)])
    return running_sum


# def calculate_mask_area_km2(base_raster_path, target_epsg):
#     source_raster = gdal.OpenEx(base_raster_path, gdal.OF_RASTER)
#     target_srs = osr.SpatialReference()
#     target_srs.ImportFromEPSG(target_epsg)

#     LOGGER.debug(f'about to warp {base_raster_path} to epsg:{target_epsg}')
#     reprojected_raster = gdal.Warp(
#         '',
#         source_raster,
#         format='MEM',
#         dstSRS=target_srs)
#     reprojected_band = reprojected_raster.GetRasterBand(1)
#     nodata = reprojected_band.GetNoDataValue()
#     LOGGER.debug(f'nodata value is {nodata}')
#     LOGGER.debug(f'read warped {base_raster_path}')
#     reprojected_data = reprojected_band.ReadAsArray()
#     reprojected_geotransform = reprojected_raster.GetGeoTransform()
#     reprojected_pixel_width, reprojected_pixel_height = (
#         reprojected_geotransform[1], abs(reprojected_geotransform[5]))

#     # Calculate the area of pixels with values > 1
#     pixel_area = reprojected_pixel_width * reprojected_pixel_height
#     LOGGER.debug('sum it up')
#     count = ((reprojected_data > 0) & (reprojected_data != nodata)).sum()
#     total_area = count * pixel_area / 1e6  # covert to km2

#     LOGGER.debug(f'total area of {base_raster_path}: {total_area}km2')
#     return total_area


def clip_and_calculate_length_in_km(
        poly_vector_path, line_vector_path, fid_value, epsg_projection):
    poly_vector = gdal.OpenEx(poly_vector_path, gdal.OF_VECTOR)
    poly_layer = poly_vector.GetLayer()
    poly_layer.SetAttributeFilter(f"FID = '{fid_value}'")

    line_vector = gdal.OpenEx(line_vector_path, gdal.OF_VECTOR)
    line_layer = line_vector.GetLayer()

    clipped_lines_mem = ogr.GetDriverByName('Memory').CreateDataSource('temp')
    clipped_lines_layer = clipped_lines_mem.CreateLayer('clipped_roads')

    target_projection = osr.SpatialReference()
    target_projection.ImportFromEPSG(epsg_projection)
    target_projection.SetAxisMappingStrategy(DEFAULT_OSR_AXIS_MAPPING_STRATEGY)

    line_layer.Clip(poly_layer, clipped_lines_layer)

    transform = osr.CreateCoordinateTransformation(
        line_layer.GetSpatialRef(), target_projection)
    total_length = 0
    for line_feature in clipped_lines_layer:
        geometry = line_feature.GetGeometryRef()
        geometry.Transform(transform)
        total_length += geometry.Length() / 1000  # convert to km

    return total_length


def calculate_length_in_km_with_raster(mask_raster_path, line_vector_path, epsg_projection):
    local_time = time.time()
    temp_raster_path = f'%s_{epsg_projection}_mask_{local_time}%s' % os.path.splitext(
        mask_raster_path)
    geoprocessing.raster_calculator(
        [(mask_raster_path, 1)], lambda a: (a > 0).astype(numpy.uint8),
        temp_raster_path,
        gdal.GDT_Byte, None,
        calc_raster_stats=False, skip_sparse=True)

    mask_raster = gdal.OpenEx(temp_raster_path, gdal.OF_RASTER)
    mask_band = mask_raster.GetRasterBand(1)
    mask_projection = osr.SpatialReference(mask_raster.GetProjection())
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

    target_projection = osr.SpatialReference()
    target_projection.ImportFromEPSG(epsg_projection)
    target_projection.SetAxisMappingStrategy(DEFAULT_OSR_AXIS_MAPPING_STRATEGY)
    transform = osr.CreateCoordinateTransformation(
        mask_projection, target_projection)

    clipped_lines_mem = ogr.GetDriverByName('Memory').CreateDataSource('temp')
    clipped_lines_layer = clipped_lines_mem.CreateLayer('clipped_roads')

    line_vector = gdal.OpenEx(line_vector_path, gdal.OF_VECTOR)
    line_layer = line_vector.GetLayer()
    LOGGER.debug(f'clipping {line_vector_path} to polygon')
    start_time = time.time()

    line_layer.Clip(
        raster_layer, clipped_lines_layer, callback=_make_logger_callback(
            "clipping line set %.1f%% complete %s"))
    LOGGER.debug(
        f'done clippping {clipped_lines_layer} to polygon in '
        f'{time.time()-start_time:.2f}s')

    total_length = 0
    for index, line_feature in enumerate(clipped_lines_layer):
        line_geometry = line_feature.GetGeometryRef()
        line_geometry.Transform(transform)
        total_length += line_geometry.Length()

    total_length_km = total_length / 1000

    mask_raster = None
    mask_band = None
    try:
        os.remove(temp_raster_path)
    except:
        pass
    return total_length_km


SCENARIO_LIST = ['restoration', 'conservation']

TOP10_SERVICE_COVERAGE_RASTERS = {
    ('PH', 'restoration'): r"D:\repositories\wwf-sipa\fig_generator_dir\overlap_rasters\overlap_combos_top_10_PH_restoration_each ecosystem service.tif",
    ('IDN', 'restoration'): r"D:\repositories\wwf-sipa\fig_generator_dir\overlap_rasters\overlap_combos_top_10_IDN_restoration_each ecosystem service.tif",
    ('PH', 'conservation'): r"D:\repositories\wwf-sipa\fig_generator_dir\overlap_rasters\overlap_combos_top_10_PH_conservation_each ecosystem service.tif",
    ('IDN', 'conservation'): r"D:\repositories\wwf-sipa\fig_generator_dir\overlap_rasters\overlap_combos_top_10_IDN_conservation_each ecosystem service.tif",
}


def main():
    task_graph = taskgraph.TaskGraph(WORKSPACE_DIR, os.cpu_count(), 10.0)
    delayed_results = {}
    delayed_province_downstream_intersection_area = {}

    scenario_downstream_coverage_percent_map = collections.defaultdict(
        lambda: collections.defaultdict(dict))
    scenario_downstream_population_coverage_map = collections.defaultdict(
        lambda: collections.defaultdict(dict))
    scenario_downstream_road_coverage_map = collections.defaultdict(
        lambda: collections.defaultdict(dict))
    for (country_id,
         dem_path,
         province_vector_path,
         province_name_key,
         epsg_projection,
         unaligned_pop_raster_path,
         road_vector_path) in [
            ('PH',
             PH_DEM_PATH,
             PH_PROViNCE_VECTOR_PATH,
             'ADM1_EN',
             PH_EPSG_PROJECTION,
             PH_POP_RASTER_PATH,
             PH_ROAD_VECTOR_PATH),
            ('IDN',
             IDN_DEM_PATH,
             IDN_PROViNCE_VECTOR_PATH,
             'NAME_1',
             IDN_EPSG_PROJECTION,
             IDN_POP_RASTER_PATH,
             IDN_ROAD_VECTOR_PATH),]:
        flow_dir_path = os.path.join(WORKSPACE_DIR, basefilename(dem_path) + '.tif')
        global_base_raster_info = geoprocessing.get_raster_info(dem_path)
        filled_pits_dem_path = os.path.join(
            DEM_DIR, 'filled_' + os.path.basename(dem_path))
        fill_dem_task = task_graph.add_task(
            func=routing.fill_pits,
            args=((dem_path, 1), filled_pits_dem_path),
            kwargs={
                'working_dir': None,
                'max_pixel_fill_count': 100000},
            target_path_list=[filled_pits_dem_path],
            task_name=f'fill pits to {filled_pits_dem_path}')
        routing_task = task_graph.add_task(
            func=routing.flow_dir_mfd,
            args=((filled_pits_dem_path, 1), flow_dir_path),
            kwargs={
                'working_dir': WORKSPACE_DIR,
                'raster_driver_creation_tuple': ('GTIFF', (
                    'TILED=YES', 'BIGTIFF=YES', 'COMPRESS=LZW',
                    'BLOCKXSIZE=256', 'BLOCKYSIZE=256', 'SPARSE_OK=TRUE'))
            },
            dependent_task_list=[fill_dem_task],
            target_path_list=[flow_dir_path],
            task_name=f'flow dir {flow_dir_path}')

        current_projection_wkt = geoprocessing.get_vector_info(
            province_vector_path)['projection_wkt']
        dem_pixel_size = abs(
            geoprocessing.get_raster_info(dem_path)['pixel_size'][0])
        simplified_vector_path = os.path.join(
            WORKSPACE_DIR, f'simple_{basefilename(province_vector_path)}.gpkg')
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

        pop_raster_path = os.path.join(
            ALIGNED_DIR, os.path.basename(unaligned_pop_raster_path))
        align_pop_layer_task = task_graph.add_task(
            func=geoprocessing.warp_raster,
            args=(
                unaligned_pop_raster_path, global_base_raster_info['pixel_size'],
                pop_raster_path, 'near'),
            kwargs={
                'target_bb': global_base_raster_info['bounding_box'],
                'target_projection_wkt': global_base_raster_info['projection_wkt'],
                'working_dir': WORKSPACE_DIR,
                'raster_driver_creation_tuple': ('GTIFF', (
                    'TILED=YES', 'BIGTIFF=YES', 'COMPRESS=LZW',
                    'BLOCKXSIZE=256', 'BLOCKYSIZE=256', 'SPARSE_OK=TRUE'))
            },
            target_path_list=[pop_raster_path],
            task_name=f'align pop layer {pop_raster_path}')

        province_scenario_masks = collections.defaultdict(
            lambda: collections.defaultdict(dict))

        simplify_vector_task.join()
        vector = gdal.OpenEx(
            simplified_vector_path, gdal.OF_VECTOR | gdal.GA_ReadOnly)
        layer = vector.GetLayer()

        align_service_raster_task_lookup = {}

        province_list = []
        for index, feature in enumerate(layer):
            province_fid = feature.GetFID()
            province_name = feature.GetField(province_name_key).strip().replace(' ', '_')
            province_mask_path = os.path.join(MASK_DIR, f'{province_name}.tif')
            province_list.append(province_name)

            rasterize_province_task = task_graph.add_task(
                func=rasterize,
                args=(
                    simplified_vector_path,
                    province_fid,
                    dem_path,
                    province_mask_path),
                ignore_path_list=[simplified_vector_path],
                dependent_task_list=[simplify_vector_task],
                target_path_list=[province_mask_path],
                task_name=f'masking {province_name}')

            # calculate area of province
            province_area_task = task_graph.add_task(
                func=calculate_mask_area_km2,
                args=(province_mask_path,),
                dependent_task_list=[rasterize_province_task],
                store_result=True,
                task_name=f'calculate area of {province_name}')
            #province_area_task.join()

            # # of people in province
            pop_count_task = task_graph.add_task(
                func=calculate_sum_over_mask,
                args=(province_mask_path, pop_raster_path),
                dependent_task_list=[
                    rasterize_province_task, align_pop_layer_task],
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
                # guard against an already calculates service overlap
                if (country_id, scenario) not in align_service_raster_task_lookup:
                    service_raster_path = os.path.join(
                        ALIGNED_DIR,
                        f'{country_id}_{scenario}_top10_aligned.tif')
                    align_service_raster_task = task_graph.add_task(
                        func=geoprocessing.warp_raster,
                        args=(
                            TOP10_SERVICE_COVERAGE_RASTERS[(country_id, scenario)],
                            global_base_raster_info['pixel_size'],
                            service_raster_path, 'near'),
                        kwargs={
                            'target_bb': global_base_raster_info['bounding_box'],
                            'target_projection_wkt': global_base_raster_info['projection_wkt'],
                            'working_dir': WORKSPACE_DIR,
                            'raster_driver_creation_tuple': ('GTIFF', (
                                'TILED=YES', 'BIGTIFF=YES', 'COMPRESS=LZW',
                                'BLOCKXSIZE=256', 'BLOCKYSIZE=256', 'SPARSE_OK=TRUE'))
                        },
                        target_path_list=[service_raster_path],
                        task_name=f'align service layer {service_raster_path}')
                    align_service_raster_task_lookup[(country_id, scenario)] = (
                        align_service_raster_task, service_raster_path)
                else:
                    align_service_raster_task, service_raster_path = align_service_raster_task_lookup[(country_id, scenario)]

                # mask TOP10_SERVICE to the current province
                masked_service_raster_path = os.path.join(
                    SERVICE_DIR,
                    f'{country_id}_{province_name}_{scenario}_top10.tif')
                mask_service_task = task_graph.add_task(
                    func=mask_raster,
                    args=(
                        service_raster_path,
                        province_mask_path,
                        masked_service_raster_path),
                    dependent_task_list=[
                        rasterize_province_task, align_service_raster_task],
                    target_path_list=[masked_service_raster_path],
                    task_name=f'masking service {province_name} {scenario}')

                # area of the mask of top 10 service in the province
                service_area_task = task_graph.add_task(
                    func=calculate_mask_area_km2,
                    args=(masked_service_raster_path,),
                    dependent_task_list=[mask_service_task],
                    store_result=True,
                    task_name=f'calculate area {masked_service_raster_path}')
                #service_area_task.join()

                # downstream areas of top 10 service
                global_downstream_coverage_raster_path = os.path.join(
                    MASK_DIR, f'{province_name}_{scenario}_top10_service_downstream_global_coverage.tif')
                downstream_coverage_task = task_graph.add_task(
                    func=routing.flow_accumulation_mfd,
                    args=(
                        (flow_dir_path, 1),
                        global_downstream_coverage_raster_path),
                    kwargs={
                        'weight_raster_path_band': (
                            masked_service_raster_path, 1),
                        'raster_driver_creation_tuple': ('GTIFF', (
                            'TILED=YES', 'BIGTIFF=YES', 'COMPRESS=LZW',
                            'BLOCKXSIZE=256', 'BLOCKYSIZE=256',
                            'SPARSE_OK=TRUE'))
                    },
                    dependent_task_list=[
                        rasterize_province_task, routing_task,
                        mask_service_task],
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

                province_scenario_masks[scenario][province_name]['province_mask_path'] = (
                    (local_mask_service_task, province_mask_path))
                local_downstream_service_area_task = task_graph.add_task(
                    func=calculate_mask_area_km2,
                    args=(local_downstream_coverage_raster_path,),
                    dependent_task_list=[local_mask_service_task],
                    store_result=True,
                    task_name=f'calculate area {local_downstream_coverage_raster_path}')
                #local_downstream_service_area_task.join()
                # calculate area of total downstream top 10% areas
                global_downstream_service_area_task = task_graph.add_task(
                    func=calculate_mask_area_km2,
                    args=(global_downstream_coverage_raster_path,),
                    dependent_task_list=[downstream_coverage_task],
                    store_result=True,
                    task_name=f'calculate area {global_downstream_coverage_raster_path}')
                #global_downstream_service_area_task.join()

                province_scenario_masks[scenario][province_name]['global_downstream_coverage_raster_path'] = (
                    (global_downstream_service_area_task, global_downstream_coverage_raster_path))

                # calculate number of people on the ds mask in the province
                local_ds_service_pop_count_task = task_graph.add_task(
                    func=calculate_sum_over_mask,
                    args=(local_downstream_coverage_raster_path, pop_raster_path),
                    dependent_task_list=[
                        local_mask_service_task,
                        align_pop_layer_task],
                    store_result=True,
                    task_name=f'calculate people in {local_downstream_coverage_raster_path}')

                # calculate km of roads on the ds mask in the province
                local_ds_length_of_roads_task = task_graph.add_task(
                    func=calculate_length_in_km_with_raster,
                    args=(
                        local_downstream_coverage_raster_path, road_vector_path,
                        epsg_projection),
                    ignore_path_list=[road_vector_path],
                    dependent_task_list=[local_mask_service_task],
                    store_result=True,
                    task_name=f'road length for {province_fid} {scenario}')

                delayed_results[(country_id, scenario, province_name)] = (
                    province_area_task,
                    pop_count_task,
                    length_of_roads_task,
                    service_area_task,
                    global_downstream_service_area_task,
                    local_downstream_service_area_task,
                    local_ds_service_pop_count_task,
                    local_ds_length_of_roads_task,)

        for scenario in SCENARIO_LIST:
            for base_province, downstream_province in itertools.product(
                    province_list, province_list):
                (base_coverage_task, base_downstream_coverage_path) = province_scenario_masks\
                    [scenario][base_province]['global_downstream_coverage_raster_path']
                (downstream_mask_task, downstream_province_mask_path) = province_scenario_masks\
                    [scenario][downstream_province]['province_mask_path']

                downstream_coverage_of_base_province_raster_path = os.path.join(
                    DOWNSTREAM_COVERAGE_DIR, f'{base_province} on {downstream_province}.tif')

                # intersection of downstream province with downstream coverage
                province_downstream_intersection_task = task_graph.add_task(
                    func=mask_raster,
                    args=(
                        base_downstream_coverage_path,
                        downstream_province_mask_path,
                        downstream_coverage_of_base_province_raster_path),
                    dependent_task_list=[
                        base_coverage_task,
                        downstream_mask_task,
                        rasterize_province_task],
                    target_path_list=[downstream_coverage_of_base_province_raster_path],
                    task_name=f'masking service {province_name} {scenario}')

                province_downstream_intersection_area_task = task_graph.add_task(
                    func=calculate_mask_area_km2,
                    args=(downstream_coverage_of_base_province_raster_path,),
                    dependent_task_list=[province_downstream_intersection_task],
                    store_result=True,
                    task_name=f'calculate area {downstream_coverage_of_base_province_raster_path}')
                #province_downstream_intersection_area_task.join()

                downstream_service_pop_count_task = task_graph.add_task(
                    func=calculate_sum_over_mask,
                    args=(downstream_coverage_of_base_province_raster_path, pop_raster_path),
                    dependent_task_list=[province_downstream_intersection_task],
                    store_result=True,
                    task_name=f'calculate people in {downstream_coverage_of_base_province_raster_path}')

                # calculate km of roads on the ds mask in the province
                downstream_length_of_roads_task = task_graph.add_task(
                    func=calculate_length_in_km_with_raster,
                    args=(
                        downstream_coverage_of_base_province_raster_path, road_vector_path,
                        epsg_projection),
                    ignore_path_list=[road_vector_path],
                    dependent_task_list=[province_downstream_intersection_task],
                    store_result=True,
                    task_name=f'road length for downstream {downstream_coverage_of_base_province_raster_path}')

                delayed_province_downstream_intersection_area[
                    (country_id, scenario, base_province, downstream_province)] = \
                    (province_downstream_intersection_area_task,
                     downstream_service_pop_count_task,
                     downstream_length_of_roads_task)

    analysis_df = collections.defaultdict(lambda: pandas.DataFrame())
    for (country_id, scenario, province_name) in delayed_results:
        (province_area_task,
         pop_count_task,
         length_of_roads_task,
         service_area_task,
         global_downstream_service_area_task,
         local_downstream_service_area_task,
         local_ds_service_pop_count_task,
         local_ds_length_of_roads_task,) = delayed_results[country_id, scenario, province_name]
        row_data = {
            'country': country_id,
            'scenario': scenario,
            'province name': province_name,
            'province_area': province_area_task.get(),
            'pop count': pop_count_task.get(),
            'length of roads km': length_of_roads_task.get(),
            'top 10% service area': service_area_task.get(),
            'top 10% service area downstream': global_downstream_service_area_task.get(),
            'top 10% service area downstream in province': local_downstream_service_area_task.get(),
            'pop count in top 10% local downstream service': local_ds_service_pop_count_task.get(),
            'length of roads in km in top 10% local downstream service': local_ds_length_of_roads_task.get(),
        }

        province_scenario_masks[scenario][province_name] = {
            'province_mask_path': province_mask_path,
            'global_downstream_coverage_raster_path': global_downstream_coverage_raster_path,
            'global_downstream_service_area_km2': global_downstream_service_area_task.get()
        }

        row_df = pandas.DataFrame([row_data])
        analysis_df[(country_id, scenario)] = pandas.concat(
            [analysis_df[scenario], row_df], ignore_index=True)

    try:
        for (country_id, scenario), dataframe in analysis_df.items():
            dataframe.to_csv(
                f'province_analysis_{country_id}_{scenario}.csv',
                index=False, na_rep='')
    except Exception:
        LOGGER.exception(f'********************* {analysis_df}')
        raise

    for country_id, scenario, base_province, downstream_province in delayed_province_downstream_intersection_area:
        (province_downstream_intersection_area_task,
         downstream_service_pop_count_task,
         downstream_length_of_roads_task) = \
            delayed_province_downstream_intersection_area[
                (country_id, scenario, base_province, downstream_province)]

        (_,
         _,
         _,
         _,
         base_downstream_area_task,
         _,
         _,
         _,) = delayed_results[(country_id, scenario, base_province)]

        scenario_downstream_coverage_percent_map[(country_id, scenario)][base_province][downstream_province] = (
            province_downstream_intersection_area_task.get() /
            base_downstream_area_task.get() * 100.0)
        scenario_downstream_population_coverage_map[(country_id, scenario)]\
            [base_province][downstream_province] = \
            downstream_service_pop_count_task.get()
        scenario_downstream_road_coverage_map[(country_id, scenario)]\
            [base_province][downstream_province] = \
            downstream_length_of_roads_task.get()


    for country_id, scenario in scenario_downstream_coverage_percent_map:
        downstream_coverage_percent_map = scenario_downstream_coverage_percent_map[(country_id, scenario)]
        downstream_population_coverage_map = scenario_downstream_population_coverage_map[(country_id, scenario)]
        downstream_road_coverage_map = scenario_downstream_road_coverage_map[(country_id, scenario)]

        downstream_coverage_df = pandas.DataFrame.from_dict(
            downstream_coverage_percent_map, orient='index')
        downstream_coverage_df = downstream_coverage_df.fillna(0)
        downstream_coverage_df = downstream_coverage_df.sort_index(axis=0)
        downstream_coverage_df = downstream_coverage_df.sort_index(axis=1)
        downstream_coverage_df.to_csv(
            f'downstream_province_coverage_{country_id}_{scenario}.csv',
            index_label='source')

        downstream_pop_coverage_df = pandas.DataFrame.from_dict(
            downstream_population_coverage_map, orient='index')
        downstream_pop_coverage_df = downstream_pop_coverage_df.fillna(0)
        downstream_pop_coverage_df = downstream_pop_coverage_df.sort_index(axis=0)
        downstream_pop_coverage_df = downstream_pop_coverage_df.sort_index(axis=1)
        downstream_pop_coverage_df.to_csv(
            f'downstream_population_count_{country_id}_{scenario}.csv',
            index_label='source')

        downstream_road_coverage_df = pandas.DataFrame.from_dict(
            downstream_road_coverage_map, orient='index')
        downstream_road_coverage_df = downstream_road_coverage_df.fillna(0)
        downstream_road_coverage_df = downstream_road_coverage_df.sort_index(axis=0)
        downstream_road_coverage_df = downstream_road_coverage_df.sort_index(axis=1)
        downstream_road_coverage_df.to_csv(
            f'downstream_road_coverage_{country_id}_{scenario}.csv',
            index_label='source')
    task_graph.join()
    task_graph.close()


if __name__ == '__main__':
    main()

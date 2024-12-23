"""What are the upstream/downstream dependencies between provinces?"""
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


SCENARIO_LIST = ['restoration', 'conservation_inf', 'conservation_all']

TOP10_SERVICE_COVERAGE_RASTERS = {
    ('PH', 'restoration'): r"./fig_generator_dir_2024_12_21\overlap_rasters\overlap_combos_top_10_PH_restoration_each ecosystem service.tif",
    ('IDN', 'restoration'): r"./fig_generator_dir_2024_12_21\overlap_rasters\overlap_combos_top_10_IDN_restoration_each ecosystem service.tif",
    ('PH', 'conservation_inf'): r"./fig_generator_dir_2024_12_21\overlap_rasters\overlap_combos_top_10_PH_conservation_inf_each ecosystem service.tif",
    ('IDN', 'conservation_inf'): r"./fig_generator_dir_2024_12_21\overlap_rasters\overlap_combos_top_10_IDN_conservation_inf_each ecosystem service.tif",
    ('PH', 'conservation_all'): r"./fig_generator_dir_2024_12_21\overlap_rasters\overlap_combos_top_10_PH_conservation_all_each ecosystem service.tif",
    ('IDN', 'conservation_all'): r"./fig_generator_dir_2024_12_21\overlap_rasters\overlap_combos_top_10_IDN_conservation_all_each ecosystem service.tif",
}

IDN_PROVINCE_VECTOR_PATH = r"./data\admin_boundaries\IDN_adm1.gpkg"
PH_PROVINCE_VECTOR_PATH = r"./data\admin_boundaries\PH_adm1.gpkg"

IDN_DEM_PATH = r"./data\idn_dem.tif"
PH_DEM_PATH = r"./data\ph_dem.tif"

ELLIPSOID_EPSG = 6933

IDN_POP_RASTER_PATH = r"./data\pop\idn_ppp_2020.tif"
PH_POP_RASTER_PATH = r"./data\pop\phl_ppp_2020.tif"

PH_ROAD_VECTOR_PATH = r"./data\infrastructure_polygons\PH_All_Roads_Merged.gpkg"
IDN_ROAD_VECTOR_PATH = r"./data\infrastructure_polygons\IDN_All_Roads_Merged.gpkg"

WORKSPACE_DIR = 'province_dependence_workspace_2024_12_25'
MASK_DIR = os.path.join(WORKSPACE_DIR, 'province_masks')
SERVICE_DIR = os.path.join(WORKSPACE_DIR, 'masked_services')
ALIGNED_DIR = os.path.join(WORKSPACE_DIR, 'aligned_rasters')
DOWNSTREAM_COVERAGE_DIR = os.path.join(WORKSPACE_DIR, 'downstream_rasters')
DEM_DIR = os.path.join(WORKSPACE_DIR, 'filled_dems')
AREA_DIRS = os.path.join(WORKSPACE_DIR, 'areas')
for dir_path in [
        WORKSPACE_DIR, MASK_DIR, SERVICE_DIR, ALIGNED_DIR,
        DOWNSTREAM_COVERAGE_DIR, DEM_DIR, AREA_DIRS]:
    os.makedirs(dir_path, exist_ok=True)


def gdal_error_handler(err_class, err_num, err_msg):
    LOGGER.error(
        '********** ERROR ***************\n'
        f"Error Number: {err_num}\n"
        f"Error Type: {err_class}\n"
        f"Error Message: {err_msg}\n")


# Register the error handler
gdal.PushErrorHandler(gdal_error_handler)
gdal.UseExceptions()


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
    result[numpy.isclose(mask_array, 0) | (mask_array < 0)] = 0.0
    return result


def calculate_mask_area_km2(base_mask_raster_path):
    """Calculate area of mask==1."""
    base_raster_info = geoprocessing.get_raster_info(
        base_mask_raster_path)

    base_srs = osr.SpatialReference()
    base_srs.ImportFromWkt(base_raster_info['projection_wkt'])
    if base_srs.IsProjected():
        # convert m^2 of pixel size to km2
        raise ValueError('supposed to not be projected')
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
    area_raster_path = os.path.join(
        AREA_DIRS,
        '%s_km2_pixel_area_%s' % os.path.splitext(os.path.basename(
            base_mask_raster_path)))

    geoprocessing.single_thread_raster_calculator(
        [(base_mask_raster_path, 1), pixel_conversion], mask_op,
        area_raster_path, gdal.GDT_Float32, nodata)

    area_sum = 0.0
    for _, area_block in geoprocessing.iterblocks((area_raster_path, 1)):
        area_sum += numpy.sum(area_block[area_block > 0])
        # if base_mask_raster_path == r'province_dependence_workspace\province_masks\National_Capital_Region_restoration_top10_service_downstream_global_coverage.tif':
        #     LOGGER.debug(f'*********** {area_block} {numpy.sum(area_block)}')
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

    raster_info = geoprocessing.get_raster_info(base_raster_path)
    geoprocessing.single_thread_raster_calculator(
        [(base_raster_path, 1), (mask_raster_path, 1)], _mask_raster,
        target_raster_path, raster_info['datatype'],
        raster_info['nodata'][0],
        raster_driver_creation_tuple=('GTIFF', (
            'TILED=YES', 'BIGTIFF=YES', 'COMPRESS=LZW',
            'BLOCKXSIZE=256', 'BLOCKYSIZE=256', 'SPARSE_OK=TRUE')))


def calculate_sum_over_mask(base_raster_path, mask_raster_path):
    running_sum = 0
    total_sum = 0
    mask_raster = gdal.OpenEx(
        mask_raster_path, gdal.OF_RASTER | gdal.GA_ReadOnly)
    mask_band = mask_raster.GetRasterBand(1)
    nodata = mask_band.GetNoDataValue()
    for offset_dict, base_array in geoprocessing.iterblocks(
            (base_raster_path, 1), skip_sparse=True):
        mask_array = mask_band.ReadAsArray(**offset_dict)
        valid_mask = (mask_array > 0) & (base_array > 0)
        if nodata is not None:
            valid_mask &= base_array != nodata
        running_sum += numpy.sum(base_array[valid_mask])
        total_sum += numpy.sum(base_array)
    return running_sum


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
        raster_layer, clipped_lines_layer, callback=_make_logger_callback(
            "clipping line set %.1f%% complete %s"))
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


def calculate_length_of_roads_in_service_areas(
        province_mask_raster_path, service_raster_path, road_vector_path, epsg_projection):
    """
    Calculate the total length of roads within the province that intersect with non-zero pixels
    in the TOP10_SERVICE_COVERAGE_RASTERS.

    Args:
        province_mask_raster_path (str): Path to the province mask raster.
        service_raster_path (str): Path to the TOP10 service coverage raster.
        road_vector_path (str): Path to the roads vector layer.
        epsg_projection (int): EPSG code for the projection to use in length calculations.

    Returns:
        float: Total length of roads in kilometers within the province and service areas.
    """
    import tempfile

    # Create a temporary raster where both the province mask and service raster are non-zero
    combined_raster_path = tempfile.mktemp(suffix='.tif')

    def combined_mask_op(province_array, service_array):
        return numpy.where((province_array > 0) & (service_array > 0), 1, 0).astype(numpy.uint8)

    geoprocessing.raster_calculator(
        [(province_mask_raster_path, 1), (service_raster_path, 1)], combined_mask_op,
        combined_raster_path, gdal.GDT_Byte, None,
        calc_raster_stats=False)

    # Convert the combined raster to polygons
    combined_raster = gdal.OpenEx(combined_raster_path, gdal.OF_RASTER)
    combined_band = combined_raster.GetRasterBand(1)
    mask_projection = osr.SpatialReference(combined_raster.GetProjection())
    mask_projection.SetAxisMappingStrategy(DEFAULT_OSR_AXIS_MAPPING_STRATEGY)
    raster_mem = ogr.GetDriverByName('Memory').CreateDataSource('temp')
    raster_layer = raster_mem.CreateLayer(
        'raster', srs=mask_projection,
        geom_type=ogr.wkbPolygon)
    raster_field = ogr.FieldDefn("value", ogr.OFTInteger)
    raster_layer.CreateField(raster_field)

    # Polygonize the raster
    gdal.Polygonize(
        combined_band, None, raster_layer, 0, [], callback=None)
    raster_layer.SetAttributeFilter("value = 1")

    # Open roads layer
    line_vector = gdal.OpenEx(road_vector_path, gdal.OF_VECTOR)
    line_layer = line_vector.GetLayer()
    line_srs = line_layer.GetSpatialRef()

    # Ensure the roads are in the same projection as the raster layer
    if not mask_projection.IsSame(line_srs):
        coord_transform = osr.CoordinateTransformation(line_srs, mask_projection)
    else:
        coord_transform = None

    # Prepare the output layer for clipped roads
    driver = ogr.GetDriverByName('Memory')
    clipped_lines_mem = driver.CreateDataSource('clipped_roads')
    clipped_lines_layer = clipped_lines_mem.CreateLayer(
        'clipped_roads', srs=mask_projection, geom_type=ogr.wkbLineString)

    # Iterate over road features and clip them using the raster polygons
    line_layer.ResetReading()
    for line_feature in line_layer:
        line_geom = line_feature.GetGeometryRef().Clone()
        if coord_transform:
            line_geom.Transform(coord_transform)
        for raster_feature in raster_layer:
            raster_geom = raster_feature.GetGeometryRef()
            if line_geom.Intersects(raster_geom):
                intersection_geom = line_geom.Intersection(raster_geom)
                if not intersection_geom.IsEmpty():
                    new_feature = ogr.Feature(clipped_lines_layer.GetLayerDefn())
                    new_feature.SetGeometry(intersection_geom)
                    clipped_lines_layer.CreateFeature(new_feature)
                    new_feature = None
        raster_layer.ResetReading()
        line_feature = None

    # Transform the clipped road geometries to the target projection
    target_projection = osr.SpatialReference()
    target_projection.ImportFromEPSG(epsg_projection)
    target_projection.SetAxisMappingStrategy(DEFAULT_OSR_AXIS_MAPPING_STRATEGY)
    transform = osr.CreateCoordinateTransformation(mask_projection, target_projection)

    # Calculate the total length
    total_length = 0
    clipped_lines_layer.ResetReading()
    for line_feature in clipped_lines_layer:
        line_geometry = line_feature.GetGeometryRef()
        line_geometry.Transform(transform)
        total_length += line_geometry.Length()
    total_length_km = total_length / 1000  # Convert to kilometers

    # Clean up temporary files and resources
    combined_raster = None
    combined_band = None
    os.remove(combined_raster_path)

    return total_length_km


def main():
    task_graph = taskgraph.TaskGraph(WORKSPACE_DIR, os.cpu_count(), 10.0, allow_different_target_paths=False)
    delayed_results = {}
    delayed_province_downstream_intersection_area = {}

    scenario_downstream_coverage_km2_map = collections.defaultdict(
        lambda: collections.defaultdict(dict))
    scenario_downstream_population_coverage_map = collections.defaultdict(
        lambda: collections.defaultdict(dict))
    scenario_downstream_road_coverage_map = collections.defaultdict(
        lambda: collections.defaultdict(dict))
    for (country_id,
         dem_path,
         province_vector_path,
         province_name_key,
         unaligned_pop_raster_path,
         road_vector_path) in [
            ('PH',
             PH_DEM_PATH,
             PH_PROVINCE_VECTOR_PATH,
             'ADM1_EN',
             PH_POP_RASTER_PATH,
             PH_ROAD_VECTOR_PATH),
            ('IDN',
             IDN_DEM_PATH,
             IDN_PROVINCE_VECTOR_PATH,
             'NAME_1',
             IDN_POP_RASTER_PATH,
             IDN_ROAD_VECTOR_PATH)]:

        # shortcut to just skip certain countries, update VALID_COUNTRY_ID if you want this
        if VALID_COUNTRY_ID is not None and country_id not in VALID_COUNTRY_ID:
            continue

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

        province_set = set()
        for index, feature in enumerate(layer):
            province_fid = feature.GetFID()
            province_name = feature.GetField(province_name_key).strip().replace(' ', '_')
            if VALID_PROVINCE_NAMES is not None and province_name not in VALID_PROVINCE_NAMES:
                continue

            province_mask_path = os.path.join(MASK_DIR, f'{province_name}.tif')
            province_set.add(province_name)

            rasterize_province_task = task_graph.add_task(
                func=rasterize,
                args=(
                    simplified_vector_path,
                    province_fid,
                    dem_path,
                    province_mask_path),
                copy_duplicate_artifact=True,
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
                args=(pop_raster_path, province_mask_path),
                dependent_task_list=[
                    rasterize_province_task, align_pop_layer_task],
                store_result=True,
                task_name=f'calculate pope in {province_name}')

            # length of roads in km
            length_of_roads_task = task_graph.add_task(
                func=clip_and_calculate_length_in_km,
                args=(
                    province_vector_path, road_vector_path,
                    province_fid, ELLIPSOID_EPSG),
                ignore_path_list=[province_vector_path, road_vector_path],
                store_result=True,
                task_name=f'road length for {country_id} {province_fid}')

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
                # TODO: THIS IS WHERE WE CALCUALTE THE PROVINCE SERVICE AREA
                service_area_task = task_graph.add_task(
                    func=calculate_mask_area_km2,
                    args=(masked_service_raster_path,),
                    dependent_task_list=[mask_service_task],
                    store_result=True,
                    task_name=f'calculate area {masked_service_raster_path}')

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
                    copy_duplicate_artifact=True,
                    target_path_list=[local_downstream_coverage_raster_path],
                    task_name=f'masking downstream coverage to {province_name} {scenario}')

                province_scenario_masks[scenario][province_name]['province_mask_path'] = (
                    (local_mask_service_task, province_mask_path))
                local_downstream_service_area_task = task_graph.add_task(
                    func=calculate_mask_area_km2,
                    args=(local_downstream_coverage_raster_path,),
                    dependent_task_list=[local_mask_service_task],
                    store_result=True,
                    task_name=f'calculate area {local_downstream_coverage_raster_path}')

                # calculate area of total downstream top 10% areas
                global_downstream_service_area_task = task_graph.add_task(
                    func=calculate_mask_area_km2,
                    args=(global_downstream_coverage_raster_path,),
                    dependent_task_list=[downstream_coverage_task],
                    store_result=True,
                    task_name=f'calculate area {global_downstream_coverage_raster_path}')

                province_scenario_masks[scenario][province_name]['global_downstream_coverage_raster_path'] = (
                    (global_downstream_service_area_task, global_downstream_coverage_raster_path))

                # calculate number of people on the ds mask in the province
                local_ds_service_pop_count_task = task_graph.add_task(
                    func=calculate_sum_over_mask,
                    args=(pop_raster_path, local_downstream_coverage_raster_path),
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
                        ELLIPSOID_EPSG),
                    ignore_path_list=[road_vector_path],
                    dependent_task_list=[local_mask_service_task],
                    store_result=True,
                    task_name=f'road length for {country_id} {province_fid} {scenario}')

                delayed_results[(country_id, scenario, province_name)] = (
                    province_area_task,
                    pop_count_task,
                    length_of_roads_task,
                    service_area_task,
                    global_downstream_service_area_task,
                    local_downstream_service_area_task,
                    local_ds_service_pop_count_task,
                    local_ds_length_of_roads_task,)

        duplicate_set = set()
        for scenario in SCENARIO_LIST:
            for base_province, downstream_province in itertools.product(
                    province_set, province_set):
                (base_coverage_task, base_downstream_coverage_path) = province_scenario_masks[
                    scenario][base_province]['global_downstream_coverage_raster_path']
                (downstream_mask_task, downstream_province_mask_path) = province_scenario_masks[
                    scenario][downstream_province]['province_mask_path']

                downstream_coverage_of_base_province_raster_path = os.path.join(
                    DOWNSTREAM_COVERAGE_DIR, f'{base_province}_on_{downstream_province}_{scenario}.tif')

                if downstream_coverage_of_base_province_raster_path in duplicate_set:
                    raise ValueError(f'{downstream_coverage_of_base_province_raster_path} was alreayd in the set!!!!')
                else:
                    duplicate_set.add(downstream_coverage_of_base_province_raster_path)
                # this function takes the "global downstream coverage raster" which is calculated
                # by taking two provences and routing one to the other, but we take that
                # data then mask it only to the downstream provence.
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
                    copy_duplicate_artifact=True,
                    target_path_list=[downstream_coverage_of_base_province_raster_path],
                    task_name=f'masking base downstream coverage to {province_name} {scenario} {downstream_coverage_of_base_province_raster_path}')

                province_downstream_intersection_area_task = task_graph.add_task(
                    func=calculate_mask_area_km2,
                    args=(downstream_coverage_of_base_province_raster_path,),
                    dependent_task_list=[province_downstream_intersection_task],
                    store_result=True,
                    task_name=f'calculate area {downstream_coverage_of_base_province_raster_path}')

                downstream_service_pop_count_task = task_graph.add_task(
                    func=calculate_sum_over_mask,
                    args=(pop_raster_path, downstream_coverage_of_base_province_raster_path),
                    dependent_task_list=[province_downstream_intersection_task],
                    store_result=True,
                    task_name=f'calculate downstream people in {downstream_coverage_of_base_province_raster_path}')

                # calculate km of roads on the ds mask in the province
                downstream_length_of_roads_task = task_graph.add_task(
                    func=calculate_length_in_km_with_raster,
                    args=(
                        downstream_coverage_of_base_province_raster_path, road_vector_path,
                        ELLIPSOID_EPSG),
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
            [analysis_df[(country_id, scenario)], row_df], ignore_index=True)

    for (country_id, scenario), dataframe in analysis_df.items():
        dataframe = dataframe.sort_values(by='province name')
        dataframe.to_csv(os.path.join(
            WORKSPACE_DIR, f'province_analysis_{country_id}_{scenario}.csv'),
            index=False, na_rep='')

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

        scenario_downstream_coverage_km2_map[(country_id, scenario)][base_province][downstream_province] = (
            province_downstream_intersection_area_task.get())

        scenario_downstream_population_coverage_map[(country_id, scenario)]\
            [base_province][downstream_province] = \
            downstream_service_pop_count_task.get()
        scenario_downstream_road_coverage_map[(country_id, scenario)]\
            [base_province][downstream_province] = \
            downstream_length_of_roads_task.get()

    for country_id, scenario in scenario_downstream_coverage_km2_map:
        downstream_coverage_km2_map = scenario_downstream_coverage_km2_map[(country_id, scenario)]
        downstream_population_coverage_map = scenario_downstream_population_coverage_map[(country_id, scenario)]
        downstream_road_coverage_map = scenario_downstream_road_coverage_map[(country_id, scenario)]

        downstream_coverage_df = pandas.DataFrame.from_dict(
            downstream_coverage_km2_map, orient='index')
        downstream_coverage_df = downstream_coverage_df.fillna(0)
        downstream_coverage_df = downstream_coverage_df.sort_index(axis=0)
        downstream_coverage_df = downstream_coverage_df.sort_index(axis=1)
        downstream_coverage_df.to_csv(
            os.path.join(
                WORKSPACE_DIR,
                f'downstream_province_km2_coverage_{country_id}_{scenario}.csv'),
                index_label='source')

        downstream_pop_coverage_df = pandas.DataFrame.from_dict(
            downstream_population_coverage_map, orient='index')
        downstream_pop_coverage_df = downstream_pop_coverage_df.fillna(0)
        downstream_pop_coverage_df = downstream_pop_coverage_df.sort_index(axis=0)
        downstream_pop_coverage_df = downstream_pop_coverage_df.sort_index(axis=1)
        downstream_pop_coverage_df.to_csv(
            os.path.join(
                WORKSPACE_DIR,
                f'downstream_population_count_{country_id}_{scenario}.csv'),
                index_label='source')

        downstream_road_coverage_df = pandas.DataFrame.from_dict(
            downstream_road_coverage_map, orient='index')
        downstream_road_coverage_df = downstream_road_coverage_df.fillna(0)
        downstream_road_coverage_df = downstream_road_coverage_df.sort_index(axis=0)
        downstream_road_coverage_df = downstream_road_coverage_df.sort_index(axis=1)
        downstream_road_coverage_df.to_csv(
            os.path.join(
                WORKSPACE_DIR,
                f'downstream_road_km_coverage_{country_id}_{scenario}.csv'),
                index_label='source')
    task_graph.join()
    task_graph.close()
    LOGGER.info('ALL DONE')


if __name__ == '__main__':
    main()

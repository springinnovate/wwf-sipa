"""
Number of people benefiting:
        take the top 10% maps, delineate the areas downstream of those areas
        (can use our old code for this, shouldn’t have to invent anything new:
        https://github.com/springinnovate/downstream-beneficiaries/blob/main/downstream_mask.py).

        Do the same for areas <2m above sea level within 2km of coasts.
        Merge those two masks together, then overlay with population maps to
        sum the total people.

    Desired output: count of total number of people, and people per island
        group, per municipality or province.

    Inputs to use:
    PH:
        * pop: "D:\repositories\wwf-sipa\data\pop\phl_ppp_2020.tif"
        * dem: "D:\repositories\wwf-sipa\data\ph_dem.tif"

    IDN:
        * pop: "D:\repositories\wwf-sipa\data\pop\idn_ppp_2020.tif"
        * dem: "D:\repositories\wwf-sipa\data\idn_dem.tif"
"""
import os
import logging
import sys
import shutil
import tempfile

from ecoshard import taskgraph
from ecoshard import geoprocessing
from osgeo import gdal
from ecoshard.geoprocessing import routing
import numpy

logging.basicConfig(
    level=logging.DEBUG,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(levelname)s %(name)s'
        ' [%(pathname)s.%(funcName)s:%(lineno)d] %(message)s'),
    stream=sys.stdout)
logging.getLogger('ecoshard.taskgraph').setLevel(logging.INFO)
LOGGER = logging.getLogger(__name__)


RESULTS_DIR = f'workspace_{os.path.basename(os.path.splitext(__file__)[0])}'
WORKING_DIR = os.path.join(RESULTS_DIR, 'working_dir')
for dir_path in [RESULTS_DIR, WORKING_DIR]:
    os.makedirs(dir_path, exist_ok=True)


REGIONS_TO_ANALYZE = ['PH',] #  TODO: 'IDN']

DEM_PATHS = {
    'PH': r"D:\repositories\wwf-sipa\data\ph_dem.tif",
    'IDN': r"D:\repositories\wwf-sipa\data\idn_dem.tif",
}

POP_PATHS = {
    'PH': r"D:\repositories\wwf-sipa\data\pop\phl_ppp_2020.tif",
    'IDN': r"D:\repositories\wwf-sipa\data\pop\idn_ppp_2020.tif",
}

SERVICE_OVERLAP_RASTERS = {
    'PH': [
        "D:/repositories/wwf-sipa/post_processing_results_no_road_recharge/summed_services/10_PH_conservation_inf_service_overlap_count.tif",
        "D:/repositories/wwf-sipa/post_processing_results_no_road_recharge/summed_services/10_PH_restoration_service_overlap_count.tif",],
    'IDN': [
        "D:/repositories/wwf-sipa/post_processing_results_no_road_recharge/summed_services/10_IDN_conservation_inf_service_overlap_count.tif",
        "D:/repositories/wwf-sipa/post_processing_results_no_road_recharge/summed_services/10_IDN_restoration_service_overlap_count.tif",],
}

AOI_REGIONS = {
    'PH': {
        'luzon': (r"D:\repositories\wwf-sipa\data\island_groups\ph_luzon.gpkg", None),
        'mindanao': (r"D:\repositories\wwf-sipa\data\island_groups\ph_mindanao.gpkg", None),
        'municipality': (r"D:\repositories\wwf-sipa\data\admin_boundaries\IDN_adm1.gpkg", 'NAME_1'),
        'total': (None, None),
        'visayas': (r"D:\repositories\wwf-sipa\data\island_groups\ph_visayas.gpkg", None),
    },
    'IDN': {
        'java': (r"D:\repositories\wwf-sipa\data\island_groups\idn_java.gpkg", None),
        'kalimantan': (r"D:\repositories\wwf-sipa\data\island_groups\idn_kalimantan.gpkg", None),
        'maluku_islands': (r"D:\repositories\wwf-sipa\data\island_groups\idn_maluku_islands.gpkg", None),
        'nusa_tenggara': (r"D:\repositories\wwf-sipa\data\island_groups\idn_nusa_tenggara.gpkg", None),
        'paupa': (r"D:\repositories\wwf-sipa\data\island_groups\idn_paupa.gpkg", None, ),
        'provence': (r"D:\repositories\wwf-sipa\data\admin_boundaries\IDN_adm1.gpkg", 'NAME_1'),
        'sulawesi': (r"D:\repositories\wwf-sipa\data\island_groups\idn_sulawesi.gpkg", None),
        'sumatra': (r"D:\repositories\wwf-sipa\data\island_groups\idn_sumatra.gpkg", None),
        'total': (None, None),
    },
}


def route_dem(
        dem_path, flow_dir_path, outlet_raster_path):
    """Turn DEM into flow direction raster."""
    basename = os.path.basename(os.path.splitext(flow_dir_path)[0])
    temp_dir = tempfile.mkdtemp(
        dir=os.path.dirname(flow_dir_path), prefix=f'route_dem_{basename}')
    filled_dem_raster_path = os.path.join(temp_dir, f'filled_{basename}.tif')

    routing.fill_pits(
        (dem_path, 1), filled_dem_raster_path,
        working_dir=temp_dir,
        max_pixel_fill_count=10000)

    flow_dir_path = os.path.join(temp_dir, f'flow_dir_d8_{basename}.tif')
    routing.flow_dir_d8(
        (filled_dem_raster_path, 1), flow_dir_path, working_dir=temp_dir)

    outlet_vector_path = os.path.join(temp_dir, f'outlets_{basename}.gpkg')
    routing.detect_outlets((flow_dir_path, 1), 'd8', outlet_vector_path)

    geoprocessing.new_raster_from_base(
        dem_path, outlet_raster_path, gdal.GDT_Byte, [0])

    vector = gdal.OpenEx(outlet_vector_path, gdal.OF_VECTOR)
    layer = vector.GetLayer()
    raster = gdal.OpenEx(outlet_raster_path, gdal.OF_RASTER | gdal.GA_Update)
    band = raster.GetRasterBand(1)
    gt = raster.GetGeoTransform()
    inv_gt = gdal.InvGeoTransform(gt)
    for feature in layer:
        geom = feature.GetGeometryRef()
        x_coord, y_coord = gdal.ApplyGeoTransform(
            inv_gt, geom.GetX(), geom.GetY())
        band.WriteArray(numpy.array([[1]]), xoff=x_coord, yoff=y_coord)
    band = None
    raster = None
    vector = None
    layer = None
    # shutil.rmtree(temp_dir)


def lowlying_area_mask(dem_path, lowlying_area_raster_path):
    """Calculate low-lying coastal areas <2m w/in 2km of coast."""
    basename = os.path.basename(os.path.splitext(lowlying_area_raster_path)[0])
    temp_dir = tempfile.mkdtemp(
        dir=os.path.dirname(lowlying_area_raster_path),
        prefix=f'lowlying_area_{basename}')
    nodata_mask_raster_path = os.path.join(temp_dir, 'nodata_mask.tif')
    nodata = geoprocessing.get_raster_info(dem_path)['nodata'][0]

    def _nodata_mask_op(array):
        result = (array == nodata).astype(int)
        return result
    geoprocessing.raster_calculator(
        [(dem_path, 1)], _nodata_mask_op, nodata_mask_raster_path,
        gdal.GDT_Byte, None, allow_different_blocksize=True)

    distance_raster_path = os.path.join(temp_dir, 'distance.tif')
    geoprocessing.distance_transform_edt(
        (nodata_mask_raster_path, 1), distance_raster_path,
        sampling_distance=(90., 90.),
        working_dir=temp_dir, clean_working_dir=True)

    def _lowlying_op(dem_array, dist_array):
        # filter by DEM <= 2m and within 2km of shore
        result = (
            (dem_array != nodata) & (dem_array <= 2) & (dist_array <= 2000))
        return result.astype(int)
    geoprocessing.raster_calculator(
        [(dem_path, 1), (distance_raster_path, 1)], _lowlying_op,
        lowlying_area_raster_path, gdal.GDT_Byte, None,
        allow_different_blocksize=True)
    # shutil.rmtree(temp_dir)


def calc_sum_by_mask(base_raster_path, vector_path, field_val):
    """Return sum of base in vector w/ field."""
    if vector_path is None:
        # sum by total
        running_sum = 0
        for _, array in geoprocessing.iterblocks(base_raster_path):
            running_sum = numpy.sum(array)
        return running_sum

    zonal_stats = geoprocessing.zonal_statistics(
        (base_raster_path, 1), vector_path,
        ignore_nodata=True, polygons_might_overlap=False,
        clean_working_dir=True)

    if field_val is None:
        running_sum = 0
        for feature_id, value_dict in zonal_stats.items():
            running_sum += value_dict['sum']
        return running_sum

    vector = gdal.OpenEx(vector_path, gdal.OF_VECTOR)
    layer = vector.GetLayer()

    result_by_fieldname = {
        layer.GetFid(fid).GetField(field_val): value_dict['sum']
        for fid, value_dict in zonal_stats.items()
    }
    vector = None
    layer = None
    return result_by_fieldname


def merge_and_mask_raster(
        mask_a_path, mask_b_path, raster_to_mask, target_mask_path):
    """Merge a and b and resize so it fits reference."""
    basename = os.path.basename(os.path.splitext(target_mask_path)[0])
    temp_dir = tempfile.mkdtemp(
        dir=os.path.dirname(target_mask_path), prefix=f'merge_and_mask_{basename}')

    merged_raster_path = os.path.join(temp_dir, f'merged_{basename}.tif')

    def _merge_op(array_a, array_b):
        return ((array_a >= 1) | (array_b >= 1)).astype(int)
    geoprocessing.raster_calculator(
        [(mask_a_path, 1), (mask_b_path, 1)], _merge_op,
        merged_raster_path, gdal.GDT_Byte, None)

    reference_raster_info = geoprocessing.get_raster_info(raster_to_mask)
    warped_merged_raster_path = os.path.join(
        temp_dir, f'warped_merged_{basename}.tif')
    geoprocessing.warp_raster(
        merged_raster_path, reference_raster_info['pixel_size'],
        warped_merged_raster_path, 'near',
        target_bb=reference_raster_info['bounding_box'],
        target_projection_wkt=reference_raster_info['projection_wkt'],
        working_dir=temp_dir,
        output_type=gdal.GDT_Byte)

    def _mask_op(mask_array, base_array):
        result = base_array.copy()
        result[mask_array < 0] = 0

    geoprocessing.raster_calculator(
        [(warped_merged_raster_path, 1), (raster_to_mask, 1)], _mask_op,
        target_mask_path, reference_raster_info['datatype'], None)
    # shutil.rmtree(temp_dir)


def main():
    """Entry point."""
    task_graph = taskgraph.TaskGraph(WORKING_DIR, os.cpu_count(), 15.0)
    sum_results = {}
    for region_id in REGIONS_TO_ANALYZE:
        basename = os.path.basename(os.path.splitext(DEM_PATHS[region_id])[0])
        flow_dir_path = os.path.join(WORKING_DIR, f'flow_dir_{basename}.tif')
        outlet_raster_path = os.path.join(
            WORKING_DIR, f'outlet_{basename}.tif')
        route_task = task_graph.add_task(
            func=route_dem,
            args=(DEM_PATHS[region_id], flow_dir_path, outlet_raster_path),
            target_path_list=[flow_dir_path, outlet_raster_path],
            task_name=f'route {basename}')

        # TODO: delinate areas <2m within 2km of the coast
        lowlying_area_raster_path = os.path.join(
            WORKING_DIR, f'lowlying_mask_{basename}.tif')
        lowlying_task = task_graph.add_task(
            func=lowlying_area_mask,
            args=(DEM_PATHS[region_id], lowlying_area_raster_path),
            target_path_list=[lowlying_area_raster_path],
            task_name=f'calc lowlying areas {basename}')

        for service_overlap_raster_path in SERVICE_OVERLAP_RASTERS[region_id]:
            # TODO: warp service overlap to fit DEM
            # delineate the areas downstream of 10% mask
            service_basename = os.path.basename(
                os.path.splitext(service_overlap_raster_path)[0])
            downstream_mask_raster_path = os.path.join(
                WORKING_DIR, f'{service_basename}_downstream_mask.tif')
            downstream_mask_task = task_graph.add_task(
                func=routing.distance_to_channel_d8,
                args=((flow_dir_path, 1), (outlet_raster_path, 1),
                      downstream_mask_raster_path),
                kwargs={
                    'weight_raster_path_band': (service_overlap_raster_path, 1)
                    },
                target_path_list=[downstream_mask_raster_path],
                dependent_task_list=[route_task],
                task_name=f'downstream mask for {service_basename}')
            pop_basename = os.path.basename(
                os.path.splitext(POP_PATHS[region_id])[0])
            masked_population_path = os.path.join(
                WORKING_DIR, f'masked_{pop_basename}_by_{service_basename}.tif')
            # warp masks so it fits the population raster
            merge_and_mask_task = task_graph.add_task(
                func=merge_and_mask_raster,
                args=(
                    lowlying_area_raster_path, downstream_mask_raster_path,
                    POP_PATHS[region_id], masked_population_path),
                dependent_task_list=[downstream_mask_task, lowlying_task],
                target_path_list=[masked_population_path],
                task_name=f'merge and resize masks for {service_basename}')

            local_region_sum_results = {}
            for local_region_id, (vector_path, field_name) in \
                    AOI_REGIONS[region_id].items():
                sum_by_mask_task = task_graph.add_task(
                    func=calc_sum_by_mask,
                    args=(masked_population_path, vector_path, field_name),
                    store_result=True,
                    dependent_task_list=[merge_and_mask_task],
                    task_name=f'sum up {local_region_id} in {region_id}')
                local_region_sum_results[local_region_id] = sum_by_mask_task
            sum_results[region_id] = local_region_sum_results

    task_graph.join()
    with open(RESULTS_DIR, 'w') as results_table:
        results_table.write('region,local,sub local,pop sum\n')
        for region_id in REGIONS_TO_ANALYZE:
            for local_region_id, value_task in sum_results[region_id].items():
                value = value_task.get()
                if isinstance(value, dict):
                    for sub_region_id, local_sum in value.items():
                        results_table.write(
                            f'{region_id},{local_region_id},{sub_region_id},'
                            f'{local_sum}\n')
                else:
                    results_table.write(
                        f'{region_id},{local_region_id},{value}\n')
    task_graph.close()


if __name__ == '__main__':
    main()

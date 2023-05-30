"""Calculate downstream benficiary coverage."""
import argparse
import hashlib
import configparser
import os
import logging
import shutil
import sys
import tempfile

from osgeo import gdal
from osgeo import ogr
from ecoshard.geoprocessing import routing
from ecoshard import geoprocessing
from ecoshard import taskgraph
import numpy

GLOBAL_WORKSPACE_DIR = 'downstream_beneficiary_workspace'
GLOBAL_DEM_PROCESSOR = {}

logging.basicConfig(
    level=logging.DEBUG,
    stream=sys.stdout,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(levelname)s %(name)s'
        ' [%(funcName)s:%(lineno)d] %(message)s'))
LOGGER = logging.getLogger(os.path.splitext(os.path.basename(__file__))[0])
LOGGER.setLevel(logging.DEBUG)
logging.getLogger('ecoshard.fetch_data').setLevel(logging.INFO)


def load_ini_file(ini_path):
    config = configparser.ConfigParser(allow_no_value=True)
    config.read(ini_path)

    everything_ok = True
    for section in config.sections():
        local_config = config[section]
        for path_key in [
                'dem_path', 'aoi_path', 'beneficiary_path',
                'upstream_mask_raster_path', 'subset_vector_path']:
            if path_key not in local_config:
                everything_ok = False
                LOGGER.error(f'expected {path_key} in section {section}')
                continue
            path = local_config[path_key]
            if path == '' and path_key not in [
                    'dem_path', 'aoi_path', 'benficiary_path']:
                continue
            if path_key == 'beneficiary_path':
                path_list = path.split(',')
            else:
                path_list = [path]
            for path in path_list:
                if not os.path.exists(path):
                    everything_ok = False
                    LOGGER.error(
                        f'expected {section}-{path_key} to have a file at {path} '
                        'but it is not found')
        if 'calculate_per_pixel_beneficiary_raster' not in local_config:
            everything_ok = False
            LOGGER.error(f'expected CALCULATE_PER_PIXEL_BENEFICIARY_RASTER in {section} but not found')
        try:
            _ = float(local_config['pixel_size'])
        except ValueError:
            everything_ok = False
            LOGGER.error(f"expected {local_config['pixel_size']} to be a float but it is not")
        except KeyError:
            everything_ok = False
            LOGGER.error(f'expected PIXEL_SIZE in {section} but not found')

    if not everything_ok:
        raise ValueError(f'error(s) on {ini_path}')
    return config


def copy_vector_to_downstream_value(
        input_vector_path, fid_to_value_map, output_vector_path):
    input_vector = ogr.Open(input_vector_path)
    input_layer = input_vector.GetLayer()

    # Create a new output vector
    output_driver = ogr.GetDriverByName('GPKG')
    output_vector = output_driver.CreateDataSource(output_vector_path)
    output_layer = output_vector.CreateLayer(
        os.path.splitext(os.path.basename(output_vector_path))[0],
        geom_type=input_layer.GetGeomType())

    # Create a new field called "downstream_value"
    downstream_value_field = ogr.FieldDefn("downstream_value", ogr.OFTInteger)
    output_layer.CreateField(downstream_value_field)

    # Copy feature geometry and set "downstream_value" field
    output_layer.StartTransaction()
    for feature in input_layer:
        output_feature = ogr.Feature(output_layer.GetLayerDefn())
        output_feature.SetGeometry(feature.GetGeometryRef())
        output_feature.SetField(
            "downstream_value", fid_to_value_map[str(feature.GetFID())].get())
        output_layer.CreateFeature(output_feature)
    output_layer.CommitTransaction()

    input_layer = None
    output_layer = None
    input_vector = None
    output_vector = None


def return_value(val):
    """Returns `val`."""
    return val


def get_fid_list(downstream_value_sum_raster_path):
    vector = gdal.OpenEx(downstream_value_sum_raster_path, gdal.OF_VECTOR)
    layer = vector.GetLayer()
    fid_list = [feature.GetFID() for feature in layer]
    return fid_list


def sum_by_coverage(value_raster_path, mask_raster_path):
    running_sum = 0
    value_nodata = geoprocessing.get_raster_info(
        value_raster_path)['nodata'][0]
    for _, (value_array, mask_array) in geoprocessing.iterblocks(
            [(value_raster_path, 1), (mask_raster_path, 1)], skip_sparse=True):
        valid_mask = mask_array > 0
        if value_nodata is not None:
            valid_mask &= value_array != value_nodata
        running_sum += numpy.sum(value_array[valid_mask])
    return running_sum


def logical_and_masks(raster_path_list, target_raster_path):
    nodata_list = [
        geoprocessing.get_raster_info(path)['nodata'][0]
        for path in raster_path_list]
    nodata_target = -1

    LOGGER.debug(f'in (logical_and_masks): {raster_path_list}, {target_raster_path}')
    for path in raster_path_list:
        LOGGER.debug(f'{path} info: {geoprocessing.get_raster_info(path)}')

    def _logical_and(*array_list):
        n_arrays = len(array_list)
        overlap_count = numpy.zeros(array_list[0].shape, dtype=int)
        nodata_count = numpy.zeros(overlap_count.shape, dtype=int)
        for nodata, array in zip(nodata_list, array_list):
            if nodata is not None:
                valid_mask = (array != nodata)
                nodata_count += ~valid_mask
            else:
                valid_mask = numpy.ones(overlap_count.shape, dtype=bool)
                nodata_count += 1
            overlap_count += (valid_mask & (array > 0)).astype(int)
        # only nodata where they were all nodata
        result = (overlap_count == n_arrays).astype(int)
        result[nodata_count == n_arrays] = nodata_target
        return result

    geoprocessing.raster_calculator(
        [(path, 1) for path in raster_path_list], _logical_and,
        target_raster_path, gdal.GDT_Int32, nodata_target)


def rasterize_from_base_raster(
        task_graph, base_raster_path, base_vector_path, rasterize_kwargs,
        target_raster_path, dependent_task_list=[],
        additional_mask_raster_path=None):

    if additional_mask_raster_path is None:
        rasterized_raster_path = target_raster_path
    else:
        rasterized_raster_path = os.path.join(
            os.path.dirname(target_raster_path),
            f'pre_masked_{os.path.basename(target_raster_path)}')

    last_task = task_graph.add_task(
        func=geoprocessing.new_raster_from_base,
        args=(
            base_raster_path,
            rasterized_raster_path,
            gdal.GDT_Byte, [0]),
        target_path_list=[rasterized_raster_path],
        dependent_task_list=dependent_task_list,
        task_name=(
            f'create a new raster rasterizing {rasterized_raster_path}'))

    last_task = task_graph.add_task(
        func=geoprocessing.rasterize,
        args=(base_vector_path, rasterized_raster_path),
        kwargs=rasterize_kwargs,
        dependent_task_list=[last_task]+dependent_task_list,
        target_path_list=[rasterized_raster_path],
        task_name=f'rasterize {base_vector_path} to {rasterized_raster_path}')

    if additional_mask_raster_path:
        LOGGER.debug(
            f'********* logical ANDing {rasterized_raster_path} and '
            f'{additional_mask_raster_path}')
        last_task = task_graph.add_task(
            func=logical_and_masks,
            args=(
                [rasterized_raster_path, additional_mask_raster_path],
                target_raster_path),
            target_path_list=[target_raster_path],
            dependent_task_list=[last_task]+dependent_task_list,
            task_name=f'logical and between {rasterized_raster_path}, {additional_mask_raster_path}'
            )

    return last_task


def warp_and_rescale(
        base_raster_path, target_pixel_size, target_bb, target_projection_wkt,
        target_raster_path):
    """Warp a raster so units are consistent with a different pixel size."""
    working_dir = tempfile.mkdtemp(dir=os.path.dirname(target_raster_path))
    warped_raster_path = os.path.join(working_dir, 'warped.tif')

    geoprocessing.warp_raster(
        base_raster_path,
        target_pixel_size,
        warped_raster_path,
        'bilinear',
        target_bb=target_bb,
        target_projection_wkt=target_projection_wkt)
    warped_raster_info = geoprocessing.get_raster_info(warped_raster_path)

    test_base_, base_pixel_area = \
        geoprocessing.get_pixel_area_in_target_projection(
            base_raster_path, warped_raster_info['projection_wkt'])

    test_val, target_pixel_area = \
        geoprocessing.get_pixel_area_in_target_projection(
            warped_raster_path, warped_raster_info['projection_wkt'])

    scale_factor = target_pixel_area / base_pixel_area
    target_nodata = warped_raster_info['nodata'][0]

    if scale_factor != 1:
        def _scale_by_factor(array):
            result = array.copy().astype(float)
            if target_nodata is not None:
                nodata_mask = array != target_nodata
                result[nodata_mask] = array[nodata_mask] * scale_factor
            else:
                result *= scale_factor
            return result

        geoprocessing.raster_calculator(
            [(warped_raster_path, 1)], _scale_by_factor, target_raster_path,
            gdal.GDT_Float32, target_nodata)
    else:
        shutil.copyfile(warped_raster_path, target_raster_path)

    shutil.rmtree(working_dir)


def _sum_all_op(raster_path_list, target_raster):

    nodata_list = [
        geoprocessing.get_raster_info(path)['nodata'][0]
        for path in raster_path_list]
    local_nodata = -1

    def _sum_op(*array_list):
        result = numpy.zeros(array_list[0].shape)
        total_valid_mask = numpy.zeros(result.shape, dtype=bool)
        for array, nodata in zip(array_list, nodata_list):
            if nodata is not None:
                valid_mask = array != nodata
            else:
                valid_mask = numpy.ones(array.shape, dtype=bool)
            result[valid_mask] += array[valid_mask]
            total_valid_mask |= valid_mask
        result[~total_valid_mask] = local_nodata
        return result

    geoprocessing.raster_calculator(
        [(path, 1) for path in raster_path_list], _sum_op,
        target_raster, gdal.GDT_Float32, local_nodata)


def process_dem(
        task_graph, base_dem_path, aoi_path, target_pixel_size,
        workspace_dir):
    """Clip, clean, and route the dem.

    Args:
        task_graph (taskgraph): taskgraph to schedule
        base_dem_path (str): path to DEM raster
        aoi_path (str): path to AOI vector
        target_pixel_size (float): size of target raster in projected units
             of the aoi_path
        workspace_dir (str): directory that is safe to create intermediate
            and final files.

    Returns:
        task that will .get() the flow_direction_raster and outlet raster path
    """
    # clip and align the dem to the aoi_path file
    # pitfill the DEM
    clipped_dem_raster_path = os.path.join(
        workspace_dir, 'clipped_dem.tif')
    if geoprocessing.get_gis_type(aoi_path) == geoprocessing.RASTER_TYPE:
        aoi_info = geoprocessing.get_raster_info(aoi_path)
    else:
        aoi_info = geoprocessing.get_vector_info(aoi_path)

    clip_raster_task = task_graph.add_task(
        func=geoprocessing.warp_raster,
        args=(
            base_dem_path, (target_pixel_size, -target_pixel_size),
            clipped_dem_raster_path, 'bilinear'),
        kwargs={
            'target_bb': aoi_info['bounding_box'],
            'target_projection_wkt': aoi_info['projection_wkt'],
            },
        target_path_list=[clipped_dem_raster_path],
        task_name=f'clip base_dem_path {clipped_dem_raster_path}')

    filled_dem_raster_path = os.path.join(
        workspace_dir, 'filled_dem.tif')
    fill_pits_task = task_graph.add_task(
        func=routing.fill_pits,
        args=(
            (clipped_dem_raster_path, 1), filled_dem_raster_path),
        kwargs={
            'working_dir': workspace_dir,
            'max_pixel_fill_count': -1},
        dependent_task_list=[clip_raster_task],
        target_path_list=[filled_dem_raster_path],
        task_name=f'fill dem pits to {filled_dem_raster_path}')
    # route the DEM
    flow_dir_mfd_raster_path = os.path.join(
        workspace_dir, 'mfd_flow_dir.tif')
    flow_dir_mfd_task = task_graph.add_task(
        func=routing.flow_dir_mfd,
        args=(
            (filled_dem_raster_path, 1), flow_dir_mfd_raster_path),
        kwargs={'working_dir': workspace_dir},
        dependent_task_list=[fill_pits_task],
        target_path_list=[flow_dir_mfd_raster_path],
        task_name=f'calc flow dir for {flow_dir_mfd_raster_path}')

    # calculate the number of downstream value pixels for any pixel on
    # the raster
    outlet_vector_path = os.path.join(workspace_dir, 'outlet_points.gpkg')
    outlet_detection_task = task_graph.add_task(
        func=routing.detect_outlets,
        args=(
            (flow_dir_mfd_raster_path, 1), 'mfd', outlet_vector_path),
        dependent_task_list=[flow_dir_mfd_task],
        target_path_list=[outlet_vector_path],
        ignore_path_list=[outlet_vector_path],
        task_name=f'detect outlets {outlet_vector_path}')

    outlet_raster_path = os.path.join(workspace_dir, 'outlet_raster.tif')
    rasterize_kwargs = {
        'burn_values': [1], 'option_list': ['ALL_TOUCHED=TRUE']}
    rasterized_outlet_task = rasterize_from_base_raster(
        task_graph, flow_dir_mfd_raster_path, outlet_vector_path,
        rasterize_kwargs, outlet_raster_path, dependent_task_list=[
            outlet_detection_task])

    result_filename_task = task_graph.add_task(
        func=return_value,
        args=((flow_dir_mfd_raster_path, outlet_raster_path),),
        dependent_task_list=[flow_dir_mfd_task, rasterized_outlet_task],
        store_result=True,
        task_name=f'return {flow_dir_mfd_raster_path}')

    return result_filename_task


def main():
    """Entrypoint."""
    parser = argparse.ArgumentParser(
        description='Downstream beneficiary analysis.')
    parser.add_argument('ini_file_path', help='Path to INI file')
    parser.add_argument(
        '--n_workers', type=int, default=os.cpu_count(),
        help='number of taskgraph workers')
    parser.add_argument(
        '--debug', action='store_true')
    args = parser.parse_args()

    os.makedirs(GLOBAL_WORKSPACE_DIR, exist_ok=True)
    task_graph = taskgraph.TaskGraph(
        GLOBAL_WORKSPACE_DIR, args.n_workers, 15.0)

    config = load_ini_file(args.ini_file_path)
    if args.debug:
        return

    for section in config.sections():
        process_section(task_graph, config, section)


def get_tuple_hash(t):
    # Convert the tuple to a string representation
    tuple_str = str(t)

    # Create a hash object
    hash_obj = hashlib.md5()

    # Calculate the hash of the tuple string
    hash_obj.update(tuple_str.encode('utf-8'))

    # Get the hexadecimal representation of the hash
    hash_str = hash_obj.hexdigest()

    return hash_str


def process_section(task_graph, config, section):
    local_config = config[section]
    local_workspace_dir = os.path.join(GLOBAL_WORKSPACE_DIR, section)
    os.makedirs(local_workspace_dir, exist_ok=True)

    pixel_size = float(local_config['pixel_size'])

    flow_dir_hash = 'dem_workspace_'+get_tuple_hash((
        local_config['dem_path'],
        local_config['aoi_path'],
        pixel_size))

    dem_workspace_dir = os.path.join(GLOBAL_WORKSPACE_DIR, flow_dir_hash)
    os.makedirs(dem_workspace_dir, exist_ok=True)

    if flow_dir_hash not in GLOBAL_DEM_PROCESSOR:
        flow_dir_task = process_dem(
            task_graph, local_config['dem_path'],
            local_config['aoi_path'],
            pixel_size,
            dem_workspace_dir)
        GLOBAL_DEM_PROCESSOR[flow_dir_hash] = flow_dir_task
    else:
        flow_dir_task = GLOBAL_DEM_PROCESSOR[flow_dir_hash]

    flow_dir_raster_path, outlet_raster_path = flow_dir_task.get()

    aoi_info = geoprocessing.get_vector_info(local_config['aoi_path'])
    beneficiary_raster_list = []
    beneficiary_task_list = []
    for index, vector_or_raster_beneficiary_path in enumerate(
            local_config['beneficiary_path'].split(',')):
        local_beneficiary_raster_path = os.path.join(
            local_workspace_dir,
            f'local_beneficiary_raster_{index}_{section}.tif')
        beneficiary_raster_list.append(local_beneficiary_raster_path)
        if geoprocessing.get_gis_type(vector_or_raster_beneficiary_path) == \
                geoprocessing.VECTOR_TYPE:
            vector_path = vector_or_raster_beneficiary_path
            reprojected_vector_path = os.path.join(
                local_workspace_dir,
                f'reprojected_{os.path.basename(vector_path)}')
            reproject_task = task_graph.add_task(
                func=geoprocessing.reproject_vector,
                args=(
                    vector_path, aoi_info['projection_wkt'],
                    reprojected_vector_path),
                target_path_list=[reprojected_vector_path],
                ignore_path_list=[reprojected_vector_path],
                task_name=f'reproject {reprojected_vector_path}')

            new_raster_task = task_graph.add_task(
                func=geoprocessing.new_raster_from_base,
                args=(
                    flow_dir_raster_path,
                    local_beneficiary_raster_path,
                    gdal.GDT_Byte, [0]),
                target_path_list=[local_beneficiary_raster_path],
                task_name=(
                    f'create a new raster for rasterization '
                    f'{local_beneficiary_raster_path}'))
            beneficiary_raster_task = task_graph.add_task(
                func=geoprocessing.rasterize,
                args=(reprojected_vector_path, local_beneficiary_raster_path),
                kwargs={'burn_values': [1]},
                dependent_task_list=[new_raster_task, reproject_task],
                target_path_list=[local_beneficiary_raster_path],
                task_name=(
                    f'rasterize {reprojected_vector_path} to {local_beneficiary_raster_path}'))
            beneficiary_task_list.append(beneficiary_raster_task)
        else:
            # clip and reproject value raster to aoi's projection
            aoi_info = geoprocessing.get_vector_info(local_config['aoi_path'])
            benficiary_raster_path = vector_or_raster_beneficiary_path
            LOGGER.debug(
                f'**** processing raster {benficiary_raster_path}')
            warp_and_rescale_raster_task = task_graph.add_task(
                func=warp_and_rescale,
                args=(
                    benficiary_raster_path,
                    (pixel_size, -pixel_size),
                    aoi_info['bounding_box'],
                    aoi_info['projection_wkt'],
                    local_beneficiary_raster_path),
                target_path_list=[local_beneficiary_raster_path],
                task_name=f'clip local beneficiary {local_beneficiary_raster_path}')
            beneficiary_task_list.append(warp_and_rescale_raster_task)

    # this is okay and should be untouched except for analysis
    local_benficiaries_per_pixel_raster_path = os.path.join(
        local_workspace_dir,
        f'benficiaries_per_pixel_{section}.tif')
    combine_local_benficiaries_task = task_graph.add_task(
        func=_sum_all_op,
        args=(beneficiary_raster_list,
              local_benficiaries_per_pixel_raster_path),
        target_path_list=[local_benficiaries_per_pixel_raster_path],
        dependent_task_list=beneficiary_task_list,
        task_name=f'sum all to {local_benficiaries_per_pixel_raster_path}')

    # this seems okay and shouldn't be used in other inputs
    calculate_per_pixel_beneficiary_raster = (
        local_config['calculate_per_pixel_beneficiary_raster'].lower() ==
        'true')
    if calculate_per_pixel_beneficiary_raster:
        num_of_downstream_beneficiaries_per_pixel_path = os.path.join(
            GLOBAL_WORKSPACE_DIR,
            f'num_of_downstream_beneficiaries_per_pixel_{section}.tif')
        task_graph.add_task(
            func=routing.distance_to_channel_mfd,
            args=(
                (flow_dir_raster_path, 1), (outlet_raster_path, 1),
                num_of_downstream_beneficiaries_per_pixel_path),
            kwargs={
                'weight_raster_path_band': (
                    local_benficiaries_per_pixel_raster_path, 1)},
            dependent_task_list=[combine_local_benficiaries_task],
            target_path_list=[num_of_downstream_beneficiaries_per_pixel_path],
            task_name='value accumulation')

    # TODO: calculate sums by the upstream raster mask if present
    mask_rasters_to_aggregate_list = []
    mask_raster_id_list = []
    mask_raster_task_list = []
    local_upstream_mask_raster_path = None
    if local_config['upstream_mask_raster_path'] != '':
        flow_dir_raster_info = geoprocessing.get_raster_info(
            flow_dir_raster_path)
        local_upstream_mask_raster_path = os.path.join(
            local_workspace_dir, f'upstream_mask_{section}.tif')
        clip_upstream_mask_task = task_graph.add_task(
            func=geoprocessing.warp_raster,
            args=(
                local_config['upstream_mask_raster_path'],
                flow_dir_raster_info['pixel_size'],
                local_upstream_mask_raster_path, 'bilinear'),
            kwargs={
                'target_bb': aoi_info['bounding_box'],
                'target_projection_wkt': aoi_info['projection_wkt'],
                },
            target_path_list=[local_upstream_mask_raster_path],
            task_name=f'clip upstream raster mask {local_upstream_mask_raster_path}')
        mask_rasters_to_aggregate_list.append(local_upstream_mask_raster_path)
        mask_raster_id_list.append(os.path.basename(os.path.splitext(
            local_upstream_mask_raster_path)[0]))
        mask_raster_task_list.append(clip_upstream_mask_task)

    if local_config['subset_vector_path'] != '':
        reprojected_subset_vector_path = os.path.join(
            local_workspace_dir,
            f'''reprojected_{
                os.path.basename(local_config['subset_vector_path'])}''')
        reproject_task = task_graph.add_task(
            func=geoprocessing.reproject_vector,
            args=(
                local_config['subset_vector_path'], aoi_info['projection_wkt'],
                reprojected_subset_vector_path),
            target_path_list=[reprojected_subset_vector_path],
            ignore_path_list=[reprojected_subset_vector_path],
            task_name=f'reproject {reprojected_subset_vector_path}')
        reproject_task.join()
        fid_list = get_fid_list(reprojected_subset_vector_path)
        result_by_id = {}
        for fid in fid_list:
            fid_mask_path = os.path.join(local_workspace_dir, f'{fid}_mask.tif')
            fid_rasterize_kwargs = {
                'burn_values': [1],
                'option_list': ['ALL_TOUCHED=TRUE'],
                'where_clause': f'FID={fid}'}
            fid_rasterize_task = rasterize_from_base_raster(
                task_graph, flow_dir_raster_path,
                reprojected_subset_vector_path,
                fid_rasterize_kwargs, fid_mask_path,
                dependent_task_list=[clip_upstream_mask_task],
                additional_mask_raster_path=local_upstream_mask_raster_path)

            mask_rasters_to_aggregate_list.append(fid_mask_path)
            mask_raster_id_list.append(fid)
            mask_raster_task_list.append(fid_rasterize_task)

    for mask_id, mask_path, dependent_mask_task in zip(
            mask_raster_id_list,
            mask_rasters_to_aggregate_list,
            mask_raster_task_list):
        downstream_coverage_raster_path = os.path.join(
            local_workspace_dir, f'downstream_coverage_{mask_id}.tif')
        LOGGER.info(f'processing downstream coverage of {downstream_coverage_raster_path}')
        downstream_coverage_task = task_graph.add_task(
            func=routing.distance_to_channel_mfd,
            args=(
                (flow_dir_raster_path, 1), (outlet_raster_path, 1),
                downstream_coverage_raster_path),
            kwargs={
                'weight_raster_path_band': (mask_path, 1)},
            target_path_list=[
                downstream_coverage_raster_path],
            dependent_task_list=[dependent_mask_task],
            task_name='mask downstream smearing')

        sum_by_coverage_task = task_graph.add_task(
            func=sum_by_coverage,
            args=(local_benficiaries_per_pixel_raster_path,
                  downstream_coverage_raster_path),
            store_result=True,
            dependent_task_list=[
                combine_local_benficiaries_task, downstream_coverage_task],
            task_name=f'sum coverage for mask_id {mask_id}')

        result_by_id[str(mask_id)] = sum_by_coverage_task

    beneficiaries_aggregated_by_subset_vector_path = os.path.join(
        GLOBAL_WORKSPACE_DIR,
        f'beneficiaries_aggregated_by_subset_{section}.gpkg')
    task_graph.join()

    if local_config['subset_vector_path'] != '':
        copy_vector_to_downstream_value(
            reprojected_subset_vector_path, result_by_id,
            beneficiaries_aggregated_by_subset_vector_path)

    table_aggregate_path = os.path.join(
        GLOBAL_WORKSPACE_DIR,
        f'benficiaries_aggregated_by_subset_{section}.csv')
    with open(table_aggregate_path, 'w') as table_file:
        table_file.write('mask ID,sum of downstream beneficiaries\n')
        for mask_id in sorted(result_by_id):
            table_file.write(f'{mask_id},{result_by_id[mask_id].get()}\n')


if __name__ == '__main__':
    # p1 = 'downstream_beneficiary_workspace\\ph_downstream_road2019_benes\\tmp3pb7zll3\\pre_masked_12_mask.tif'
    # p2 = 'downstream_beneficiary_workspace\\ph_downstream_road2019_benes\\upstream_mask_ph_downstream_road2019_benes.tif'
    # p1 = r"D:\repositories\wwf-sipa\downstream_beneficiary_workspace\ph_downstream_road2019_benes\benficiaries_per_pixel_ph_downstream_road2019_benes.tif"
    # p2 = r"D:\repositories\wwf-sipa\downstream_beneficiary_workspace\ph_downstream_road2019_benes\tmp3ovr__an\pre_masked_4_mask.tif"
    # p1 = 'downstream_beneficiary_workspace\\ph_downstream_road2019_benes\\tmp32ahx8oa\\pre_masked_1_mask.tif'
    # p2 = 'downstream_beneficiary_workspace\\ph_downstream_road2019_benes\\upstream_mask_ph_downstream_road2019_benes.tif'
    # logical_and_masks([p1, p2], 'road1mask.tif')

    # flow_dir_raster_path = r"D:\repositories\wwf-sipa\downstream_beneficiary_workspace\dem_workspace_b46be7ea77e145786964fff94064e033\mfd_flow_dir.tif"
    # outlet_raster_path = r"D:\repositories\wwf-sipa\downstream_beneficiary_workspace\dem_workspace_b46be7ea77e145786964fff94064e033\outlet_raster.tif"
    # downstream_coverage_raster_path = 'covered.tif'
    # mask_path = r"D:\repositories\wwf-sipa\downstream_beneficiary_workspace\ph_downstream_road2019_benes\1_mask.tif"
    # routing.distance_to_channel_mfd(
    #     (flow_dir_raster_path, 1), (outlet_raster_path, 1),
    #     downstream_coverage_raster_path,
    #     weight_raster_path_band=(mask_path, 1))
    main()

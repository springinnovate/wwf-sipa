"""Land change scenarios.

Model land use change by modeling the distance decay effect from new
infrastructure.

"""
import argparse
import collections
import os
import logging
import sys
import shutil
import tempfile

import numpy
import scipy
import pyproj
from osgeo import osr
from osgeo import ogr
from osgeo import gdal
from shapely.geometry import Point, LineString
from ecoshard import geoprocessing
from ecoshard import taskgraph
from ecoshard.geoprocessing.geoprocessing_core import DEFAULT_GTIFF_CREATION_TUPLE_OPTIONS
from ecoshard.geoprocessing.geoprocessing_core import DEFAULT_OSR_AXIS_MAPPING_STRATEGY
import pandas


RASTER_CREATE_OPTIONS = DEFAULT_GTIFF_CREATION_TUPLE_OPTIONS[1]

logging.basicConfig(
    level=logging.DEBUG,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(levelname)s %(name)s'
        ' [%(pathname)s.%(funcName)s:%(lineno)d] %(message)s'),
    stream=sys.stdout)
LOGGER = logging.getLogger(__name__)
logging.getLogger('ecoshard.geoprocessing').setLevel(logging.DEBUG)
LOGGER.setLevel(logging.DEBUG)

WORKSPACE_DIR = '_workspace_land_change_scenario'

PATH_FIELD = 'path'
INFLUENCE_DIST_FIELD = 'effective impact dist'
MAX_INFLUENCE_DIST_FIELD = 'max impact dist'
CONVERSION_CODE_FIELD = 'conversion code'
ATTRIBUTE_KEY_FIELD = 'attribute key'
ATTRIBUTE_VALUE_FIELD = 'attribute code'
GIS_TYPE_FIELD = 'type'
RASTER_VALUE_FIELD = 'raster value'


def mask_out_value_op(value):
    def _mask_out_value(x):
        result = (x == value)
        LOGGER.debug(f'mask out value {x} {result}')
        return result
    return _mask_out_value


def raw_basename(path): return os.path.basename(os.path.splitext(path)[0])


def square_blocksize(path):
    """Return true if 256x256."""
    raster_info = geoprocessing.get_raster_info(path)
    return tuple(raster_info['block_size']) == (256, 256)


def load_table(table_path):
    """Load infrastructure table and raise errors if needed."""
    table = pandas.read_csv(table_path)
    error_list = []
    for field_name in [
            INFLUENCE_DIST_FIELD, CONVERSION_CODE_FIELD, PATH_FIELD,
            GIS_TYPE_FIELD, RASTER_VALUE_FIELD, MAX_INFLUENCE_DIST_FIELD]:
        if field_name not in table:
            error_list.append(f'Expected field `{field_name}` but not found')
    if (ATTRIBUTE_VALUE_FIELD in table) ^ (ATTRIBUTE_KEY_FIELD in table):
        error_list.append(
            f'If attributes are used, expect both `{ATTRIBUTE_VALUE_FIELD}` '
            f'and `{ATTRIBUTE_KEY_FIELD}` to be defined but only one of them '
            f'was.')

    if error_list:
        raise ValueError(
            '\n\nThe following errors were detected when parsing the table:\n'+
            '\t* '+'\n\t* '.join(error_list) +
            '\n\nFor reference, the following column headings were detected '
            'in the table:\n' +
            '\t* '+'\n\t* '.join(table.columns))
    return table


def convert_meters_to_pixel_units(raster_path, value):
    """Return `value` as a distance in `raster_path` units."""
    raster_info = geoprocessing.get_raster_info(raster_path)
    proj = pyproj.CRS(raster_info['projection_wkt'])
    if proj.is_projected:
        # convert to n pixels
        pixel_units = [abs(value/raster_info['pixel_size'][i]) for i in [0, 1]]
    else:
        # convert to degrees
        centroid_pixel = [
            raster_info['raster_size'][0]//2, raster_info['raster_size'][1]//2]
        # 111111 meters in the y direction is 1 degree (of lat)
        # 111111*cos(lat) meters in the x direction is 1 degree (of long)
        pixel_units = [
            abs(value/(raster_info['pixel_size'][0]*111111*numpy.cos(
                numpy.radians(centroid_pixel[1])))),
            abs(value/(raster_info['pixel_size'][1]*111111))
            ]

        LOGGER.debug(proj)
    return pixel_units


def main():
    """Entry point."""
    parser = argparse.ArgumentParser(description='Model land change')
    parser.add_argument('base_raster_path', help='Path to base raster.')
    parser.add_argument(
        'infrastructure_scenario_path', help='Path to land change pressure '
        f'table. Expected format is to have the columns `{PATH_FIELD}`, '
        f'`{INFLUENCE_DIST_FIELD}`, and `{CONVERSION_CODE_FIELD}` where:\n'
        '\t`path`: path to vector\n'
        f'\t`{INFLUENCE_DIST_FIELD}`: effective distance of infrastructure '
        'impact in meters\n'
        f'\t`{GIS_TYPE_FIELD}`: has the value `raster` or `vector`\n'
        f'\t`{RASTER_VALUE_FIELD}`: used for `raster` types to provide '
        f'additional effect\n'
        f'\t`{CONVERSION_CODE_FIELD}`: if target conversion landscape code\n'
        f'additional parameters include `{ATTRIBUTE_KEY_FIELD}` and '
        f'`{ATTRIBUTE_VALUE_FIELD}` to allow subsets of the vectors where\n'
        f'\t`{ATTRIBUTE_KEY_FIELD}` (optional): field in the vector to filter '
        'by\n'
        f'\t`{ATTRIBUTE_VALUE_FIELD}` (optional): if field is defined, what '
        'value to match')
    parser.add_argument(
        'probability_of_conversion',
        help='Value in 0..1 for when to flip a landcover effect',
        type=float)
    args = parser.parse_args()

    infrastructure_scenario_table = load_table(
        args.infrastructure_scenario_path)
    local_workspace = os.path.join(
        WORKSPACE_DIR, raw_basename(args.infrastructure_scenario_path))
    os.makedirs(local_workspace, exist_ok=True)

    # convert to correct block size

    raster_info = geoprocessing.get_raster_info(args.base_raster_path)
    if not square_blocksize(args.base_raster_path):
        working_base_raster_path = os.path.join(
            local_workspace, os.path.basename(args.base_raster_path))
        geoprocessing.warp_raster(
            args.base_raster_path, raster_info['pixel_size'],
            working_base_raster_path, 'near')
    else:
        working_base_raster_path = args.base_raster_path

    effect_path_code_list = collections.defaultdict(list)
    for index, row in infrastructure_scenario_table.iterrows():
        LOGGER.debug(f'processing {row}')
        if row['type'] == 'vector':
            pixel_units = convert_meters_to_pixel_units(
                working_base_raster_path, row[INFLUENCE_DIST_FIELD])

            tol = raster_info['pixel_size'][0]/2
            where_filter = None
            if ATTRIBUTE_KEY_FIELD in row and not numpy.isnan(
                    row[ATTRIBUTE_KEY_FIELD]):
                where_filter = (
                    f'{row[ATTRIBUTE_KEY_FIELD]}={row[ATTRIBUTE_VALUE_FIELD]}')
            reprojected_vector_path = os.path.join(
                local_workspace,
                f'{raw_basename(row[PATH_FIELD])}_{where_filter}.gpkg')
            vector_info = geoprocessing.get_vector_info(row[PATH_FIELD])
            geoprocessing.reproject_vector(
                row[PATH_FIELD], raster_info['projection_wkt'],
                reprojected_vector_path, layer_id=0,
                driver_name='GPKG', copy_fields=True,
                geometry_type=vector_info['geometry_type'],
                simplify_tol=tol,
                where_filter=where_filter)

            decay_kernel_path = os.path.join(
                local_workspace, f'{where_filter}_{pixel_units}.tif')

            mask_raster_path = (
                f'{os.path.splitext(reprojected_vector_path)[0]}.tif')
            geoprocessing.new_raster_from_base(
                working_base_raster_path, mask_raster_path, gdal.GDT_Byte,
                [0])
            LOGGER.debug(f'rasterize {mask_raster_path}')
            geoprocessing.rasterize(
                reprojected_vector_path, mask_raster_path, burn_values=[1],
                option_list=['ALL_TOUCHED=TRUE'])
        else:
            mask_raster_path = os.path.join(
                local_workspace,
                f"{raw_basename(row['path'])}_mask_{row['raster value']}.tif")

            if not square_blocksize(row['path']):
                working_row_raster_path = os.path.join(
                    local_workspace,
                    f'256x256_{os.path.basename(args.base_raster_path)}')
                geoprocessing.warp_raster(
                    row['path'], raster_info['pixel_size'],
                    working_row_raster_path, 'near')
            else:
                working_row_raster_path = row['path']

            geoprocessing.single_thread_raster_calculator(
                [(working_row_raster_path, 1)],
                mask_out_value_op(row['raster value']), mask_raster_path,
                gdal.GDT_Byte, 0)
        max_extent_in_pixel_units = convert_meters_to_pixel_units(
            working_base_raster_path, row[MAX_INFLUENCE_DIST_FIELD])[0]
        effective_extent_in_pixel_units = convert_meters_to_pixel_units(
            working_base_raster_path, row[INFLUENCE_DIST_FIELD])[0]

        for prob_key, prob_conversion in [('full', 1.0), ('current', args.probability_of_conversion)]:
            base_array = numpy.ones((2*int(max_extent_in_pixel_units)+1,)*2)
            base_array[base_array.shape[0]//2, base_array.shape[1]//2] = 0
            LOGGER.debug('calculate distance transform')
            decay_kernel = scipy.ndimage.distance_transform_edt(base_array)
            valid_mask = decay_kernel < (max(base_array.shape)/2)
            decay_kernel[~valid_mask] = 0
            # calculate what threshold of gaussian is 0.5 when p=1 and distance
            # is effective distance
            s_val = numpy.log(0.5)/(-(
                effective_extent_in_pixel_units/max_extent_in_pixel_units)**2)
            decay_kernel[valid_mask] = (
                numpy.exp(-(decay_kernel[valid_mask]/max_extent_in_pixel_units)**2)**(
                    s_val/prob_conversion))
            LOGGER.debug(s_val)
            LOGGER.debug(decay_kernel)

            decay_kernel_path = os.path.join(
                local_workspace,
                f"{raw_basename(row['path'])}_decay_{effective_extent_in_pixel_units}_{max_extent_in_pixel_units}_{prob_conversion}.tif")
            geoprocessing.numpy_array_to_raster(
                decay_kernel, None, [1, -1], [0, 0], None, decay_kernel_path)
            effect_path = (
                f'{os.path.splitext(mask_raster_path)[0]}_effect_{prob_conversion}.tif')
            LOGGER.debug(f'mask_raster_path info: {geoprocessing.get_raster_info(mask_raster_path)}')
            LOGGER.debug(f'decay_kernel_path info: {geoprocessing.get_raster_info(decay_kernel_path)}')
            LOGGER.debug(f'calculate effect for {effect_path}')
            geoprocessing.convolve_2d(
                (mask_raster_path, 1), (decay_kernel_path, 1), effect_path,
                ignore_nodata_and_edges=False, mask_nodata=False,
                normalize_kernel=True, target_datatype=gdal.GDT_Float64,
                target_nodata=None, working_dir=None, set_tol_to_zero=1e-8)
            effect_path_code_list[prob_key].extend([
                (effect_path, 1), (row[CONVERSION_CODE_FIELD], 'raw')])

    def sum_op(*array_list):
        result = numpy.zeros(array_list[0].shape)
        for array in array_list[0::2]:
            result += array
        return result

    for prob_type in ['full', 'conversion']:
        base_raster_path_list = [
            path_tuple[0] for path_tuple in effect_path_code_list[prob_type]]
        aligned_raster_path_list = [
            os.path.join(os.path.dirname(path), f'{raw_basename(path)}_aligned.tif')
            for path in base_raster_path_list]

        LOGGER.debug(
            f'aligning:\n{base_raster_path_list}\n\n\tto\n\n'
            f'{aligned_raster_path_list}')

        geoprocessing.align_and_resize_raster_stack(
            base_raster_path_list, aligned_raster_path_list,
            ['near']*len(aligned_raster_path_list),
            raster_info['pixel_size'], 'union')

        for index, aligned_raster_path in enumerate(aligned_raster_path_list):
            effect_path_code_list[prob_type][index*2] = (
                aligned_raster_path, 1)
        LOGGER.debug(
            f'algiend path code list: {effect_path_code_list[prob_type]}')

    full_effect_path = os.path.join(local_workspace, 'full_effect.tif')
    geoprocessing.raster_calculator(
        effect_path_code_list['full'], sum_op,
        full_effect_path, gdal.GDT_Float32, None)

    decayed_full_effect_path = os.path.join(local_workspace, 'decay_full_effect.tif')
    geoprocessing.raster_calculator(
        effect_path_code_list['current'], sum_op,
        decayed_full_effect_path, gdal.GDT_Float32, None)

    def conversion_op(base_lulc_array, decayed_effect_array, full_effect_array, *effect_path_code_list):
        result = base_lulc_array.copy()
        effect_sum = numpy.zeros(effect_path_code_list[0].shape)
        for array in effect_path_code_list[0::2]:
            effect_sum += array

        conversion_code = base_lulc_array.copy()
        max_effect_so_far = numpy.zeros(result.shape)
        for effect_array, code in zip(
                effect_path_code_list[0::2], effect_path_code_list[1::2]):
            # account for multiple pressures
            if not numpy.isnan(code):
                effect_larger_than_max = effect_array > max_effect_so_far
                conversion_code[effect_larger_than_max] = code
                max_effect_so_far[effect_larger_than_max] = (
                    effect_array[effect_larger_than_max])
        threshold_mask = decayed_effect_array >= full_effect_array
        result[threshold_mask] = conversion_code[threshold_mask]
        result[base_lulc_array == raster_info['nodata']] = (
            raster_info['nodata'])
        return result

    converted_raster_path = (
        f'{raw_basename(working_base_raster_path)}_'
        f'{raw_basename(args.infrastructure_scenario_path)}_'
        f'{args.probability_of_conversion}_.tif')
    geoprocessing.raster_calculator(
        [(working_base_raster_path, 1), (decayed_full_effect_path, 1), (full_effect_path, 1)] +
        effect_path_code_list['current'], conversion_op, converted_raster_path,
        raster_info['datatype'], raster_info['nodata'][0])


if __name__ == '__main__':
    main()

"""Land change scenarios.

Change in rasters by modeling the distance decay effect from other markers.
"""
import argparse
import os
import logging
import multiprocessing
import sys

from scipy.optimize import fsolve
import numpy
import scipy
import pyproj
from osgeo import gdal
from ecoshard import geoprocessing
from ecoshard.geoprocessing.geoprocessing_core \
    import DEFAULT_GTIFF_CREATION_TUPLE_OPTIONS
from ecoshard import taskgraph
import pandas


RASTER_CREATE_OPTIONS = DEFAULT_GTIFF_CREATION_TUPLE_OPTIONS[1]

logging.basicConfig(
    level=logging.DEBUG,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(levelname)s %(name)s'
        ' [%(pathname)s.%(funcName)s:%(lineno)d] %(message)s'),
    stream=sys.stdout)
LOGGER = logging.getLogger(__name__)
logging.getLogger('ecoshard.geoprocessing').setLevel(logging.WARN)
logging.getLogger('ecoshard.taskgraph').setLevel(logging.WARN)
LOGGER.setLevel(logging.DEBUG)

WORKSPACE_DIR = '_workspace_land_change_scenario'

PATH_FIELD = 'path'
GIS_TYPE_FIELD = 'type'
VECTOR_TYPE = 'vector'
RASTER_TYPE = 'raster'
GIS_TYPES = [RASTER_TYPE, VECTOR_TYPE]
VECTOR_KEY_FIELD = 'vector key'
VECTOR_VALUE_FIELD = 'vector value'
RASTER_VALUE_FIELD = 'raster value'
PARAM_VAL_AT_MIN_DIST_FIELD = 'value of parameter at min distance'
MAX_IMPACT_DIST_FIELD = 'max impact dist'

# Decay describes how quickly the effect falls off
DECAY_TYPE_FIELD = 'decay type'
LINEAR_DECAY_TYPE = 'linear'
EXPONENTIAL_DECAY_TYPE = 'exponential'
SIGMOID_DECAY_TYPE = 'sigmoid'
DECAY_TYPES = [LINEAR_DECAY_TYPE, EXPONENTIAL_DECAY_TYPE, SIGMOID_DECAY_TYPE]

# effect describes how the effect propagates from the source, is it
# a function of only how near it is, or is it how near and how many
# there are?
EFFECT_DISTANCE_TYPE_FIELD = 'effect distance type'
NEAREST_DISTANCE_TYPE = 'nearest'
CONVOLUTION_DISTANCE_TYPE = 'convolution'
DISTANCE_TYPES = [NEAREST_DISTANCE_TYPE, CONVOLUTION_DISTANCE_TYPE]

# Exponential decay alpha
ALPHA = 3


def mask_out_value_op(value):
    def _mask_out_value(x):
        result = (x == value)
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
            DECAY_TYPE_FIELD, PATH_FIELD,
            GIS_TYPE_FIELD, MAX_IMPACT_DIST_FIELD,
            VECTOR_VALUE_FIELD, PARAM_VAL_AT_MIN_DIST_FIELD,
            EFFECT_DISTANCE_TYPE_FIELD]:
        if field_name not in table:
            error_list.append(f'Expected field `{field_name}` but not found')
    if (VECTOR_VALUE_FIELD in table) ^ (VECTOR_KEY_FIELD in table):
        error_list.append(
            f'If attributes are used, expect both `{VECTOR_VALUE_FIELD}` '
            f'and `{VECTOR_KEY_FIELD}` to be defined but only one of them '
            f'was.')

    if error_list:
        raise ValueError(
            '\n\nThe following errors were detected when parsing the table:\n'
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

    return pixel_units


def _mask_op(raster_mask_value_list):
    # Use in raster calculator to make a mask based off the value list
    def _internal_mask_op(array):
        result = numpy.isin(array, raster_mask_value_list)
        return result
    return _internal_mask_op


def _mask_raster(task_graph, base_raster_path, row, target_raster_path):
    """Warp local."""
    base_raster_info = geoprocessing.get_raster_info(base_raster_path)
    raster_mask_value = row[RASTER_VALUE_FIELD]
    if numpy.isnan(raster_mask_value):
        # no need to further mask
        intermediate_raster_path = target_raster_path
    else:
        intermediate_raster_path = f'%s_warp_for_{str(raster_mask_value)}%s' % (
            os.path.splitext(target_raster_path))

    task_graph.add_task(
        func=geoprocessing.warp_raster,
        args=(
            base_raster_path, base_raster_info['pixel_size'],
            intermediate_raster_path, 'near'),
        kwargs={
            'target_bb': base_raster_info['bounding_box'],
            'target_projection_wkt': base_raster_info['projection_wkt'],
            'working_dir': os.path.dirname(intermediate_raster_path),
        },
        target_path_list=[intermediate_raster_path],
        task_name=f'warp {intermediate_raster_path}')

    if numpy.isnan(raster_mask_value):
        # just a warp is all that's needed
        return

    try:
        raster_mask_value_list = [float(raster_mask_value)]
    except TypeError:
        if ',' in raster_mask_value:
            split_char = ','
        else:
            split_char = ' '
        raster_mask_value_list = [
            float(x) for x in raster_mask_value.split(split_char)]

    task_graph.add_task(
        func=geoprocessing.raster_calculator,
        args=(
            [(intermediate_raster_path, 1)], _mask_op(raster_mask_value_list),
            target_raster_path, gdal.GDT_Byte, None),
        target_path_list=[target_raster_path],
        task_name=f'mask {target_raster_path} with {raster_mask_value_list}')


def _rasterize_vector(
        base_raster_path, vector_path, row,
        target_rasterized_vector_path):
    base_raster_info = geoprocessing.get_raster_info(base_raster_path)
    tol = base_raster_info['pixel_size'][0]/2
    target_projection_wkt = base_raster_info['projection_wkt']
    where_filter = None
    if VECTOR_KEY_FIELD in row and not numpy.isnan(
            row[VECTOR_KEY_FIELD]):
        where_filter = (
            f'{row[VECTOR_KEY_FIELD]}={row[RASTER_VALUE_FIELD]}')
    reprojected_vector_path = os.path.join(
        os.path.dirname(target_rasterized_vector_path),
        f'{raw_basename(row[PATH_FIELD])}_{where_filter}.gpkg')
    vector_info = geoprocessing.get_vector_info(row[PATH_FIELD])
    geoprocessing.reproject_vector(
        row[PATH_FIELD], target_projection_wkt,
        reprojected_vector_path, layer_id=0,
        driver_name='GPKG', copy_fields=True,
        geometry_type=vector_info['geometry_type'],
        simplify_tol=tol,
        where_filter=where_filter)

    mask_raster_path = (
        f'{os.path.splitext(reprojected_vector_path)[0]}.tif')
    geoprocessing.new_raster_from_base(
        base_raster_path, mask_raster_path, gdal.GDT_Byte, [0])
    LOGGER.info(f'rasterize {mask_raster_path}')
    geoprocessing.rasterize(
        reprojected_vector_path, mask_raster_path, burn_values=[1],
        option_list=['ALL_TOUCHED=TRUE'])


"""
print((numpy.cos((xp*numpy.pi))+1)/2)

y = -(numpy.sin(((x-0.5)*numpy.pi))-1)/2
plt.plot(x, y)
#y = numpy.exp(-x-1)
alpha = 3

def func(x):
    return [
        x[0]*numpy.exp(-alpha*(0-1))+x[1]-1,
        x[0]*numpy.exp(-alpha*(1-1))+x[1]]

from scipy.optimize import fsolve
A, B = fsolve(func, [1, 1])
y = A*numpy.exp(-alpha*(x-1))+B
plt.plot(x, y)

y = 1-x
"""

def decay_op(decay_type, max_dist, base_nodata, target_nodata):
    """Defines a function for decay based on the types in `DECAY_TYPES`."""
    def _calc_valid_mask(array, max_dist, base_nodata):
        valid_mask = (array <= max_dist)
        if base_nodata is not None:
            valid_mask &= (array != base_nodata)
        return valid_mask

    if decay_type == EXPONENTIAL_DECAY_TYPE:
        def func(x):
            return [
                x[0]*numpy.exp(-ALPHA*(0-1))+x[1]-1,
                x[0]*numpy.exp(-ALPHA*(1-1))+x[1]]
        A, B = fsolve(func, [1, 1])
        def _decay_op(array):
            valid_mask = _calc_valid_mask(array, max_dist, base_nodata)
            result = numpy.empty(array.shape)
            result[valid_mask] = A*numpy.exp(-ALPHA*(array[valid_mask]/max_dist-1))+B
            result[~valid_mask] = target_nodata
            return result
    elif decay_type == LINEAR_DECAY_TYPE:
        # a linear decay is just inverting the distance from the max
        def _decay_op(array):
            valid_mask = _calc_valid_mask(array, max_dist, base_nodata)
            result = numpy.empty(array.shape)
            result[valid_mask] = (max_dist-array[valid_mask])
            result[~valid_mask] = target_nodata
            return result
    elif decay_type == SIGMOID_DECAY_TYPE:
        def _decay_op(array):
            valid_mask = _calc_valid_mask(array, max_dist, base_nodata)
            result = numpy.empty(array.shape)
            result[valid_mask] = numpy.cos(numpy.pi*array[valid_mask]/max_dist)/2+0.5
            result[~valid_mask] = target_nodata
            return result
    return _decay_op


def main():
    """Entry point."""
    parser = argparse.ArgumentParser(description='Model land change')
    parser.add_argument('base_raster_path', help='Path to base raster.')
    parser.add_argument(
        'infrastructure_scenario_path', help='Path to land change pressure '
        f'table. Expected format is to have the columns:\n'
        f'\t`{PATH_FIELD}`: path to vector or raster\n'
        f'\t`{DECAY_TYPE_FIELD}`: decay type in {DECAY_TYPES} where:\n'
        '\t`path`: path to vector\n'
        f'\t`{MAX_IMPACT_DIST_FIELD}`: effective maximum distance of '
        'impact in meters\n'
        f'\t`{GIS_TYPE_FIELD}`: has one of the values {GIS_TYPES}\n'
        f'value in the raster produces the effect\n'
        f'\t`{VECTOR_KEY_FIELD}` (optional): field in the vector to filter '
        'by\n'
        f'\t`{VECTOR_VALUE_FIELD}` value in the key field to filter the '
        f'vector by if {VECTOR_KEY_FIELD} is defined\n'
        f'\t`{RASTER_VALUE_FIELD}` (optional): if raster type then what '
        f'value is used for the mask to weight\n'
        f'\t{PARAM_VAL_AT_MIN_DIST_FIELD}: the value in parameter units the '
        f'\tvalue should be at the closest distance\n'
        )
    parser.add_argument('--convert_mask_path',  help=(
        'Raster whose 1 values indicate areas that should be affected, '
        'if not set, all areas are affected.'))
    args = parser.parse_args()

    infrastructure_scenario_table = load_table(
        args.infrastructure_scenario_path)

    local_workspace = os.path.join(
        WORKSPACE_DIR, raw_basename(args.infrastructure_scenario_path))
    os.makedirs(local_workspace, exist_ok=True)

    task_graph = taskgraph.TaskGraph(local_workspace, n_workers=-1)

    # Align input rasters from table and --convert_mask_path to
    # base_raster_path
    pressure_mask_raster_list = []
    for index, row in infrastructure_scenario_table.iterrows():
        LOGGER.info(f'processing path {row[PATH_FIELD]}')
        LOGGER.info(f'{row[RASTER_VALUE_FIELD]}: {numpy.isnan(row[RASTER_VALUE_FIELD])}')
        pressure_mask_raster_info = {}
        if row[GIS_TYPE_FIELD] == RASTER_TYPE:
            raster_path = row[PATH_FIELD]
            local_path = os.path.join(
                local_workspace, os.path.basename(raster_path))
            pressure_mask_raster_info[PATH_FIELD] = local_path
            _mask_raster(task_graph, raster_path, row, local_path)
        elif row[GIS_TYPE_FIELD] == VECTOR_TYPE:
            vector_path = row[PATH_FIELD]
            rasterized_vector_path = os.path.join(
                local_workspace, f'{raw_basename(vector_path)}.tif')
            _rasterize_vector(
                args.base_raster_path, vector_path, row,
                rasterized_vector_path)
            pressure_mask_raster_info[PATH_FIELD] = rasterized_vector_path

        pressure_mask_raster_info.update({
            x: row[x] for x in [
                DECAY_TYPE_FIELD, PARAM_VAL_AT_MIN_DIST_FIELD,
                MAX_IMPACT_DIST_FIELD, RASTER_VALUE_FIELD]
            })
        pressure_mask_raster_list.append(pressure_mask_raster_info)
        del pressure_mask_raster_info

    LOGGER.debug(pressure_mask_raster_list)
    # At this point, all the paths in pressure_mask_raster_list are rasterized and
    # ready to be spread over space

    for pressure_mask_raster_dict in pressure_mask_raster_list:
        # save the mask for later in case we need to mask it out further
        # before we
        pressure_mask_raster_path = pressure_mask_raster_dict[PATH_FIELD]
        LOGGER.info(
            f'processing mask {pressure_mask_raster_path}/{row}')
        max_extent_in_pixel_units = convert_meters_to_pixel_units(
            pressure_mask_raster_path, row[MAX_IMPACT_DIST_FIELD])[0]

        effect_path = (
            f'{os.path.splitext(pressure_mask_raster_path)[0]}_'
            f'{row[EFFECT_DISTANCE_TYPE_FIELD]}_'
            f'{row[DECAY_TYPE_FIELD]}_effect.tif')
        effect_path_list.append(effect_path)

        if row[EFFECT_DISTANCE_TYPE_FIELD] == NEAREST_DISTANCE_TYPE:
            # TODO: distance transform
            # TODO: pass distance transform to the correct kind of decay function
            #   that function will first subtract by the max distance then divide by
            #   it so we get a 1 to 0 (and negative) distance, from there apply
            #   exponential/sigmoid/linear decay as appropriate.
            nearest_dist_raster_path = '%s_nearest_dist%s' % os.splitext(
                pressure_mask_raster_path)
            geoprocessing.distance_transform_edt(
                (pressure_mask_raster_path, 1), nearest_dist_raster_path,
                working_dir=os.dirname(nearest_dist_raster_path))

            target_nodata = -1
            geoprocessing.raster_calculator(
                [(nearest_dist_raster_path, 1)], decay_op(
                    row[DECAY_TYPE_FIELD], max_extent_in_pixel_units, None,
                    target_nodata),
                effect_path, gdal.GDT_Float32, target_nodata)

        elif row[EFFECT_DISTANCE_TYPE_FIELD] == CONVOLUTION_DISTANCE_TYPE:
            # build a distance transform kernel by converting meter extent
            # to pixel extent
            base_array = numpy.ones((2*int(max_extent_in_pixel_units)+1,)*2)
            base_array[base_array.shape[0]//2, base_array.shape[1]//2] = 0
            decay_kernel = scipy.ndimage.distance_transform_edt(base_array)
            # only valid where it's <= than the maximum distance defined
            # in meters but turned to pixels
            valid_mask = decay_kernel <= (max(base_array.shape)/2)
            decay_kernel[~valid_mask] = 0

            decay_kernel[valid_mask] = decay_op(
                row[DECAY_TYPE_FIELD], max_extent_in_pixel_units)(
                decay_kernel[valid_mask], 0, 0)
            decay_kernel /= numpy.sum(decay_kernel)

            decay_kernel_path = os.path.join(
                local_workspace,
                f"{raw_basename(pressure_mask_raster_path)}_decay_{effective_extent_in_pixel_units}_{max_extent_in_pixel_units}_{args.probability_of_conversion}.tif")
            geoprocessing.numpy_array_to_raster(
                decay_kernel, None, [1, -1], [0, 0], None, decay_kernel_path)
            LOGGER.debug(f'calculate effect for {effect_path}')
            geoprocessing.convolve_2d(
                (pressure_mask_raster_path, 1), (decay_kernel_path, 1), effect_path,
                ignore_nodata_and_edges=False, mask_nodata=False,
                normalize_kernel=False, target_datatype=gdal.GDT_Float64,
                target_nodata=None, working_dir=None, set_tol_to_zero=1e-8,
                n_workers=multiprocessing.cpu_count()//4)

    def conversion_op(base_raster, *decay_effect_list):
        valid_mask = (base_raster_path != nodata) & ~(
            numpy.isnan(base_raster_path))
        decay_effect_exponent = numpy.zeros(base_raster_path.shape)

        for decay_effect_array in decay_effect_list:
            decay_effect_exponent[valid_mask] += numpy.log(
                1-decay_effect_array[valid_mask])
        result = numpy.empty(base_raster.shape)
        result[valid_mask] = base_raster[valid_mask] * numpy.exp(
            decay_effect_exponent[valid_mask])
        return result

    converted_raster_path = (
        f'{raw_basename(args.base_raster_path)}_'
        f'{raw_basename(args.infrastructure_scenario_path)}_'
        f'{args.probability_of_conversion}_.tif')
    geoprocessing.single_thread_raster_calculator(
        [(args.base_raster_path, 1)]+[(path, 1) for path in effect_path_list],
        conversion_op, converted_raster_path,
        raster_info['datatype'], raster_info['nodata'][0])

    # TODO: I think there's a 'min/max' conversion level and stuff that needs to
    #   go in there'


if __name__ == '__main__':
    main()
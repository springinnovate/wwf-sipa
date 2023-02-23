"""Land change scenarios.

Model land use change by modeling the distance decay effect from new
infrastructure.

"""
import argparse
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
    level=logging.WARNING,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(levelname)s %(name)s'
        ' [%(pathname)s.%(funcName)s:%(lineno)d] %(message)s'),
    stream=sys.stdout)
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)

WORKSPACE_DIR = '_workspace_land_change_scenario'

PATH_FIELD = 'path'
INFLUENCE_DIST_FIELD = 'effective impact dist'
CONVERSION_CODE_FIELD = 'conversion code'
ATTRIBUTE_KEY_FIELD = 'attribute key'
ATTRIBUTE_VALUE_FIELD = 'attribute code'


def raw_basename(path): return os.path.basename(os.path.splitext(path)[0])

def load_table(table_path):
    """Load infrastructure table and raise errors if needed."""
    table = pandas.read_csv(table_path)
    error_list = []
    for field_name in [
            INFLUENCE_DIST_FIELD, CONVERSION_CODE_FIELD, PATH_FIELD]:
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
        f'\t`{CONVERSION_CODE_FIELD}`: if target conversion landscape code\n'
        f'additional parameters include `{ATTRIBUTE_KEY_FIELD}` and '
        f'`{ATTRIBUTE_VALUE_FIELD}` to allow subsets of the vectors where\n'
        f'\t`{ATTRIBUTE_KEY_FIELD}` (optional): field in the vector to filter '
        'by\n'
        f'\t`{ATTRIBUTE_VALUE_FIELD}` (optional): if field is defined, what '
        'value to match')
    args = parser.parse_args()

    infrastructure_scenario_table = load_table(
        args.infrastructure_scenario_path)
    local_workspace = os.path.join(
        WORKSPACE_DIR, raw_basename(args.infrastructure_scenario_path))
    os.makedirs(local_workspace, exist_ok=True)

    raster_info = geoprocessing.get_raster_info(args.base_raster_path)
    for index, row in infrastructure_scenario_table.iterrows():
        pixel_units = convert_meters_to_pixel_units(
            args.base_raster_path, row[INFLUENCE_DIST_FIELD])

        tol = raster_info['pixel_size'][0]/2
        where_filter = None
        if ATTRIBUTE_KEY_FIELD in row:
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

        base_array = numpy.ones(([2*int(v)+1 for v in pixel_units]))
        base_array[base_array.shape[0]//2, base_array.shape[1]//2] = 0
        LOGGER.debug('calculate distance transform')
        decay_kernel = scipy.ndimage.distance_transform_edt(
            base_array)
        valid_mask = decay_kernel < (max(base_array.shape)/2)
        decay_kernel[~valid_mask] = 0
        decay_kernel[valid_mask] = (
            numpy.max(decay_kernel)-decay_kernel[valid_mask])

        LOGGER.debug(decay_kernel)
        geoprocessing.numpy_array_to_raster(
            decay_kernel, None, [1, -1], [0, 0], None, decay_kernel_path)

        mask_raster_path = (
            f'{os.path.splitext(reprojected_vector_path)[0]}.tif')
        geoprocessing.new_raster_from_base(
            args.base_raster_path, mask_raster_path, gdal.GDT_Byte,
            [0])
        LOGGER.debug(f'rasterize {mask_raster_path}')
        geoprocessing.rasterize(
            reprojected_vector_path, mask_raster_path, burn_values=[1],
            option_list=['ALL_TOUCHED=TRUE'])

        effect_path = (
            f'{os.path.splitext(reprojected_vector_path)[0]}_effect.tif')
        geoprocessing.convolve_2d(
            (mask_raster_path, 1), (decay_kernel_path, 1), effect_path,
            ignore_nodata_and_edges=False, mask_nodata=False,
            normalize_kernel=True, target_datatype=gdal.GDT_Float64,
            target_nodata=None, working_dir=None, set_tol_to_zero=1e-8)
        LOGGER.debug(pixel_units)
        return


if __name__ == '__main__':
    main()

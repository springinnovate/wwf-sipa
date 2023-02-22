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

from osgeo import osr
from osgeo import ogr
from osgeo import gdal
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

TABLE_PATH_FIELD = 'path'
INFLUENCE_DIST_FIELD = 'effective impact dist'
CONVERSION_CODE_FIELD = 'conversion code'
ATTRIBUTE_KEY_FIELD = 'attribute key'
ATTRIBUTE_VALUE_FIELD = 'attribute code'



def raw_basename(path): return os.path.basename(os.path.splitext(path)[0])


def main():
    """Entry point."""
    parser = argparse.ArgumentParser(description='Model land change')
    parser.add_argument('base_raster_path', help='Path to base raster.')
    parser.add_argument(
        'infrastructure_scenario_path', help='Path to land change pressure '
        f'table. Expected format is to have the columns `{TABLE_PATH_FIELD}`, '
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

    infrastructure_scenario_table = pandas.read_csv(
        args.infrastructure_scenario_path)
    local_workspace = os.path.join(
        WORKSPACE_DIR, raw_basename(args.infrastructure_scenario_path))
    os.makedirs(local_workspace, exist_ok=True)

    for index, row in infrastructure_scenario_table.iterrows():
        LOGGER.debug(row['ID'])


if __name__ == '__main__':
    main()

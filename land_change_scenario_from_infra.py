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


def main():
    """Entry point."""
    parser = argparse.ArgumentParser(description='Model land change')
    parser.add_argument('base_raster_path', help='Path to base raster.')
    parser.add_argument(
        'infrastructure_scenario_path', help='Path to land change pressure '
        'table. Expected format is to have the columns `path`, '
        '`influence dist`, and `conversion code` where:\n'
        '\t`path`: path to vector\n'
        '\t`influence dist`: maximum influence distance in meters\n'
        '\t`conversion code`: if target conversion landscape code\n'
        'additional parameters include `attribute key` and `attribute value` '
        'to allow subsets of the vectors where\n'
        '\t`attribute key` (optional): field in the vector to filter by\n'
        '\t`attribute code` (optional): if field is defined, what value to '
        'match')
    args = parser.parse_args()

    pandas.read_csv(args.infrastructure_scenario_path)

    os.makedirs(WORKSPACE_DIR, exist_ok=True)


if __name__ == '__main__':
    main()

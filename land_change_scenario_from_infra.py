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
        'change_pressure_table_path', help='Path to land change pressure '
        'table.')
    args = parser.parse_args()

    os.makedirs(WORKSPACE_DIR, exist_ok=True)


if __name__ == '__main__':
    main()

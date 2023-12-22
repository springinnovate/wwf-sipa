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
import collections
import os
import logging
import sys
import subprocess
import shutil
import tempfile

from ecoshard import taskgraph
from ecoshard import geoprocessing
from osgeo import gdal
import numpy
from osgeo import ogr

logging.basicConfig(
    level=logging.DEBUG,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(levelname)s %(name)s'
        ' [%(pathname)s.%(funcName)s:%(lineno)d] %(message)s'),
    stream=sys.stdout)
logging.getLogger('ecoshard.taskgraph').setLevel(logging.INFO)
LOGGER = logging.getLogger(__name__)

REGIONS_TO_ANALYZE = ['PH', 'IDN']

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
        "D:/repositories/wwf-sipa/data/admin_boundaries/summed_services/10_PH_conservation_inf_service_overlap_count.tif",
        "D:/repositories/wwf-sipa/data/admin_boundaries/summed_services/10_PH_restoration_service_overlap_count.tif",],
    'IDN': [
        "D:/repositories/wwf-sipa/data/admin_boundaries/summed_services/10_IDN_conservation_inf_service_overlap_count.tif",
        "D:/repositories/wwf-sipa/data/admin_boundaries/summed_services/10_IDN_restoration_service_overlap_count.tif",],
}

AOI_REGIONS = {
    'PH': {
        'municipality': ('NAME_1', r"D:\repositories\wwf-sipa\data\admin_boundaries\IDN_adm1.gpkg"),
        'visayas': (None, r"D:\repositories\wwf-sipa\data\island_groups\ph_visayas.gpkg"),
        'luzon': (None, r"D:\repositories\wwf-sipa\data\island_groups\ph_luzon.gpkg"),
        'mindanao': (None, r"D:\repositories\wwf-sipa\data\island_groups\ph_mindanao.gpkg"),
    },
    'IDN': {
        'provence': ('NAME_1', r"D:\repositories\wwf-sipa\data\admin_boundaries\IDN_adm1.gpkg"),
        'java': (None, r"D:\repositories\wwf-sipa\data\island_groups\idn_java.gpkg"),
        'kalimantan': (None, r"D:\repositories\wwf-sipa\data\island_groups\idn_kalimantan.gpkg"),
        'maluku_islands': (None, r"D:\repositories\wwf-sipa\data\island_groups\idn_maluku_islands.gpkg"),
        'nusa_tenggara': (None, r"D:\repositories\wwf-sipa\data\island_groups\idn_nusa_tenggara.gpkg"),
        'paupa': (None, r"D:\repositories\wwf-sipa\data\island_groups\idn_paupa.gpkg"),
        'sulawesi': (None, r"D:\repositories\wwf-sipa\data\island_groups\idn_sulawesi.gpkg"),
        'sumatra': (None, r"D:\repositories\wwf-sipa\data\island_groups\idn_sumatra.gpkg"),
    },
}


def route_dem(dem_path, raster_to_align_path, flow_dir_path):
    """Turn DEM into flow direction raster."""
    basename = os.path.dirname(os.path.splitext(flow_dir_path)[0])
    temp_dir = tempfile.mkdtemp(
        dir=os.path.dirname(flow_dir_path), prefix=f'route_dem_{basename}')



def main():
    """Entry point."""
    # delineate the areas downstream of 10% mask
    # delinate areas <2m within 2km of the coast
    # use these to mask out population rasters
    # then count people in:
    #   * total
    #   * per AOI region (island group and municipality (PH) or provence (IDN) AOI)
    pass


if __name__ == '__main__':
    main()

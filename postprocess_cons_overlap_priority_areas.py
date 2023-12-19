"""
Conservation priorities overlap analysis:
1) how much of high value ES area (top 10% map) is outside of current
   protected area (PAs) network? Then,
2) overlap the areas of top 10% outside of the PA with KBAs and answer:
    2a) How much of high value ES area would be priorities for protection or
        restoration for ES?

Desired output: maps, % values
Inputs to use:
    PH PA   D:/repositories/wwf-sipa/data/protected_areas/PH_Combined_PAs
    PH KBA  D:/repositories/wwf-sipa/data/protected_areas/PH_KBA
    IDN PA  D:/repositories/wwf-sipa/data/protected_areas/ID_Combined PAs
    IDN KBA D:/repositories/wwf-sipa/data/protected_areas/Indonesia_KBA
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


REGIONS_TO_ANALYZE = ['PH', 'IDN']
PROTECTED_AREAS = {
    'PH': 'D:/repositories/wwf-sipa/data/protected_areas/PH_Combined_PAs',
    'IDN': 'D:/repositories/wwf-sipa/data/protected_areas/ID_Combined PAs',
}

KEY_BIODIVERSITY_AREAS = {
    'PH': 'D:/repositories/wwf-sipa/data/protected_areas/PH_KBA',
    'IDN': 'D:/repositories/wwf-sipa/data/protected_areas/Indonesia_KBA',
}

SERVICE_OVERLAP_RASTERS = {
    'PH': '',
    'IDN': '',
}

RESULTS_DIR = f'workspace_{os.path.basename(os.path.splitext(__file__)[0])}'
WORKING_DIR = os.path.join(RESULTS_DIR, 'working_dir')
for dir_path in [RESULTS_DIR, WORKING_DIR]:
    os.path.makedirs(dir_path)


def main():
    """Entry point."""
    task_graph = taskgraph.TaskGraph(RESULTS_DIR, os.cpu_count()//2+2, 15.0)

    for region_id in REGIONS_TO_ANALYZE:
        service_overlap_raster_path = SERVICE_OVERLAP_RASTERS[region_id]
        service_overlap_in_pa_path = os.path.join(
            WORKING_DIR,
            f'%s_{region_id}_protected_areas%s' % os.path.splitext(
                service_overlap_raster_path)
            )
        pa_overlap_task = task_graph.add_task(
            func=geoprocessing.mask_raster,
            args=(
                service_overlap_raster_path,
                PROTECTED_AREAS[region_id],
                service_overlap_in_pa_path),
            kwargs={
                'working_dir': WORKING_DIR,
                'all_touched': True,
                'allow_different_blocksize': True},
            target_path_list=[service_overlap_in_pa_path],
            task_name=f'pa overlap for {region_id}')

        service_overlap_in_kba_path = os.path.join(
            WORKING_DIR,
            f'%s_{region_id}_protected_areas%s' % os.path.splitext(
                service_overlap_raster_path)
            )
        kba_overlap_task = task_graph.add_task(
            func=geoprocessing.mask_raster,
            args=(
                service_overlap_raster_path,
                KEY_BIODIVERSITY_AREAS[region_id],
                service_overlap_in_kba_path),
            kwargs={
                'working_dir': WORKING_DIR,
                'all_touched': True,
                'allow_different_blocksize': True},
            target_path_list=[service_overlap_in_kba_path],
            task_name=f'kba overlap for {region_id}')


if __name__ == '__main__':
    main()
"""Convert ."""
import argparse
import os
import logging
import sys

from osgeo import gdal
from ecoshard import geoprocessing
from ecoshard import taskgraph
import numpy

NODATA = -1
GLOBAL_WORKSPACE_DIR = 'flood_risk_workspace'

logging.basicConfig(
    level=logging.DEBUG,
    stream=sys.stdout,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(levelname)s %(name)s'
        ' [%(funcName)s:%(lineno)d] %(message)s'))
LOGGER = logging.getLogger(os.path.splitext(os.path.basename(__file__))[0])
LOGGER.setLevel(logging.DEBUG)
logging.getLogger('ecoshard.fetch_data').setLevel(logging.INFO)


def year_flood_to_prob(*args):
    """args: [flood_map0, flood_years0, flood_map1, flood_years1, ...]"""
    flood_mask_year_tuples = [
        (args[i], args[i+1]) for i in range(0, len(args), 2)]
    flood_risk = numpy.zeros(args[0].shape, dtype=float)
    for flood_mask, year in sorted(
            flood_mask_year_tuples, key=lambda x: x[1], reverse=True):
        flood_risk[flood_mask.astype(bool)] = 1/year
    return flood_risk


def main():
    """Entrypoint."""
    parser = argparse.ArgumentParser(
        description='Distributed flood risk analysis.')
    parser.add_argument(
        'flood_risk_year_path_list', nargs='+', help=(
            'Path to flood risk followed by a = and an integer indicating '
            'what flood year risk level it is i.e. path1=1 path2=5 '
            'path3=100'))
    parser.add_argument(
        '--target_raster_path', help='Path to desired output path.')
    parser.add_argument(
        '--file_prefix', default='',
        help='Added to intermediate files to avoid collision.')
    args = parser.parse_args()

    os.makedirs(GLOBAL_WORKSPACE_DIR, exist_ok=True)
    task_graph = taskgraph.TaskGraph(
        GLOBAL_WORKSPACE_DIR, len(args.flood_risk_year_path_list), 15.0)

    # align flood risk maps
    flood_risk_year_path_list = [
        path.split('=') for path in args.flood_risk_year_path_list]
    bounding_box_list = [
        geoprocessing.get_raster_info(path_year_tuple[0])['bounding_box']
        for path_year_tuple in flood_risk_year_path_list]
    target_bounding_box = geoprocessing.merge_bounding_box_list(
        bounding_box_list, 'intersection')
    raster_info = geoprocessing.get_raster_info(
        flood_risk_year_path_list[0][0])
    aligned_flood_risk_path_list = []
    for raster_path, year in flood_risk_year_path_list:
        aligned_raster_path = os.path.join(
            GLOBAL_WORKSPACE_DIR,
            f'{args.file_prefix}_aligned_flood_{year}.tif')
        task_graph.add_task(
            func=geoprocessing.warp_raster,
            args=(
                raster_path, raster_info['pixel_size'], aligned_raster_path,
                'near'),
            kwargs={
                'target_bb': target_bounding_box,
                'target_projection_wkt': raster_info['projection_wkt']},
            target_path_list=[aligned_raster_path],
            task_name=f'align {aligned_raster_path}')
        aligned_flood_risk_path_list.append((aligned_raster_path, 1))
        aligned_flood_risk_path_list.append((int(year), 'raw'))
    task_graph.join()
    LOGGER.debug(f'********* processing {aligned_flood_risk_path_list}')
    # calc year flood to prob on them
    task_graph.add_task(
        func=geoprocessing.raster_calculator,
        args=(
            aligned_flood_risk_path_list, year_flood_to_prob,
            args.target_raster_path, gdal.GDT_Float32, None),
        target_path_list=[args.target_raster_path],
        transient_run=True,
        task_name=f'calc flood risk raster {args.target_raster_path}')

    task_graph.join()
    task_graph.close()


if __name__ == '__main__':
    main()

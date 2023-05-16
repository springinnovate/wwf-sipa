"""Calculate downstream vector coverage."""
import argparse
import os

from ecoshard.geoprocessing import routing
from ecoshard import taskgraph
from osgeo import gdal

def main():
    """Entrypoint."""
    parser = argparse.ArgumentParser(description='Downstream vector intersection')
    parser.add_argument('dem_path', help='path to DEM file')
    parser.add_argument('vector_path', help='path to vector coverage')
    args = parser.parse_args()

    target_dir = (
        f'{os.path.splitext(os.path.basename(dem_path))[0]}_'
        f'{os.path.splitext(os.path.basename(vector_path))[0]}')

    workspace_dir = os.path.join(
        'downstream_vector_coverage_workspace',
        target_dir)
    os.makedirs(workspace_dir, exist_ok=True)

    task_graph = taskgraph.TaskGraph(workspace_dir, 2)

    # pitfill the DEM
    filled_dem_raster_path = os.path.join(
        workspace_dir, 'filled_dem.tif')
    fill_pits_task = task_graph.add_task(
        func=routing.fill_pits,
        args=(
            (args.dem_path, 1), filled_dem_raster_path),
        kwargs={
            'working_dir': workspace_dir,
            'max_pixel_fill_count': -1},
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

    rasterized_vector_path = os.path.join(
        workspace_dir, 'rasterized_vector.tif')
    new_raster_task = task_graph.add_task(
        func=geoprocessing.new_raster_from_base,
        args=(
            args.dem_path, rasterized_vector_path, gdal.GDT_Byte, [0]),
        target_path_list=[rasterized_vector_path],
        task_name='new raster')

    # rasterize the vector
    rasterize_task = task_graph.add_task(
        func=geoprocessing.rasterize,
        args=(args.vector_path, rasterized_vector_path),
        kwargs={'burn_values': [1]},
        dependent_task_list=[new_raster_task],
        target_path_list=[rasterized_vector_path],
        task_name='rasterize')

    # flow accum the vector on the routed DEM
    target_flow_accum_raster_path = os.path.join(
        workspace_dir, 'downstream_vector_coverage.tif')
    task_graph.add_task(
        func=routing.flow_accumulation_mfd,
        args=(
            (flow_dir_mfd_raster_path, 1), target_flow_accum_raster_path),
        kwargs={
            'weight_raster_path_band': (rasterized_vector_path, 1)},
        target_path_list=[target_flow_accum_raster_path],
        dependent_task_list=[rasterize_task, flow_dir_mfd_task],
        task_name='flow accumulation')

    task_graph.close()
    task_graph.join()


if __name__ == '__main__':
    main()

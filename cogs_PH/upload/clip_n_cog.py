import time
from osgeo import gdal
from ecoshard import geoprocessing
from ecoshard import taskgraph
import os
import glob
import logging
import sys

logging.basicConfig(
    level=logging.DEBUG,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(levelname)s %(name)s'
        ' [%(funcName)s:%(lineno)d] %(message)s'),
    stream=sys.stdout)
LOGGER = logging.getLogger(__name__)


def _make_logger_callback(message, timeout=5.0):
    """Build a timed logger callback that prints ``message`` replaced.

    Args:
        message (string): a string that expects 2 placement %% variables,
            first for % complete from ``df_complete``, second from
            ``p_progress_arg[0]``.
        timeout (float): number of seconds to wait until print

    Returns:
        Function with signature:
            logger_callback(df_complete, psz_message, p_progress_arg)

    """
    def logger_callback(df_complete, _, p_progress_arg):
        """Argument names come from the GDAL API for callbacks."""
        current_time = time.time()
        if ((current_time - logger_callback.last_time) > timeout or
                (df_complete == 1.0 and
                 logger_callback.total_time >= timeout)):
            # In some multiprocess applications I was encountering a
            # ``p_progress_arg`` of None. This is unexpected and I suspect
            # was an issue for some kind of GDAL race condition. So I'm
            # guarding against it here and reporting an appropriate log
            # if it occurs.
            progress_arg = ''
            if p_progress_arg is not None:
                progress_arg = p_progress_arg[0]

            LOGGER.info(message, df_complete * 100, progress_arg)
            logger_callback.last_time = current_time
            logger_callback.total_time += current_time
    logger_callback.last_time = time.time()
    logger_callback.total_time = 0.0

    return logger_callback


def cogit(file_path):
    # create copy with COG
    cog_driver = gdal.GetDriverByName('COG')
    base_raster = gdal.OpenEx(file_path, gdal.OF_RASTER)
    cog_file_path = os.path.join(
        f'cog_{os.path.basename(file_path)}')
    options = ('COMPRESS=LZW', 'NUM_THREADS=ALL_CPUS', 'BIGTIFF=YES')
    LOGGER.info(f'convert {file_path} to COG {cog_file_path} with {options}')
    cog_raster = cog_driver.CreateCopy(
        cog_file_path, base_raster, options=options,
        callback=_make_logger_callback(
            f"COGing {cog_file_path} %.1f%% complete %s"))
    del cog_raster


def main():
    mask_dir = 'masked'
    task_graph = taskgraph.TaskGraph(mask_dir, os.cpu_count(), 15)
    os.makedirs(mask_dir, exist_ok=True)
    for raster_path in glob.glob('cog_*.tif'):
        raster_info = geoprocessing.get_raster_info(raster_path)
        masked_raster_path = os.path.join(
            mask_dir,
            os.path.basename(raster_path)[4:])
        mask_task = task_graph.add_task(
            func=geoprocessing.warp_raster,
            args=(
                raster_path, raster_info['pixel_size'], masked_raster_path,
                'near'),
            kwargs={
                'vector_mask_options': {
                    'mask_vector_path': r"D:\repositories\wwf-sipa\data\admin_boundaries\PH_outline.gpkg"},
                'working_dir': mask_dir},
            target_path_list=[masked_raster_path],
            task_name=f'mask {masked_raster_path}')
        _ = task_graph.add_task(
            func=cogit,
            args=(masked_raster_path,),
            dependent_task_list=[mask_task],
            task_name=f'cog {masked_raster_path}')
    task_graph.join()
    task_graph.close()


if __name__ == '__main__':
    main()

from datetime import datetime
import collections
from pathlib import Path
import csv
import glob
import itertools
import logging
import numpy
import operator
import os
import sys
import tempfile
import time

from ecoshard import geoprocessing
from ecoshard import taskgraph
from matplotlib.colors import LinearSegmentedColormap
import matplotlib
matplotlib.use('Agg')
from concurrent.futures import ThreadPoolExecutor
from osgeo import gdal
from osgeo import osr
import geopandas
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas

logging.basicConfig(
    level=logging.DEBUG,
    stream=sys.stdout,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(levelname)s %(name)s'
        ' [%(pathname)s.%(funcName)s:%(lineno)d] %(message)s'))
LOGGER = logging.getLogger(__name__)
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
logging.getLogger('PIL').setLevel(logging.ERROR)
logging.getLogger('ecoshard.taskgraph').setLevel(logging.INFO)

CUSTOM_STYLE_DIR = 'custom_styles'
WORKING_DIR = 'fig_generator_dir_2024_12_11'
timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
FIG_DIR = os.path.join(WORKING_DIR, f'rendered_figures_{timestamp}')
ALGINED_DIR = os.path.join(WORKING_DIR, 'aligned_rasters')
OVERLAP_DIR = os.path.join(WORKING_DIR, 'overlap_rasters')
SCALED_DIR = os.path.join(WORKING_DIR, 'scaled_rasters')
COMBINED_SERVICE_DIR = os.path.join(WORKING_DIR, 'combined_services')
COG_DIR = os.path.join(WORKING_DIR, 'cog')
PA_KBA_OVERLAP_DIR = os.path.join(WORKING_DIR, 'pa_kba_raster_overlaps')
for dir_path in [
        WORKING_DIR,
        FIG_DIR,
        ALGINED_DIR,
        OVERLAP_DIR,
        SCALED_DIR,
        PA_KBA_OVERLAP_DIR,
        COG_DIR]:
    os.makedirs(dir_path, exist_ok=True)

REMOTE_BUCKET_PATH = 'gs://ecoshard-root/wwf_sipa_viewer_2024_07_10/'

ROOT_DATA_DIR = './post_processing_results_updated_R_worstcase_2024_11_21'
PROTECTED_AREAS = {
    'PH': './data/protected_areas/PH_Combined_PAs',
    'IDN': './data/protected_areas/ID_Combined PAs',
}
KEY_BIODIVERSITY_AREAS = {
    'PH': './data/protected_areas/PH_KBA',
    'IDN': './data/protected_areas/Indonesia_KBA',
}

GLOBAL_PIXEL_SIZE = (0.0008333333333333333868, -0.0008333333333333333868)
LOW_PERCENTILE = 10
HIGH_PERCENTILE = 90
BASE_FONT_SIZE = 12
GLOBAL_FIG_SIZE = 10
GLOBAL_DPI = 800
ELLIPSOID_EPSG = 6933

RASTER_STYLE_LOG_PATH = 'viewer_info.txt'
FLOOD_MITIGATION_SERVICE = 'flood mitigation'
RECHARGE_SERVICE = 'recharge'
SEDIMENT_SERVICE = 'sediment'
CV_SERVICE = 'coastal vulnerability'
RESTORATION_SCENARIO = 'restoration'
CONSERVATION_SCENARIO_INF = 'conservation_inf'
CONSERVATION_SCENARIO_ALL = 'conservation_all'
EACH_ECOSYSTEM_SERVICE_ID = 'each ecosystem service'
OVERLAPPING_SERVICES_ID = 'overlapping services'
ROAD_AND_PEOPLE_BENFICIARIES_ID = 'road and people beneficiaries'
PEOPLE_ONLY_BENEFICIARIES_ID = 'people beneficiaries'
CONSERVATION_OVERLAP_HEATMAP = 'conservation overlap heatmap'
RESTORATION_OVERLAP_HEATMAP = 'restoration overlap heatmap'

NODATA_COLOR = '#ffffff'
COLOR_LIST = {
    ROAD_AND_PEOPLE_BENFICIARIES_ID: [NODATA_COLOR, '#674ea7', '#a64d79', '#4c1130'],
    PEOPLE_ONLY_BENEFICIARIES_ID: [NODATA_COLOR, '#a64d79'],
    SEDIMENT_SERVICE: [NODATA_COLOR, '#ffbd4b', '#cc720a', '#8c4e07', '#4d2b04'],
    RECHARGE_SERVICE: [NODATA_COLOR, '#cfffff', '#72b1cc', '#4f7abc', '#2b424d'],
    FLOOD_MITIGATION_SERVICE: [NODATA_COLOR, '#d9fff8', '#adccc6', '#778c88', '#414d4a'],
    CV_SERVICE: [NODATA_COLOR, '#f4cccc', '#ea9999', '#e06666', '#990000'],
    EACH_ECOSYSTEM_SERVICE_ID: [NODATA_COLOR, '#CC720A', '#72B1CC', '#ADCCC6', '#C1CC43', '#000000'],
    OVERLAPPING_SERVICES_ID: [NODATA_COLOR, '#8C4E07', '#4F7A8C', '#778C88', '#848C2E', '#F0027F'],
    CONSERVATION_OVERLAP_HEATMAP: [NODATA_COLOR, '#DCEAB6', '#B3C169', '#828434', '#464A1E'],
    RESTORATION_OVERLAP_HEATMAP: [NODATA_COLOR, '#D0E4D6', '#9BC8AA', '#42A16E', '#008742']
}

COUNTRY_OUTLINE_PATH = {
    'PH': "data/admin_boundaries/PH_outline.gpkg",
    'IDN': "data/admin_boundaries/IDN_outline.gpkg",
}


FILENAMES = {
    'PH': {
        RESTORATION_SCENARIO: {
            FLOOD_MITIGATION_SERVICE: {
                'diff': os.path.join(ROOT_DATA_DIR, 'diff_flood_mitigation_PH_restoration.tif'),
                'service_dspop': os.path.join(ROOT_DATA_DIR, 'service_dspop_flood_mitigation_PH_restoration.tif'),
                'service_road': os.path.join(ROOT_DATA_DIR, 'service_road_flood_mitigation_PH_restoration.tif'),
                'top_10th_percentile_service_dspop': os.path.join(ROOT_DATA_DIR, 'top_10th_percentile_service_dspop_flood_mitigation_PH_restoration.tif'),
                'top_10th_percentile_service_road': os.path.join(ROOT_DATA_DIR, 'top_10th_percentile_service_road_flood_mitigation_PH_restoration.tif'),
            },
            RECHARGE_SERVICE: {
                'diff': os.path.join(ROOT_DATA_DIR, 'diff_recharge_PH_restoration.tif'),
                'service_dspop': os.path.join(ROOT_DATA_DIR, 'service_dspop_recharge_PH_restoration.tif'),
                'top_10th_percentile_service_dspop': os.path.join(ROOT_DATA_DIR, 'top_10th_percentile_service_dspop_recharge_PH_restoration.tif'),
            },
            SEDIMENT_SERVICE: {
                'diff': os.path.join(ROOT_DATA_DIR, 'diff_sediment_PH_restoration.tif'),
                'service_dspop': os.path.join(ROOT_DATA_DIR, 'service_dspop_sediment_PH_restoration.tif'),
                'service_road': os.path.join(ROOT_DATA_DIR, 'service_road_sediment_PH_restoration.tif'),
                'top_10th_percentile_service_dspop': os.path.join(ROOT_DATA_DIR, 'top_10th_percentile_service_dspop_sediment_PH_restoration.tif'),
                'top_10th_percentile_service_road': os.path.join(ROOT_DATA_DIR, 'top_10th_percentile_service_road_sediment_PH_restoration.tif'),
            },
            CV_SERVICE: {
                'service_dspop': os.path.join(ROOT_DATA_DIR, 'service_dspop_cv_ph_restoration_result.tif'),
                'service_road': os.path.join(ROOT_DATA_DIR, 'service_road_cv_ph_restoration_result.tif'),
                'top_10th_percentile_service_dspop': os.path.join(ROOT_DATA_DIR, 'top_10th_percentile_service_dspop_cv_ph_restoration_result.tif'),
                'top_10th_percentile_service_road': os.path.join(ROOT_DATA_DIR, 'top_10th_percentile_service_road_cv_ph_restoration_result.tif'),
            },
        },
        CONSERVATION_SCENARIO_INF: {
            FLOOD_MITIGATION_SERVICE: {
                'diff': os.path.join(ROOT_DATA_DIR, 'diff_flood_mitigation_PH_conservation_inf.tif'),
                'service_dspop': os.path.join(ROOT_DATA_DIR, 'service_dspop_flood_mitigation_PH_conservation_inf.tif'),
                'service_road': os.path.join(ROOT_DATA_DIR, 'service_road_flood_mitigation_PH_conservation_inf.tif'),
                'top_10th_percentile_service_dspop': os.path.join(ROOT_DATA_DIR, 'top_10th_percentile_service_dspop_flood_mitigation_PH_conservation_inf.tif'),
                'top_10th_percentile_service_road': os.path.join(ROOT_DATA_DIR, 'top_10th_percentile_service_road_flood_mitigation_PH_conservation_inf.tif'),
            },
            RECHARGE_SERVICE: {
                'diff': os.path.join(ROOT_DATA_DIR, 'diff_recharge_PH_conservation_inf.tif'),
                'service_dspop': os.path.join(ROOT_DATA_DIR, 'service_dspop_recharge_PH_conservation_inf.tif'),
                'top_10th_percentile_service_dspop': os.path.join(ROOT_DATA_DIR, 'top_10th_percentile_service_dspop_recharge_PH_conservation_inf.tif'),
            },
            SEDIMENT_SERVICE: {
                'diff': os.path.join(ROOT_DATA_DIR, 'diff_sediment_PH_conservation_inf.tif'),
                'service_dspop': os.path.join(ROOT_DATA_DIR, 'service_dspop_sediment_PH_conservation_inf.tif'),
                'service_road': os.path.join(ROOT_DATA_DIR, 'service_road_sediment_PH_conservation_inf.tif'),
                'top_10th_percentile_service_road': os.path.join(ROOT_DATA_DIR, 'top_10th_percentile_service_road_sediment_PH_conservation_inf.tif'),
                'top_10th_percentile_service_dspop': os.path.join(ROOT_DATA_DIR, 'top_10th_percentile_service_dspop_sediment_PH_conservation_inf.tif'),
            },
            CV_SERVICE: {
                'top_10th_percentile_service_dspop': os.path.join(ROOT_DATA_DIR, 'top_10th_percentile_service_dspop_cv_ph_conservation_inf_result.tif'),
                'service_dspop': os.path.join(ROOT_DATA_DIR, 'service_dspop_cv_ph_conservation_inf_result.tif'),
                'service_road': os.path.join(ROOT_DATA_DIR, 'service_road_cv_ph_conservation_inf_result.tif'),
                'top_10th_percentile_service_road': os.path.join(ROOT_DATA_DIR, 'top_10th_percentile_service_road_cv_ph_conservation_inf_result.tif'),
            },
        },
        CONSERVATION_SCENARIO_ALL: {
            FLOOD_MITIGATION_SERVICE: {
                'diff': os.path.join(ROOT_DATA_DIR, 'diff_flood_mitigation_PH_conservation_all.tif'),
                'service_dspop': os.path.join(ROOT_DATA_DIR, 'service_dspop_flood_mitigation_PH_conservation_all.tif'),
                'service_road': os.path.join(ROOT_DATA_DIR, 'service_road_flood_mitigation_PH_conservation_all.tif'),
                'top_10th_percentile_service_dspop': os.path.join(ROOT_DATA_DIR, 'top_10th_percentile_service_dspop_flood_mitigation_PH_conservation_all.tif'),
                'top_10th_percentile_service_road': os.path.join(ROOT_DATA_DIR, 'top_10th_percentile_service_road_flood_mitigation_PH_conservation_all.tif'),
            },
            RECHARGE_SERVICE: {
                'diff': os.path.join(ROOT_DATA_DIR, 'diff_recharge_PH_conservation_all.tif'),
                'service_dspop': os.path.join(ROOT_DATA_DIR, 'service_dspop_recharge_PH_conservation_all.tif'),
                'top_10th_percentile_service_dspop': os.path.join(ROOT_DATA_DIR, 'top_10th_percentile_service_dspop_recharge_PH_conservation_all.tif'),
            },
            SEDIMENT_SERVICE: {
                'diff': os.path.join(ROOT_DATA_DIR, 'diff_sediment_PH_conservation_all.tif'),
                'service_dspop': os.path.join(ROOT_DATA_DIR, 'service_dspop_sediment_PH_conservation_all.tif'),
                'service_road': os.path.join(ROOT_DATA_DIR, 'service_road_sediment_PH_conservation_all.tif'),
                'top_10th_percentile_service_road': os.path.join(ROOT_DATA_DIR, 'top_10th_percentile_service_road_sediment_PH_conservation_all.tif'),
                'top_10th_percentile_service_dspop': os.path.join(ROOT_DATA_DIR, 'top_10th_percentile_service_dspop_sediment_PH_conservation_all.tif'),
            },
            CV_SERVICE: {
                # note that conservation for CV is the same as "inf" so we didn't run an additional scenario on it, just using the INF version here
                'top_10th_percentile_service_dspop': os.path.join(ROOT_DATA_DIR, 'top_10th_percentile_service_dspop_cv_ph_conservation_inf_result.tif'),
                'service_dspop': os.path.join(ROOT_DATA_DIR, 'service_dspop_cv_ph_conservation_inf_result.tif'),
                'service_road': os.path.join(ROOT_DATA_DIR, 'service_road_cv_ph_conservation_inf_result.tif'),
                'top_10th_percentile_service_road': os.path.join(ROOT_DATA_DIR, 'top_10th_percentile_service_road_cv_ph_conservation_inf_result.tif'),
            },
        },
    },
    'IDN': {
        RESTORATION_SCENARIO: {
            FLOOD_MITIGATION_SERVICE: {
                'diff': os.path.join(ROOT_DATA_DIR, 'diff_flood_mitigation_IDN_restoration.tif'),
                'service_dspop': os.path.join(ROOT_DATA_DIR, 'service_dspop_flood_mitigation_IDN_restoration.tif'),
                'service_road': os.path.join(ROOT_DATA_DIR, 'service_road_flood_mitigation_IDN_restoration.tif'),
                'top_10th_percentile_service_dspop': os.path.join(ROOT_DATA_DIR, 'top_10th_percentile_service_dspop_flood_mitigation_IDN_restoration.tif'),
                'top_10th_percentile_service_road': os.path.join(ROOT_DATA_DIR, 'top_10th_percentile_service_road_flood_mitigation_IDN_restoration.tif'),
            },
            RECHARGE_SERVICE: {
                'diff': os.path.join(ROOT_DATA_DIR, 'diff_recharge_IDN_restoration.tif'),
                'service_dspop': os.path.join(ROOT_DATA_DIR, 'service_dspop_recharge_IDN_restoration.tif'),
                'top_10th_percentile_service_dspop': os.path.join(ROOT_DATA_DIR, 'top_10th_percentile_service_dspop_recharge_IDN_restoration.tif'),
            },
            SEDIMENT_SERVICE: {
                'diff': os.path.join(ROOT_DATA_DIR, 'diff_sediment_IDN_restoration.tif'),
                'service_dspop': os.path.join(ROOT_DATA_DIR, 'service_dspop_sediment_IDN_restoration.tif'),
                'service_road': os.path.join(ROOT_DATA_DIR, 'service_road_sediment_IDN_restoration.tif'),
                'top_10th_percentile_service_dspop': os.path.join(ROOT_DATA_DIR, 'top_10th_percentile_service_dspop_sediment_IDN_restoration.tif'),
                'top_10th_percentile_service_road': os.path.join(ROOT_DATA_DIR, 'top_10th_percentile_service_road_sediment_IDN_restoration.tif'),
            },
            CV_SERVICE: {
                'service_dspop': os.path.join(ROOT_DATA_DIR, 'service_dspop_cv_idn_restoration_result.tif'),
                'service_road': os.path.join(ROOT_DATA_DIR, 'service_road_cv_idn_restoration_result.tif'),
                'top_10th_percentile_service_dspop': os.path.join(ROOT_DATA_DIR, 'top_10th_percentile_service_dspop_cv_idn_restoration_result.tif'),
                'top_10th_percentile_service_road': os.path.join(ROOT_DATA_DIR, 'top_10th_percentile_service_road_cv_idn_restoration_result.tif'),
            },
        },
        CONSERVATION_SCENARIO_INF: {
            FLOOD_MITIGATION_SERVICE: {
                'diff': os.path.join(ROOT_DATA_DIR, 'diff_flood_mitigation_IDN_conservation_inf.tif'),
                'service_dspop': os.path.join(ROOT_DATA_DIR, 'service_dspop_flood_mitigation_IDN_conservation_inf.tif'),
                'service_road': os.path.join(ROOT_DATA_DIR, 'service_road_flood_mitigation_IDN_conservation_inf.tif'),
                'top_10th_percentile_service_dspop': os.path.join(ROOT_DATA_DIR, 'top_10th_percentile_service_dspop_flood_mitigation_IDN_conservation_inf.tif'),
                'top_10th_percentile_service_road': os.path.join(ROOT_DATA_DIR, 'top_10th_percentile_service_road_flood_mitigation_IDN_conservation_inf.tif'),
            },
            RECHARGE_SERVICE: {
                'diff': os.path.join(ROOT_DATA_DIR, 'diff_recharge_IDN_conservation_inf.tif'),
                'service_dspop': os.path.join(ROOT_DATA_DIR, 'service_dspop_recharge_IDN_conservation_inf.tif'),
                'top_10th_percentile_service_dspop': os.path.join(ROOT_DATA_DIR, 'top_10th_percentile_service_dspop_recharge_IDN_conservation_inf.tif'),
            },
            SEDIMENT_SERVICE: {
                'diff': os.path.join(ROOT_DATA_DIR, 'diff_sediment_IDN_conservation_inf.tif'),
                'service_dspop': os.path.join(ROOT_DATA_DIR, 'service_dspop_sediment_IDN_conservation_inf.tif'),
                'service_road': os.path.join(ROOT_DATA_DIR, 'service_road_sediment_IDN_conservation_inf.tif'),
                'top_10th_percentile_service_road': os.path.join(ROOT_DATA_DIR, 'top_10th_percentile_service_road_sediment_IDN_conservation_inf.tif'),
                'top_10th_percentile_service_dspop': os.path.join(ROOT_DATA_DIR, 'top_10th_percentile_service_dspop_sediment_IDN_conservation_inf.tif'),
            },
            CV_SERVICE: {
                'top_10th_percentile_service_dspop': os.path.join(ROOT_DATA_DIR, 'top_10th_percentile_service_dspop_cv_idn_conservation_inf_result.tif'),
                'service_dspop': os.path.join(ROOT_DATA_DIR, 'service_dspop_cv_idn_conservation_inf_result.tif'),
                'service_road': os.path.join(ROOT_DATA_DIR, 'service_road_cv_idn_conservation_inf_result.tif'),
                'top_10th_percentile_service_road': os.path.join(ROOT_DATA_DIR, 'top_10th_percentile_service_road_cv_idn_conservation_inf_result.tif'),
            },
        },
        CONSERVATION_SCENARIO_ALL: {
            FLOOD_MITIGATION_SERVICE: {
                'diff': os.path.join(ROOT_DATA_DIR, 'diff_flood_mitigation_IDN_conservation_all.tif'),
                'service_dspop': os.path.join(ROOT_DATA_DIR, 'service_dspop_flood_mitigation_IDN_conservation_all.tif'),
                'service_road': os.path.join(ROOT_DATA_DIR, 'service_road_flood_mitigation_IDN_conservation_all.tif'),
                'top_10th_percentile_service_dspop': os.path.join(ROOT_DATA_DIR, 'top_10th_percentile_service_dspop_flood_mitigation_IDN_conservation_all.tif'),
                'top_10th_percentile_service_road': os.path.join(ROOT_DATA_DIR, 'top_10th_percentile_service_road_flood_mitigation_IDN_conservation_all.tif'),
            },
            RECHARGE_SERVICE: {
                'diff': os.path.join(ROOT_DATA_DIR, 'diff_recharge_IDN_conservation_all.tif'),
                'service_dspop': os.path.join(ROOT_DATA_DIR, 'service_dspop_recharge_IDN_conservation_all.tif'),
                'top_10th_percentile_service_dspop': os.path.join(ROOT_DATA_DIR, 'top_10th_percentile_service_dspop_recharge_IDN_conservation_all.tif'),
            },
            SEDIMENT_SERVICE: {
                'diff': os.path.join(ROOT_DATA_DIR, 'diff_sediment_IDN_conservation_all.tif'),
                'service_dspop': os.path.join(ROOT_DATA_DIR, 'service_dspop_sediment_IDN_conservation_all.tif'),
                'service_road': os.path.join(ROOT_DATA_DIR, 'service_road_sediment_IDN_conservation_all.tif'),
                'top_10th_percentile_service_road': os.path.join(ROOT_DATA_DIR, 'top_10th_percentile_service_road_sediment_IDN_conservation_all.tif'),
                'top_10th_percentile_service_dspop': os.path.join(ROOT_DATA_DIR, 'top_10th_percentile_service_dspop_sediment_IDN_conservation_all.tif'),
            },
            CV_SERVICE: {
                # note that conservation for CV is the same as "inf" so we didn't run an additional scenario on it, just using the INF version here
                'top_10th_percentile_service_dspop': os.path.join(ROOT_DATA_DIR, 'top_10th_percentile_service_dspop_cv_idn_conservation_inf_result.tif'),
                'service_dspop': os.path.join(ROOT_DATA_DIR, 'service_dspop_cv_idn_conservation_inf_result.tif'),
                'service_road': os.path.join(ROOT_DATA_DIR, 'service_road_cv_idn_conservation_inf_result.tif'),
                'top_10th_percentile_service_road': os.path.join(ROOT_DATA_DIR, 'top_10th_percentile_service_road_cv_idn_conservation_inf_result.tif'),
            },
        },
    }
}


def calculate_pa_kba_overlaps(
    task_graph,
    service_raster_path,
    basename_raster_path,
    country,
    scenario,
    service,
    projection_epsg
):
    service_on_protected_areas_path = os.path.join(
        PA_KBA_OVERLAP_DIR,
        f'pa_{os.path.basename(basename_raster_path)}')

    pa_mask_task = task_graph.add_task(
        func=geoprocessing.mask_raster,
        args=(
            (service_raster_path, 1),
            PROTECTED_AREAS[country],
            service_on_protected_areas_path),
        kwargs={
            'working_dir': WORKING_DIR,
            'all_touched': True,
            'allow_different_blocksize': True},
        target_path_list=[service_on_protected_areas_path],
        task_name=f'pa overlap for {country}_{scenario}_{service}')

    pa_area_task = task_graph.add_task(
        func=calculate_pixel_area_km2,
        args=(service_on_protected_areas_path, projection_epsg),
        dependent_task_list=[pa_mask_task],
        store_result=True,
        task_name=f'pixel area km for {country}_{scenario}_{service}')

    service_on_kba_path = os.path.join(
        PA_KBA_OVERLAP_DIR,
        f'kba_{os.path.basename(basename_raster_path)}')
    kba_mask_task = task_graph.add_task(
        func=geoprocessing.mask_raster,
        args=(
            (service_raster_path, 1),
            KEY_BIODIVERSITY_AREAS[country],
            service_on_kba_path),
        kwargs={
            'working_dir': WORKING_DIR,
            'all_touched': True,
            'allow_different_blocksize': True},
        target_path_list=[service_on_kba_path],
        task_name=f'kba overlap for {country}_{scenario}_{service}')

    kba_area_task = task_graph.add_task(
        func=calculate_pixel_area_km2,
        args=(service_on_kba_path, projection_epsg),
        dependent_task_list=[kba_mask_task],
        store_result=True,
        task_name=f'pixel area km for {country}_{scenario}_{service}')

    return pa_area_task, kba_area_task


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


def adjust_font_size(ax, fig, base_size):
    fig_width, fig_height = fig.get_size_inches()
    mean_dim = (fig_width + fig_height) / 2

    # Scale font size based on figure dimensions
    scaled_font_size = base_size * (mean_dim / 10)  # 10 is a normalization factor
    ax.title.set_fontsize(scaled_font_size)


def adjust_suptitle_fontsize(fig, base_size):
    fig_width, fig_height = fig.get_size_inches()
    mean_dim = (fig_width + fig_height) / 2

    # Scale font size based on figure dimensions
    scaled_font_size = base_size * (mean_dim / 10)  # Normalize based on an arbitrary value
    return scaled_font_size


def print_colormap_colors(cmap, num_samples):
    # Generate evenly spaced numbers between 0 and 1 to sample the colormap
    sample_points = np.linspace(0, 1, num_samples+1)[:1]  # skipping the nodata color
    colors = cmap(sample_points)  # Sample colors from the colormap

    # Print each color as RGBA and as hexadecimal
    for i, color in enumerate(colors):
        # Normalize and convert RGBA color to hexadecimal, handling transparency
        rgba_normalized = np.clip(color[:3], 0, 1)  # Ensure RGB values are within [0, 1]
        hex_color = mcolors.rgb2hex(rgba_normalized)
        print(f"Color {sample_points[i]} {i+1}: RGBA={color}, Hex={hex_color}")


def overlap_colormap(version):
    # Define the colors - these could be any valid matplotlib color.
    colors = COLOR_LIST[version]
    cmap = LinearSegmentedColormap.from_list("overlap_colors", colors, N=len(colors))
    return cmap


def interpolated_colormap(cmap):
    return plt.get_cmap(cmap)


def read_raster_csv(file_path):
    raster_dict = {}
    with open(file_path, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        current_raster_name = ""
        for i, row in enumerate(reader):
            if i % 4 == 0:
                current_raster_name = row[0]
                raster_dict[current_raster_name] = {}
            elif i % 4 == 1:
                raster_dict[current_raster_name]['position'] = [float(x) for x in row[1:] if x != '']
            elif i % 4 == 2:
                raster_dict[current_raster_name]['color'] = [x for x in row[1:] if x != '']
            elif i % 4 == 3:
                raster_dict[current_raster_name]['transparency'] = [int(x) for x in row[1:] if x != '']
    return raster_dict


def hex_to_rgba(hex_code, transparency):
    hex_code = hex_code.lstrip('#')
    rgb = tuple(int(hex_code[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
    alpha = transparency / 100  # Convert 0-100 scale to 0-1 scale
    return rgb + (alpha,)


CUSTOM_STYLES = {}
for style_file_path in glob.glob(os.path.join(CUSTOM_STYLE_DIR, '*.csv')):
    CUSTOM_STYLES.update(read_raster_csv(style_file_path))


def root_filename(path):
    return os.path.splitext(os.path.basename(path))[0]


def calculate_figsize(aspect_ratio, grid_size, subplot_size):
    rows, cols = grid_size
    subplot_width, subplot_height = subplot_size
    subplot_width = subplot_height * aspect_ratio

    total_width = cols * subplot_width
    total_height = rows * subplot_height

    return (total_width, total_height)


def get_percentiles(base_raster_path, min_percentile, max_percentile):
    base_raster = gdal.Open(base_raster_path, gdal.GA_ReadOnly)
    width = base_raster.RasterXSize // 4
    height = base_raster.RasterYSize // 4

    warp_options = gdal.WarpOptions(
        width=width,
        height=height,
        resampleAlg=gdal.GRA_NearestNeighbour
    )

    with tempfile.NamedTemporaryFile(delete=False, suffix='.tif', dir='.') as tmp_file:
        tmp_file.close()
        output_raster_path = tmp_file.name
        gdal.Warp(output_raster_path, base_raster, options=warp_options)
        base_raster = None

        warped_raster = gdal.OpenEx(output_raster_path)
        band = warped_raster.GetRasterBand(1)
        nodata = band.GetNoDataValue()
        base_array = band.ReadAsArray()
        nodata_mask = ((base_array == nodata) | np.isnan(base_array)) | (base_array == 0)
        valid_base_array = base_array[~nodata_mask]
        base_array = None
        nodata_mask = None
        base_min = np.percentile(valid_base_array, min_percentile)
        base_max = np.percentile(valid_base_array, max_percentile)
        valid_base_array = None
        band = None
        warped_raster = None
        os.remove(output_raster_path)

    return base_min, base_max


def cogit(file_path, target_dir):
    # create copy with COG
    os.makedirs(target_dir, exist_ok=True)
    cog_driver = gdal.GetDriverByName('COG')
    base_raster = gdal.OpenEx(file_path, gdal.OF_RASTER)
    cog_file_path = os.path.join(
        target_dir,
        f'cog_{os.path.basename(file_path)}')
    options = ('COMPRESS=LZW', 'NUM_THREADS=ALL_CPUS', 'BIGTIFF=YES', 'RESAMPLING=mode')
    LOGGER.info(f'convert {file_path} to COG {cog_file_path} with {options}')
    cog_raster = cog_driver.CreateCopy(
        cog_file_path, base_raster, options=options,
        callback=_make_logger_callback(
            f"COGing {cog_file_path} %.1f%% complete %s"))
    del cog_raster
    return cog_file_path


def style_rasters(
        country_outline_vector_path, country_id, raster_paths, category_list,
        stack_vertical, color_map_or_list, color_palette_list,
        percentile_or_categorical_list,
        base_min_max_list, fig_size, fig_path, overall_title,
        subfigure_title_list, dpi, task_graph, pixel_coarsen_factor=1):

    with open(RASTER_STYLE_LOG_PATH, 'a') as file:
        for (raster_path, color_palette, category_labels,
             percentile_or_categorical, base_min_max, subfigure_title) in zip(
                raster_paths, color_palette_list, category_list,
                percentile_or_categorical_list, base_min_max_list, subfigure_title_list):
            if subfigure_title is None:
                continue

            cog_task = task_graph.add_task(
                func=cogit,
                args=(raster_path, COG_DIR),
                store_result=True,
                task_name=f'cog {raster_path}')
            if percentile_or_categorical == 'categorical':
                base_min = 0
                base_max = len(category_labels)
            elif base_min_max is not None:
                base_min, base_max = base_min_max
            else:
                min_percentile, max_percentile = percentile_or_categorical
                LOGGER.debug(f'loading {raster_path}')
                percentile_task = task_graph.add_task(
                    func=get_percentiles,
                    args=(raster_path, min_percentile, max_percentile),
                    store_result=True,
                    task_name=f'percentiles for {raster_path}')
                base_min, base_max = percentile_task.get()
            cog_path = os.path.join(
                REMOTE_BUCKET_PATH, os.path.basename(cog_task.get()))

            key = f'{country_id}_{overall_title}_{subfigure_title}'
            file.write(f'\t"{key}"'+': {\n')
            file.write('\t\tfig_title:"' + f'{overall_title}, {subfigure_title}' + '",\n')
            file.write(f'\t\tcountry:"{country_id}",\n')
            file.write('\t\tlabels:[' + ','.join([f'"{x}"' for x in category_labels]) + '],\n')
            file.write('\t\tvisParams:{\n')
            file.write(f'\t\t\tmin:{base_min},\n')
            file.write(f'\t\t\tmax:{base_max},\n')
            file.write('\t\t\tpalette:[' + ','.join([f'"{x.strip("#")}"' for x in color_palette[1:]]) + '],\n')
            file.write('\t\t},\n')
            file.write(f'\t\tremote_path: "{cog_path}",' + '\n\t},\n')
            file.flush()

    if fig_path is None:
        # we aren't rendering
        return

    outline_gdf = geopandas.read_file(country_outline_vector_path)

    if not isinstance(color_map_or_list, list):
        colormap_list = [color_map_or_list] * len(raster_paths)
    else:
        colormap_list = color_map_or_list
    single_raster_mode = len(raster_paths) == 1
    for path in raster_paths:
        if path is None:
            continue
        raster_info = geoprocessing.get_raster_info(path)
        aspect_ratio = raster_info['raster_size'][0] / raster_info['raster_size'][1]
        break

    if single_raster_mode:
        rows = 1
        columns = 1
    elif stack_vertical:
        rows = 4
        columns = 1
    else:
        rows = 2
        columns = 2

    fig_width, fig_height = calculate_figsize(
        aspect_ratio, (rows, columns), (fig_size, fig_size))

    fig, axs = plt.subplots(rows, columns, figsize=(fig_width, fig_height))  # Set up a 2x2 grid of plots
    n_pixels = fig_width*dpi
    if not single_raster_mode:
        axs = axs.flatten()  # Flatten the 2D array of axes for easier iteration
    else:
        axs = [axs]

    for idx, (base_raster_path, categories, color_map, percentile_or_categorical, base_min_max) in enumerate(zip(raster_paths, category_list, colormap_list, percentile_or_categorical_list, base_min_max_list)):
        if base_raster_path is None:
            axs[idx].axis('off')
            continue
        raster_info = geoprocessing.get_raster_info(base_raster_path)
        target_pixel_size = scale_pixel_size(
            raster_info['raster_size'], n_pixels/pixel_coarsen_factor,
            raster_info['pixel_size'])

        LOGGER.info('skipping analyses')

        scaled_path = os.path.join(
            SCALED_DIR,
            f'scaled_for_fig_{n_pixels}_{pixel_coarsen_factor}_{os.path.basename(base_raster_path)}')

        LOGGER.info(f'scaling {scaled_path}')
        warp_task = task_graph.add_task(
            func=geoprocessing.warp_raster,
            args=(
                base_raster_path, target_pixel_size, scaled_path,
                'mode'),
            target_path_list=[scaled_path],
            task_name=f'scaling {scaled_path} in fig generator')
        warp_task.join()
        LOGGER.info('scaled!')

        base_raster = gdal.OpenEx(scaled_path, gdal.OF_RASTER)
        base_array = base_raster.GetRasterBand(1).ReadAsArray()
        gt = base_raster.GetGeoTransform()
        xmin = gt[0]
        xmax = xmin + (gt[1] * base_raster.RasterXSize)
        ymax = gt[3]
        ymin = ymax + (gt[5] * base_raster.RasterYSize)
        extent = [xmin, xmax, ymin, ymax]

        # Create a color gradient
        color_map = interpolated_colormap(color_map)
        no_data_color = [0, 0, 0, 0]  # Assuming a black NoData color with full transparency

        nodata = geoprocessing.get_raster_info(scaled_path)['nodata'][0]
        nodata_mask = ((base_array == nodata) | np.isnan(base_array)) | (base_array == 0)
        styled_array = np.empty(base_array.shape + (4,), dtype=float)
        valid_base_array = base_array[~nodata_mask]
        if percentile_or_categorical == 'categorical':
            base_min = 0
            base_max = len(categories)
        elif base_min_max is not None:
            base_min, base_max = base_min_max
        else:
            min_percentile, max_percentile = percentile_or_categorical
            base_min = np.percentile(valid_base_array, min_percentile)
            base_max = np.percentile(valid_base_array, max_percentile)
        norm = mcolors.Normalize(vmin=base_min, vmax=base_max)
        sm = cm.ScalarMappable(cmap=color_map, norm=norm)
        styled_array[~nodata_mask] = sm.to_rgba(valid_base_array)

        styled_array[nodata_mask] = no_data_color

        subfigure_title = subfigure_title_list[idx]
        if subfigure_title is not None:
            axs[idx].set_title(subfigure_title_list[idx], wrap=True)
            adjust_font_size(axs[idx], fig, BASE_FONT_SIZE)
        axs[idx].imshow(styled_array, origin='upper', extent=extent)
        axs[idx].axis('off')  # Turn off axis labels
        if categories is not None:
            # skipping the nodata color
            values = numpy.linspace(0, 1, len(categories)+1)[1:]
            print_colormap_colors(color_map, len(categories))
            colors = [color_map(value) for value in values]
            fig_width, fig_height = fig.get_size_inches()
            patches = [mpatches.Patch(color=colors[i], label=categories[i]) for i in range(len(values))]
            axs[idx].legend(
                fontsize=adjust_suptitle_fontsize(fig, BASE_FONT_SIZE * 0.75),
                handles=patches,
                loc='upper right')
        outline_gdf.boundary.plot(ax=axs[idx], color='black', linewidth=0.5)

    fontsize_for_suptitle = adjust_suptitle_fontsize(
        fig, BASE_FONT_SIZE)
    fig.suptitle(overall_title, fontsize=fontsize_for_suptitle)

    adjust_font_size(axs[idx], fig, BASE_FONT_SIZE)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(fig_path, dpi=dpi)
    plt.close(fig)


def intersection_op(raster_a_path, raster_b_path, target_path):
    def _and(array_a, array_b):
        return (array_a > 0) & (array_b > 0)

    aligned_rasters = [
        os.path.join(ALGINED_DIR, f'aligned_{os.path.basename(path)}')
        for path in [raster_a_path, raster_b_path]]
    geoprocessing.align_and_resize_raster_stack(
        [raster_a_path, raster_b_path], aligned_rasters, ['near']*2,
        GLOBAL_PIXEL_SIZE, 'intersection')
    geoprocessing.single_thread_raster_calculator(
        [(path, 1) for path in aligned_rasters], _and, target_path,
        gdal.GDT_Int16, 0, allow_different_blocksize=True)


def overlap_dspop_road_op(raster_a_path, raster_b_path, unique_prefix, target_path):
    def _overlap_dspop_road_op(array_a, array_b):
        result = (array_a > 0) + (2 * (array_b > 0))
        return result

    aligned_rasters = [
        os.path.join(ALGINED_DIR, f'aligned_{unique_prefix}_{os.path.basename(path)}')
        for path in [raster_a_path, raster_b_path]]
    LOGGER.debug(f'for {raster_a_path} does it exist: {os.path.exists(raster_a_path)}')
    geoprocessing.align_and_resize_raster_stack(
        [raster_a_path, raster_b_path], aligned_rasters, ['near']*2,
        GLOBAL_PIXEL_SIZE, 'intersection')
    geoprocessing.single_thread_raster_calculator(
        [(path, 1) for path in aligned_rasters], _overlap_dspop_road_op, target_path,
        gdal.GDT_Int16, 0, allow_different_blocksize=True)


def list_to_unique_and_index(flat_path_list):
    unique_dict = {}
    index_list = []
    unique_list = []

    for path in flat_path_list:
        if path not in unique_dict:
            unique_dict[path] = len(unique_list)
            unique_list.append(path)
        index_list.append(unique_dict[path])
    return unique_list, index_list


def overlap_combos_op(task_graph, overlap_combo_list, prefix, target_path):
    """
        overlap_combo_list - [(required raster list), (optional raster lsit),
            threshold for how many optional needed]
    """
    flat_path_list = [
        path
        for required_path_list, optional_path_list, _, _ in overlap_combo_list
        for path_list in [required_path_list, optional_path_list]
        for index, path in enumerate(path_list)
    ]

    unique_path_list, flat_path_to_unique_list = list_to_unique_and_index(
        flat_path_list)
    aligned_rasters = [
        os.path.join(
            ALGINED_DIR,
            f'aligned_{prefix}_{os.path.basename(path)}')
        for path in unique_path_list]

    aligned_path_band_list = [
        (aligned_rasters[index], 1)
        for index in flat_path_to_unique_list]

    def _overlap_combos_op(index_list, *array_list):
        '''index_list = [(index_for_next_optional, index_fo_next_required, threshold)]'''
        result = numpy.zeros(array_list[0].shape, dtype=int)
        service_index = 1
        local_index_list = index_list.copy()
        next_optional_index, next_required_index, overlap_threshold, comparitor_op = (
            local_index_list.pop(0))
        local_overlap = numpy.zeros(result.shape, dtype=int)
        required_valid_overlap = numpy.ones(result.shape, dtype=bool)
        local_max_overlap = 0
        for array_index, (array, array_path) in enumerate(zip(array_list, aligned_path_band_list)):
            if (array_index < next_optional_index):
                local_overlap_mask = (array > 0)
                required_valid_overlap &= local_overlap_mask
            else:
                # we're in optional mask territory
                if array_index == next_required_index:
                    # process this next
                    result[
                        required_valid_overlap &
                        (comparitor_op(local_overlap, overlap_threshold))] = service_index
                    local_overlap[:] = 0
                    local_max_overlap = 0
                    service_index += 1
                    next_optional_index, next_required_index, overlap_threshold, comparitor_op = (
                        local_index_list.pop(0))
                    if (array_index < next_optional_index):
                        local_overlap_mask = (array > 0)
                        required_valid_overlap = local_overlap_mask
                        continue
                    else:
                        required_valid_overlap[:] = True
                valid_mask = array > 0
                local_overlap[valid_mask] += 1
                local_max_overlap += 1
        result[
            required_valid_overlap &
            (local_overlap >= overlap_threshold)] = service_index
        return result

    index_list = [(0, 0, 0)]  # just to initialize, it's dropped later
    next_offset = 0
    for required_path_list, optional_path_list, overlap_threshold, comparitor_op in overlap_combo_list:
        index_list.append((
            next_offset+len(required_path_list),
            next_offset+len(required_path_list)+len(optional_path_list),
            overlap_threshold,
            comparitor_op))
        next_offset = index_list[-1][1]
    index_list.pop(0)

    # calculate if any of the target paths are missing
    if not all([os.path.exists(path) for path in aligned_rasters]):
        task_graph.add_task(
            func=geoprocessing.align_and_resize_raster_stack,
            args=(
                unique_path_list, aligned_rasters,
                ['near'] * len(unique_path_list),
                GLOBAL_PIXEL_SIZE, 'intersection'),
            target_path_list=aligned_rasters,
            task_name='alignining in overlap op')
    task_graph.join()

    combined_index_raster_path_list = (
        [(index_list, 'raw')] + aligned_path_band_list)
    geoprocessing.single_thread_raster_calculator(
        combined_index_raster_path_list, _overlap_combos_op,
        target_path, gdal.GDT_Int16, None, allow_different_blocksize=True)


def scale_pixel_size(dimensions, n_pixels, pixel_size):
    x, y = dimensions

    # Determine the scaling factor based on the larger dimension
    if x > y and x > n_pixels:
        scale_factor = x / n_pixels
    elif y > x and y > n_pixels:
        scale_factor = y / n_pixels
    else:
        scale_factor = 1

    return (pixel_size[0]*scale_factor, pixel_size[1]*scale_factor)


def subtract_paths(base_path, full_path):
    base = Path(base_path).resolve()
    full = Path(full_path).resolve()
    return full.relative_to(base)


def do_analyses(task_graph, processed_raster_path_set):
    scenario_service_tuples = list(itertools.product(
        [CONSERVATION_SCENARIO_ALL, CONSERVATION_SCENARIO_INF, RESTORATION_SCENARIO],
        [SEDIMENT_SERVICE, FLOOD_MITIGATION_SERVICE,
         RECHARGE_SERVICE, CV_SERVICE]))

    result_df = pandas.DataFrame()

    for country, vector_path, projection_epsg in [
            ('PH', COUNTRY_OUTLINE_PATH['PH'], ELLIPSOID_EPSG),
            ('IDN', COUNTRY_OUTLINE_PATH['IDN'], ELLIPSOID_EPSG)]:
        # (2) Total area of the country
        country_area_km2 = calculate_vector_area_km2(vector_path, projection_epsg)

        # How much do services overlap with each other? Which one overlaps the least?
        # Need to know (for each scenario, conservation and restoration; for each country)
        # Each service:

        # I: (4*2*2) Total area of each service’s top_10th_percentile_service_dspop or _road
        for scenario, service in scenario_service_tuples:
            # loop through all the services, they always have a dspop and some of them have a roads, if roads then combine
            dspop_road_overlap_path = os.path.join(OVERLAP_DIR, f'dspop_road_overlap_{country}_{scenario}_{service}.tif')
            top_10th_percentile_service_dspop_path = FILENAMES[country][scenario][service]['top_10th_percentile_service_dspop']
            dspop_road_id = 'dspop/road'
            if 'top_10th_percentile_service_road' in FILENAMES[country][scenario][service]:
                # combine road and dspop if road exists
                top_10th_percentile_service_road_path = FILENAMES[country][scenario][service]['top_10th_percentile_service_road']
                if dspop_road_overlap_path not in processed_raster_path_set:
                    task_graph.add_task(
                        func=overlap_dspop_road_op,
                        args=(
                            top_10th_percentile_service_dspop_path,
                            top_10th_percentile_service_road_path,
                            f'top10_{country}_{scenario}',
                            dspop_road_overlap_path),
                        target_path_list=[dspop_road_overlap_path],
                        task_name=f'dspop road {service} {country} {scenario}')
                    processed_raster_path_set.add(dspop_road_overlap_path)
            else:
                # doesn't exist but we don't lose anything by just doing the dspop
                dspop_road_overlap_path = top_10th_percentile_service_dspop_path
                dspop_road_id = 'dspop'

            pa_overlap_task, kba_overlap_task = calculate_pa_kba_overlaps(
                task_graph,
                top_10th_percentile_service_road_path,
                dspop_road_overlap_path,
                country,
                scenario,
                service,
                projection_epsg
            )

            service_area_km2_task = task_graph.add_task(
                func=calculate_pixel_area_km2,
                args=(dspop_road_overlap_path, projection_epsg),
                store_result=True,
                task_name=f'calculate pixel area for {projection_epsg}')
            service_area_km2 = service_area_km2_task.get()
            row_data = {
                'country': country,
                'country area km^2': country_area_km2,
                'scenario': scenario,
                'service': service,
                'summary': f'top 10th percentile {dspop_road_id} km^2',
                'value': service_area_km2,
                'source_file': dspop_road_overlap_path,
                'top 10th percentile service km^2 ON PROTECTED AREAS': pa_overlap_task.get(),
                'top 10th percentile service km^2 km^2 ON KBAS': kba_overlap_task.get(),
            }
            row_df = pandas.DataFrame([row_data])
            result_df = pandas.concat([result_df, row_df], ignore_index=True)

        # This is panel 4 on the 4 panel displays right now the ugly colored one (combined beneficiary map, where those maps = non-zero)
        # E.g., cv_IDN_conservation_inf
        # II: (4*2*2) Area where each service’s beneficiaries overlap (subset of the above)
            if 'top_10th_percentile_service_road' in FILENAMES[country][scenario][service]:
                combined_percentile_service_path = os.path.join(
                    COMBINED_SERVICE_DIR, f'combined_percentile_service_{service}_{country}_{scenario}.tif')
                if combined_percentile_service_path not in processed_raster_path_set:
                    task_graph.add_task(
                        func=intersection_op,
                        args=(
                            top_10th_percentile_service_dspop_path,
                            top_10th_percentile_service_road_path,
                            combined_percentile_service_path),
                        target_path_list=[combined_percentile_service_path],
                        task_name=f'combined service {service} {country} {scenario}')
                    processed_raster_path_set.add(combined_percentile_service_path)

                pa_overlap_task, kba_overlap_task = calculate_pa_kba_overlaps(
                    task_graph,
                    combined_percentile_service_path,
                    combined_percentile_service_path,
                    country,
                    scenario,
                    service,
                    projection_epsg
                )

                service_area_km2_task = task_graph.add_task(
                    func=calculate_pixel_area_km2,
                    args=(combined_percentile_service_path, projection_epsg),
                    store_result=True,
                    task_name=f'calculate pixel area for {combined_percentile_service_path}')
                service_area_km2 = service_area_km2_task.get()
                row_data = {
                    'country': country,
                    'country area km^2': country_area_km2,
                    'scenario': scenario,
                    'service': service,
                    'summary': 'top 10th percentile combined beneficiaries km^2',
                    'value': service_area_km2,
                    'source_file': combined_percentile_service_path,
                    'top 10th percentile service km^2 ON PROTECTED AREAS': pa_overlap_task.get(),
                    'top 10th percentile service km^2 km^2 ON KBAS': kba_overlap_task.get(),
                }
            else:
                row_data = {
                    'country': country,
                    'country area km^2': country_area_km2,
                    'scenario': scenario,
                    'service': service,
                    'summary': 'top 10th percentile combined beneficiaries km^2',
                    'value': 'no roads',
                    'source_file': combined_percentile_service_path,
                    'top 10th percentile service km^2 ON PROTECTED AREAS': '0',
                    'top 10th percentile service km^2 km^2 ON KBAS': '0',
                }
            row_df = pandas.DataFrame([row_data])
            result_df = pandas.concat([result_df, row_df], ignore_index=True)

            # Each service [sediment, recharge, flood, cv] vs overlap of all services:
            # (4*2*2) Area of overlap between the top 10% overlap map (IV, not III) and each service’s top_10th_percentile_service_dspop or _road - if all services overlap AND a service exists, then consider that one

            # count area of these overlaps:
            overlap_combo_service_path = os.path.join(
                OVERLAP_DIR, f'overlap_combos_top_10_{country}_{scenario}_{OVERLAPPING_SERVICES_ID}.tif')
            service_vs_overlap_path = os.path.join(
                OVERLAP_DIR, f'{country}_{scenario}_{service}_overlapped_{OVERLAPPING_SERVICES_ID}.tif')

            if service_vs_overlap_path not in processed_raster_path_set:
                task_graph.add_task(
                    func=intersection_op,
                    args=(
                        dspop_road_overlap_path, overlap_combo_service_path,
                        service_vs_overlap_path),
                    target_path_list=[service_vs_overlap_path],
                    task_name=(
                        f'overlap combos {country} {scenario} '
                        f'{OVERLAPPING_SERVICES_ID}'))
                processed_raster_path_set.add(service_vs_overlap_path)

            service_area_km2_task = task_graph.add_task(
                func=calculate_pixel_area_km2,
                args=(service_vs_overlap_path, projection_epsg),
                store_result=True,
                task_name=f'calculate pixel area for {projection_epsg}')
            service_area_km2 = service_area_km2_task.get()

            pa_overlap_task, kba_overlap_task = calculate_pa_kba_overlaps(
                task_graph,
                service_vs_overlap_path,
                service_vs_overlap_path,
                country,
                scenario,
                service,
                projection_epsg
            )

            row_data = {
                'country': country,
                'country area km^2': country_area_km2,
                'scenario': scenario,
                'service': service,
                'summary': r'coverage of service top 10% on top 10% overlapping services',
                'value': service_area_km2,
                'source_file': service_vs_overlap_path,
                'top 10th percentile service km^2 ON PROTECTED AREAS': pa_overlap_task.get(),
                'top 10th percentile service km^2 km^2 ON KBAS': kba_overlap_task.get(),
            }
            row_df = pandas.DataFrame([row_data])
            result_df = pandas.concat([result_df, row_df], ignore_index=True)

        # In panel 4 on the 4 panel displays, where those maps = 3
        # All services:
        # III: (2*2) Total area of top 10% solutions overlap map =
        # Total solution - any service anywhere and also all the overlaps
        # IV: (2*2) Area of top 10% solutions overlap (combined services, i.e. single panel fig) =
        # Where just the overlaps are, not where any of the services have their top 10% but don’t overlap with other services
        # E.g., top_10p_overlap_IDN_conservation_inf_each_ecosystem_service_400

        for each_or_other in [EACH_ECOSYSTEM_SERVICE_ID, OVERLAPPING_SERVICES_ID]:
            for scenario in [RESTORATION_SCENARIO, CONSERVATION_SCENARIO_INF, CONSERVATION_SCENARIO_ALL]:
                overlap_combo_service_path = os.path.join(
                    OVERLAP_DIR, f'overlap_combos_top_10_{country}_{scenario}_{each_or_other}.tif')
                service_area_km2_task = task_graph.add_task(
                    func=calculate_pixel_area_km2,
                    args=(overlap_combo_service_path, projection_epsg),
                    store_result=True,
                    task_name=f'calculate pixel area for {projection_epsg}')
                service_area_km2 = service_area_km2_task.get()
                pa_overlap_task, kba_overlap_task = calculate_pa_kba_overlaps(
                    task_graph,
                    overlap_combo_service_path,
                    overlap_combo_service_path,
                    country,
                    scenario,
                    service,
                    projection_epsg
                )
                row_data = {
                    'country': country,
                    'country area km^2': country_area_km2,
                    'scenario': scenario,
                    'summary': f'top 10th percentile {each_or_other} km^2',
                    'value': service_area_km2,
                    'source_file': overlap_combo_service_path,
                    'top 10th percentile service km^2 ON PROTECTED AREAS': pa_overlap_task.get(),
                    'top 10th percentile service km^2 km^2 ON KBAS': kba_overlap_task.get(),
                }
                row_df = pandas.DataFrame([row_data])
                result_df = pandas.concat([result_df, row_df], ignore_index=True)

    result_df.to_csv('analysisv2222.csv', index=False, na_rep='')
    LOGGER.debug('finished analysis, exitin')


def main():
    task_graph = taskgraph.TaskGraph(WORKING_DIR, -1)
    top_10_percent_maps = [
        ('PH', CONSERVATION_SCENARIO_INF,),
        ('PH', CONSERVATION_SCENARIO_ALL,),
        ('PH', RESTORATION_SCENARIO,),
        ('IDN', CONSERVATION_SCENARIO_INF,),
        ('IDN', CONSERVATION_SCENARIO_ALL,),
        ('IDN', RESTORATION_SCENARIO,),
    ]

    overlapping_services = [
        ((), (SEDIMENT_SERVICE, FLOOD_MITIGATION_SERVICE), 2, operator.eq, 'sed/flood'),
        ((), (FLOOD_MITIGATION_SERVICE, RECHARGE_SERVICE), 2, operator.eq, 'flood/recharge'),
        ((), (SEDIMENT_SERVICE, RECHARGE_SERVICE), 2, operator.eq, 'sed/recharge'),
        ((CV_SERVICE,), (SEDIMENT_SERVICE, FLOOD_MITIGATION_SERVICE, RECHARGE_SERVICE), 1, operator.eq, 'cv/and one other service'),
        ((), (CV_SERVICE, SEDIMENT_SERVICE, FLOOD_MITIGATION_SERVICE, RECHARGE_SERVICE), 3, operator.eq, '3 service overlaps'),
        #((), (CV_SERVICE, SEDIMENT_SERVICE, FLOOD_MITIGATION_SERVICE, RECHARGE_SERVICE), 4, operator.eq, '4 service overlaps'),
    ]

    each_service = [
        ((), (SEDIMENT_SERVICE,), 1, operator.eq, "sediment"),
        ((), (FLOOD_MITIGATION_SERVICE,), 1, operator.eq, "flood"),
        ((), (RECHARGE_SERVICE,), 1, operator.eq, "recharge"),
        ((), (CV_SERVICE,), 1, operator.eq, 'coastal v'),
        ((), (SEDIMENT_SERVICE, FLOOD_MITIGATION_SERVICE, RECHARGE_SERVICE, CV_SERVICE), 2, operator.ge, '> 1 service overlap'),
    ]

    combined_dspop_overlap_service_map = collections.defaultdict(dict)
    processed_raster_path_set = set()

    executor = ThreadPoolExecutor()
    for country, scenario in top_10_percent_maps:
        for service_set, service_set_title in [
                (each_service, EACH_ECOSYSTEM_SERVICE_ID),
                (overlapping_services, OVERLAPPING_SERVICES_ID),
                ]:
            figure_title = f'Overlaps between top 10% of priorities for each ecosystem service ({scenario})'
            overlap_sets = []
            category_list = ['none']
            for required_service_tuple, optional_service_tuple, overlap_threshold, comparitor_op, legend_category in service_set:
                required_service_subset = []
                optional_service_subset = []
                category_list.append(legend_category)

                for service_tuple, service_subset in [
                        (required_service_tuple, required_service_subset),
                        (optional_service_tuple, optional_service_subset)]:
                    for service in service_tuple:
                        # loop through all the services, they always have a dspop and some of them have a roads, if roads then combine
                        dspop_road_overlap_path = os.path.join(OVERLAP_DIR, f'dspop_road_overlap_{country}_{scenario}_{service}.tif')

                        top_10th_percentile_service_dspop_path = FILENAMES[country][scenario][service]['top_10th_percentile_service_dspop']
                        if 'top_10th_percentile_service_road' in FILENAMES[country][scenario][service]:
                            # combine road and dspop if road exists
                            top_10th_percentile_service_road_path = FILENAMES[country][scenario][service]['top_10th_percentile_service_road']
                            if dspop_road_overlap_path not in processed_raster_path_set:
                                task_graph.add_task(
                                    func=overlap_dspop_road_op,
                                    args=(
                                        top_10th_percentile_service_dspop_path,
                                        top_10th_percentile_service_road_path,
                                        f'top10_{country}_{scenario}',
                                        dspop_road_overlap_path),
                                    target_path_list=[dspop_road_overlap_path],
                                    task_name=f'dspop road {service} {country} {scenario}')
                                processed_raster_path_set.add(dspop_road_overlap_path)

                        else:
                            # doesn't exist but we don't lose anything by just doing the dspop
                            dspop_road_overlap_path = top_10th_percentile_service_dspop_path
                        combined_dspop_overlap_service_map[f'{country}_{scenario}'][service] = dspop_road_overlap_path
                        service_subset.append(dspop_road_overlap_path)

                overlap_sets.append((required_service_subset, optional_service_subset, overlap_threshold, comparitor_op))

            overlap_combo_service_path = os.path.join(
                OVERLAP_DIR, f'overlap_combos_top_10_{country}_{scenario}_{service_set_title}.tif')
            if overlap_combo_service_path not in processed_raster_path_set:
                task_graph.add_task(
                    func=overlap_combos_op,
                    args=(
                        task_graph,
                        overlap_sets,
                        f'{country}_{scenario}',
                        overlap_combo_service_path),
                    target_path_list=[overlap_combo_service_path],
                    task_name=f'top 10% of combo priorities {country} {scenario}')
                processed_raster_path_set.add(overlap_combo_service_path)
            LOGGER.debug(overlap_combo_service_path)

            figure_title = f'Top 10% of priorities for {service_set_title} ({scenario})'
            executor.submit(
                style_rasters, (
                    COUNTRY_OUTLINE_PATH[country],
                    country,
                    [overlap_combo_service_path],
                    [category_list],
                    country == 'IDN',
                    overlap_colormap(service_set_title),
                    COLOR_LIST[service_set_title],
                    ['categorical'],
                    [(0, 5)] if service_set_title==OVERLAPPING_SERVICES_ID else [None],
                    GLOBAL_FIG_SIZE,
                    os.path.join(FIG_DIR, f'top_10p_overlap_{country}_{scenario}_{service_set_title}_{GLOBAL_DPI}.png'),
                    figure_title, [None], GLOBAL_DPI, task_graph))

    # make 'heat map' overlap
    for country_scenario, ds_pop_rasters in combined_dspop_overlap_service_map.items():
        four_service_overlap_path = os.path.join(
            OVERLAP_DIR, f'four_service_overlap_{country_scenario}.tif')
        if four_service_overlap_path not in processed_raster_path_set:
            task_graph.add_task(
                func=add_masks,
                args=(ds_pop_rasters.values(), four_service_overlap_path),
                target_path_list=[four_service_overlap_path],
                task_name=f'four overlaps for {country_scenario}')
            processed_raster_path_set.add(four_service_overlap_path)

        country, _, scenario = country_scenario.partition('_')
        executor.submit(
            style_rasters, (
                COUNTRY_OUTLINE_PATH[country],
                country,
                [four_service_overlap_path], #raster_paths
                [['1 service', '2 services', '3 services', '4 services']], #category_list
                None, # stack vertical
                None, # color map or list
                [COLOR_LIST[CONSERVATION_OVERLAP_HEATMAP] if scenario =='conservation' else COLOR_LIST[RESTORATION_OVERLAP_HEATMAP]], #color palette list
                ['categorical'],  #percentil or catgrical
                [None], # base min max list
                None, # fig size
                None, # fig path
                "Top 10% of service overlap for ", # overall title
                [f'{country} {scenario}'], # subfigure title
                None,  # dpi
                task_graph) #task graph
            )

    four_panel_tuples = [
        (SEDIMENT_SERVICE, 'PH', CONSERVATION_SCENARIO_INF, 'Sediment retention (Conservation)'),
        (SEDIMENT_SERVICE, 'IDN', CONSERVATION_SCENARIO_INF, 'Sediment retention (Conservation)'),
        (SEDIMENT_SERVICE, 'PH', CONSERVATION_SCENARIO_ALL, 'Sediment retention (Conservation -- All)'),
        (SEDIMENT_SERVICE, 'IDN', CONSERVATION_SCENARIO_ALL, 'Sediment retention (Conservation -- All)'),
        (SEDIMENT_SERVICE, 'PH', RESTORATION_SCENARIO, 'Sediment retention (Restoration)'),
        (SEDIMENT_SERVICE, 'IDN', RESTORATION_SCENARIO, 'Sediment retention (Restoration)'),
        (FLOOD_MITIGATION_SERVICE, 'PH', CONSERVATION_SCENARIO_INF, 'Flood mitigation (Conservation)'),
        (FLOOD_MITIGATION_SERVICE, 'IDN', CONSERVATION_SCENARIO_INF, 'Flood mitigation (Conservation)'),
        (FLOOD_MITIGATION_SERVICE, 'PH', CONSERVATION_SCENARIO_ALL, 'Flood mitigation (Conservation -- All)'),
        (FLOOD_MITIGATION_SERVICE, 'IDN', CONSERVATION_SCENARIO_ALL, 'Flood mitigation (Conservation -- All)'),
        (FLOOD_MITIGATION_SERVICE, 'PH', RESTORATION_SCENARIO, 'Flood mitigation (Restoration)'),
        (FLOOD_MITIGATION_SERVICE, 'IDN', RESTORATION_SCENARIO, 'Flood mitigation (Restoration)'),
    ]

    for service, country, scenario, figure_title in four_panel_tuples:
        try:
            diff_path = FILENAMES[country][scenario][service]['diff']
            service_dspop_path = FILENAMES[country][scenario][service]['service_dspop']
            service_road_path = FILENAMES[country][scenario][service]['service_road']
            top_10th_percentile_service_dspop_path = FILENAMES[country][scenario][service]['top_10th_percentile_service_dspop']
            top_10th_percentile_service_road_path = FILENAMES[country][scenario][service]['top_10th_percentile_service_road']
            if any([not os.path.exists(path) for path in [diff_path, service_dspop_path, service_road_path, top_10th_percentile_service_dspop_path, top_10th_percentile_service_road_path]]):
                LOGGER.error('missing!')

            combined_percentile_service_path = os.path.join(
                COMBINED_SERVICE_DIR, f'combined_percentile_service_{service}_{country}_{scenario}.tif')
            if combined_percentile_service_path not in processed_raster_path_set:
                task_graph.add_task(
                    func=overlap_dspop_road_op,
                    args=(
                        top_10th_percentile_service_dspop_path,
                        top_10th_percentile_service_road_path,
                        f'fourpanel_{service}_{country}_{scenario}',
                        combined_percentile_service_path),
                    target_path_list=[combined_percentile_service_path],
                    task_name=f'combined service {service} {country} {scenario}')
                processed_raster_path_set.add(combined_percentile_service_path)

            fig_1_title = f'Biophysical supply of {service}'
            fig_2_title = f'{service} for downstream people'
            fig_3_title = f'{service} for downstream roads'
            fig_4_title = f'Top 10% of priorities for {service} for downstream beneficiaries'

            executor.submit(
                style_rasters, (
                COUNTRY_OUTLINE_PATH[country],
                country,
                [diff_path,
                 service_dspop_path,
                 service_road_path,
                 combined_percentile_service_path],
                [[f'{percentile:.0f}th percentile' for percentile in
                  np.linspace(LOW_PERCENTILE, HIGH_PERCENTILE, len(COLOR_LIST[service])-1, endpoint=True)]] * 3 +
                [['benefiting roads only', 'benefiting people only',
                 'benefiting both']],
                country == 'IDN',
                [overlap_colormap(service),
                 overlap_colormap(service),
                 overlap_colormap(service),
                 overlap_colormap(ROAD_AND_PEOPLE_BENFICIARIES_ID)],
                [COLOR_LIST[service],
                 COLOR_LIST[service],
                 COLOR_LIST[service],
                 COLOR_LIST[ROAD_AND_PEOPLE_BENFICIARIES_ID],],
                [(LOW_PERCENTILE, HIGH_PERCENTILE),
                 (LOW_PERCENTILE, HIGH_PERCENTILE),
                 (LOW_PERCENTILE, HIGH_PERCENTILE),
                 'categorical'],
                [None]*4,
                GLOBAL_FIG_SIZE,
                os.path.join(FIG_DIR, f'{service}_{country}_{scenario}.png'),
                figure_title, [
                    fig_1_title,
                    fig_2_title,
                    fig_3_title,
                    fig_4_title,], GLOBAL_DPI, task_graph))
            print(f'done with {service}_{country}_{scenario}.png')

        except Exception:
            LOGGER.error(f'{service} {country} {scenario}')
            raise

    three_panel_no_road_tuple = [
        (RECHARGE_SERVICE, 'IDN', CONSERVATION_SCENARIO_INF, 'Water recharge (Conservation -- Infrastructure)'),
        (RECHARGE_SERVICE, 'PH', CONSERVATION_SCENARIO_INF, 'Water recharge (Conservation -- Infrastructure)'),
        (RECHARGE_SERVICE, 'IDN', CONSERVATION_SCENARIO_ALL, 'Water recharge (Conservation -- All)'),
        (RECHARGE_SERVICE, 'PH', CONSERVATION_SCENARIO_ALL, 'Water recharge (Conservation -- All)'),
        (RECHARGE_SERVICE, 'PH', RESTORATION_SCENARIO, 'Water recharge (Restoration)'),
        (RECHARGE_SERVICE, 'IDN', RESTORATION_SCENARIO, 'Water recharge (Restoration)'),
    ]

    for service, country, scenario, figure_title in three_panel_no_road_tuple:
        try:
            diff_path = FILENAMES[country][scenario][service]['diff']
            service_dspop_path = FILENAMES[country][scenario][service]['service_dspop']
            top_10th_percentile_service_dspop_path = FILENAMES[country][scenario][service]['top_10th_percentile_service_dspop']
            # if any([not os.path.exists(path) for path in [diff_path, service_dspop_path, service_road_path, top_10th_percentile_service_dspop_path, top_10th_percentile_service_road_path]]):
            #     LOGGER.error('missing!')

            fig_1_title = 'Biophysical supply of water recharge'
            fig_2_title = 'Water recharge for downstream people'
            fig_3_title = None
            fig_4_title = 'Top 10% of priorities'

            executor.submit(
                style_rasters, (
                    COUNTRY_OUTLINE_PATH[country],
                    country,
                    [diff_path,
                     service_dspop_path, None,
                     top_10th_percentile_service_dspop_path],
                    [[f'{percentile:.0f}th percentile' for percentile in
                      np.linspace(LOW_PERCENTILE, HIGH_PERCENTILE, len(COLOR_LIST[service])-1, endpoint=True)]] * 3 +
                    [['none', 'benefiting people only',]],
                    country == 'IDN',
                    [overlap_colormap(service),
                     overlap_colormap(service),
                     None,
                     overlap_colormap(PEOPLE_ONLY_BENEFICIARIES_ID),],
                    [COLOR_LIST[service],
                     COLOR_LIST[service],
                     None,
                     COLOR_LIST[PEOPLE_ONLY_BENEFICIARIES_ID],],
                    [(LOW_PERCENTILE, HIGH_PERCENTILE),
                     (LOW_PERCENTILE, HIGH_PERCENTILE),
                     (LOW_PERCENTILE, HIGH_PERCENTILE),
                     'categorical'],
                    [None]*4,
                    GLOBAL_FIG_SIZE,
                    os.path.join(FIG_DIR, f'{service}_{country}_{scenario}.png'),
                    figure_title, [
                        fig_1_title,
                        fig_2_title,
                        fig_3_title,
                        fig_4_title,], GLOBAL_DPI, task_graph))
        except Exception:
            LOGGER.error(f'{service} {country} {scenario}')
            raise

    three_panel_no_diff_tuple = [
        (CV_SERVICE, 'IDN', CONSERVATION_SCENARIO_INF, 'Coastal protection (Conservation - Infrastructure)'),
        (CV_SERVICE, 'PH', CONSERVATION_SCENARIO_INF, 'Coastal protection (Conservation - Infrastructure)'),
        (CV_SERVICE, 'IDN', CONSERVATION_SCENARIO_ALL, 'Coastal protection (Conservation - All)'),
        (CV_SERVICE, 'PH', CONSERVATION_SCENARIO_ALL, 'Coastal protection (Conservation - All)'),
        (CV_SERVICE, 'PH', RESTORATION_SCENARIO, 'Coastal protection (Restoration)'),
        (CV_SERVICE, 'IDN', RESTORATION_SCENARIO, 'Coastal protection (Restoration)'),
    ]

    for service, country, scenario, figure_title in three_panel_no_diff_tuple:
        try:
            service_dspop_path = FILENAMES[country][scenario][service]['service_dspop']
            service_road_path = FILENAMES[country][scenario][service]['service_road']
            top_10th_percentile_service_dspop_path = FILENAMES[country][scenario][service]['top_10th_percentile_service_dspop']
            top_10th_percentile_service_road_path = FILENAMES[country][scenario][service]['top_10th_percentile_service_road']
            if any([not os.path.exists(path) for path in [service_dspop_path, service_road_path, top_10th_percentile_service_dspop_path, top_10th_percentile_service_road_path]]):
                LOGGER.error('missing!')
            combined_percentile_service_path = os.path.join(
                COMBINED_SERVICE_DIR, f'combined_percentile_service_{service}_{country}_{scenario}.tif')

            if combined_percentile_service_path not in processed_raster_path_set:
                task_graph.add_task(
                    func=overlap_dspop_road_op,
                    args=(
                        top_10th_percentile_service_dspop_path,
                        top_10th_percentile_service_road_path,
                        f'3panel_{service}_{country}_{scenario}',
                        combined_percentile_service_path),
                    target_path_list=[combined_percentile_service_path],
                    task_name=f'combined service {service} {country} {scenario}')
                processed_raster_path_set.add(combined_percentile_service_path)

            fig_1_title = None
            fig_2_title = 'Coastal protection for coastal people'
            fig_3_title = 'Coastal protection for coastal roads'
            fig_4_title = 'Top 10% of priorities for coastal protection for coastal beneficiaries'
            executor.submit(
                style_rasters, (
                    COUNTRY_OUTLINE_PATH[country],
                    country,
                    [None, service_dspop_path,
                     service_road_path,
                     combined_percentile_service_path],
                    [[f'{percentile:.0f}th percentile' for percentile in
                      np.linspace(LOW_PERCENTILE, HIGH_PERCENTILE, len(COLOR_LIST[service])-1, endpoint=True)]] * 3 +
                    [['benefiting roads only', 'benefiting people only',
                     'benefiting both']],
                    country == 'IDN',
                    [overlap_colormap(service),
                     overlap_colormap(service),
                     overlap_colormap(service),
                     overlap_colormap(ROAD_AND_PEOPLE_BENFICIARIES_ID),],
                    [COLOR_LIST[service],
                     COLOR_LIST[service],
                     COLOR_LIST[service],
                     COLOR_LIST[ROAD_AND_PEOPLE_BENFICIARIES_ID],],
                    [(LOW_PERCENTILE, HIGH_PERCENTILE),
                     (LOW_PERCENTILE, HIGH_PERCENTILE),
                     (LOW_PERCENTILE, HIGH_PERCENTILE),
                     'categorical'],
                    [None]*4,
                    GLOBAL_FIG_SIZE,
                    os.path.join(FIG_DIR, f'{service}_{country}_{scenario}.png'),
                    figure_title, [
                        fig_1_title,
                        fig_2_title,
                        fig_3_title,
                        fig_4_title,], GLOBAL_DPI, task_graph,
                        ), {'pixel_coarsen_factor': 50})
        except Exception:
            LOGGER.error(f'{service} {country} {scenario}')
            raise

    # calculate total overlap

    executor.shutdown(wait=True)
    do_analyses(task_graph, processed_raster_path_set)

    task_graph.close()
    task_graph.join()


def add_masks(raster_path_list, target_raster_path):
    """Where a raster is > 0, add a "1" to the final result."""

    aligned_rasters = [
        os.path.join(ALGINED_DIR, f'aligned_for_add_masks_{os.path.basename(path)}')
        for path in raster_path_list]
    geoprocessing.align_and_resize_raster_stack(
        raster_path_list, aligned_rasters, ['near']*4,
        GLOBAL_PIXEL_SIZE, 'intersection')

    nodata_list = [
        geoprocessing.get_raster_info(path)['nodata'][0]
        for path in raster_path_list]
    target_nodata = -1
    def _add_masks(*array_list):
        result = numpy.zeros(array_list[0].shape, dtype=int)
        valid_mask = numpy.zeros(result.shape, dtype=bool)
        for nodata, array in zip(nodata_list, array_list):
            if nodata is not None:
                local_valid_mask = array != nodata
            else:
                local_valid_mask = numpy.ones(result.shape, dtype=bool)
            result[local_valid_mask] += array[local_valid_mask] > 0
            valid_mask |= local_valid_mask
        result[~valid_mask] = target_nodata
        return result

    geoprocessing.single_thread_raster_calculator(
        [(path, 1) for path in aligned_rasters], _add_masks,
        target_raster_path, gdal.GDT_Int16, target_nodata,
        allow_different_blocksize=True)


def calculate_pixel_area_km2(base_raster_path, target_epsg):
    source_raster = gdal.OpenEx(base_raster_path, gdal.OF_RASTER)
    target_srs = osr.SpatialReference()
    target_srs.ImportFromEPSG(target_epsg)

    LOGGER.debug(f'about to warp {base_raster_path} to epsg:{target_epsg}')
    reprojected_raster = gdal.Warp(
        '',
        source_raster,
        format='MEM',
        dstSRS=target_srs)
    reprojected_band = reprojected_raster.GetRasterBand(1)
    nodata = reprojected_band.GetNoDataValue()
    LOGGER.debug(f'nodata value is {nodata}')
    LOGGER.debug(f'read warped {base_raster_path}')
    reprojected_data = reprojected_band.ReadAsArray()
    reprojected_geotransform = reprojected_raster.GetGeoTransform()
    reprojected_pixel_width, reprojected_pixel_height = (
        reprojected_geotransform[1], abs(reprojected_geotransform[5]))

    # Calculate the area of pixels with values > 1
    pixel_area = reprojected_pixel_width * reprojected_pixel_height
    count = ((reprojected_data > 0) & (reprojected_data != nodata)).sum()
    total_area = count * pixel_area / 1e6  # covert to km2

    LOGGER.debug(f'total area of {base_raster_path}: {total_area}km2')
    return total_area


def calculate_vector_area_km2(vector_path, target_epsg):
    source = gdal.OpenEx(vector_path, gdal.OF_VECTOR)
    layer = source.GetLayer()
    source_srs = layer.GetSpatialRef()
    target_srs = osr.SpatialReference()
    target_srs.ImportFromEPSG(target_epsg)
    transformer = osr.CoordinateTransformation(source_srs, target_srs)
    total_area_km2 = 0  # Initialize total area in square kilometers

    for feature in layer:
        geom = feature.GetGeometryRef()
        geom.Transform(transformer)
        area_m2 = geom.GetArea()
        total_area_km2 += area_m2 / 1e6

    return total_area_km2


if __name__ == '__main__':
    with open(RASTER_STYLE_LOG_PATH, 'w') as file:
        file.write('var datasets={\n\t"(*clear*)": "",\n')
    main()
    with open(RASTER_STYLE_LOG_PATH, 'a') as file:
        file.write('};\n')

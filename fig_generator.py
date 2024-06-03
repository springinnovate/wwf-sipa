from pathlib import Path
import csv
import glob
import itertools
import logging
import numpy
import operator
import os
import sys

from ecoshard import geoprocessing
from ecoshard import taskgraph
from matplotlib.colors import LinearSegmentedColormap
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
WORKING_DIR = 'fig_generator_dir'
FIG_DIR = os.path.join(WORKING_DIR, 'rendered_figures')
ALGINED_DIR = os.path.join(WORKING_DIR, 'aligned_rasters')
OVERLAP_DIR = os.path.join(WORKING_DIR, 'overlap_rasters')
SCALED_DIR = os.path.join(WORKING_DIR, 'scaled_rasters')
COMBINED_SERVICE_DIR = os.path.join(WORKING_DIR, 'combined_services')
for dir_path in [
        WORKING_DIR,
        FIG_DIR,
        ALGINED_DIR,
        OVERLAP_DIR,
        SCALED_DIR]:
    os.makedirs(dir_path, exist_ok=True)

ROOT_DATA_DIR = r'D:\repositories\wwf-sipa\post_processing_results_no_road_recharge'

GLOBAL_PIXEL_SIZE = (0.0008333333333333333868, -0.0008333333333333333868)
LOW_PERCENTILE = 10
HIGH_PERCENTILE = 90
BASE_FONT_SIZE = 12
GLOBAL_FIG_SIZE = 10
GLOBAL_DPI = 800
ELLIPSOID_EPSG = 6933

FLOOD_MITIGATION_SERVICE = 'flood mitigation'
RECHARGE_SERVICE = 'recharge'
SEDIMENT_SERVICE = 'sediment'
CV_SERVICE = 'coastal vulnerability'
RESTORATION_SCENARIO = 'restoration'
CONSERVATION_SCENARIO = 'conservation'
EACH_ECOSYSTEM_SERVICE_ID = 'each ecosystem service'
OVERLAPPING_SERVICES_ID = 'overlapping services'
ROAD_AND_PEOPLE_BENFICIARIES_ID = 'road and people beneficiaries'
PEOPLE_ONLY_BENEFICIARIES_ID = 'people beneficiares'

NODATA_COLOR = '#ffffff'
COLOR_LIST = {
    ROAD_AND_PEOPLE_BENFICIARIES_ID: [NODATA_COLOR, '#674ea7', '#a64d79', '#4c1130'],
    PEOPLE_ONLY_BENEFICIARIES_ID: [NODATA_COLOR, '#a64d79'],
    SEDIMENT_SERVICE: [NODATA_COLOR, '#ffbd4b', '#cc720a', '#8c4e07', '#4d2b04'],
    RECHARGE_SERVICE: [NODATA_COLOR, '#cfffff', '#72b1cc', '#4f7abc', '#2b424d'],
    FLOOD_MITIGATION_SERVICE: [NODATA_COLOR, '#d9fff8', '#adccc6', '#778c88', '#414d4a'],
    CV_SERVICE: [NODATA_COLOR, '#e5ffc9', '#c1cc42', '#848c2e', '#484D19'],
    EACH_ECOSYSTEM_SERVICE_ID: [NODATA_COLOR, '#CC720A', '#72B1CC', '#ADCCC6', '#C1CC43', '#000000'],
    OVERLAPPING_SERVICES_ID: [NODATA_COLOR, '#8C4E07', '#4F7A8C', '#778C88', '#848C2E', '#F0027F', '#000000'],
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
        CONSERVATION_SCENARIO: {
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
        }
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
        CONSERVATION_SCENARIO: {
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
        }
    }
}


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
    sample_points = np.linspace(0, 1, num_samples)
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

    # Ensure the subplot height aligns with the desired aspect ratio
    subplot_width = subplot_height * aspect_ratio

    # Calculate total figure size in inches
    total_width = cols * subplot_width
    total_height = rows * subplot_height

    return (total_width, total_height)


def style_rasters(
        country_outline_vector_path, raster_paths, category_list,
        stack_vertical, color_map_or_list, percentile_or_categorical_list, fig_size,
        fig_path, overall_title, subfigure_title_list, dpi, pixel_coarsen_factor=1):
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

    for idx, (base_raster_path, categories, color_map, percentile_or_categorical) in enumerate(zip(raster_paths, category_list, colormap_list, percentile_or_categorical_list)):
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
            f'scaled_for_fig_{os.path.basename(base_raster_path)}')

        LOGGER.info(f'scaling {scaled_path}')
        geoprocessing.warp_raster(
            base_raster_path, target_pixel_size, scaled_path,
            'mode')
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
            values = numpy.linspace(0, 1, len(categories))
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


def do_analyses(task_graph):
    scenario_service_tuples = list(itertools.product(
        [CONSERVATION_SCENARIO, RESTORATION_SCENARIO],
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
                task_graph.add_task(
                    func=overlap_dspop_road_op,
                    args=(
                        top_10th_percentile_service_dspop_path,
                        top_10th_percentile_service_road_path,
                        f'top10_{country}_{scenario}',
                        dspop_road_overlap_path),
                    target_path_list=[dspop_road_overlap_path],
                    task_name=f'dspop road {service} {country} {scenario}')
            else:
                # doesn't exist but we don't lose anything by just doing the dspop
                dspop_road_overlap_path = top_10th_percentile_service_dspop_path
                dspop_road_id = 'dspop'
            # TODO: reproject dspop_road_overlap_path, then sum nonzero pixels
            service_area_km2 = calculate_pixel_area_km2(
                dspop_road_overlap_path, projection_epsg)
            row_data = {
                'country': country,
                'country area km^2': country_area_km2,
                'scenario': scenario,
                'service': service,
                'summary': f'top 10th percentile {dspop_road_id} km^2',
                'value': service_area_km2,
                'source_file': dspop_road_overlap_path,
            }
            row_df = pandas.DataFrame([row_data])
            result_df = pandas.concat([result_df, row_df], ignore_index=True)

        # This is panel 4 on the 4 panel displays right now the ugly colored one (combined beneficiary map, where those maps = non-zero)
        # E.g., cv_IDN_conservation_inf
        # II: (4*2*2) Area where each service’s beneficiaries overlap (subset of the above)
            if 'top_10th_percentile_service_road' in FILENAMES[country][scenario][service]:
                combined_percentile_service_path = os.path.join(
                    COMBINED_SERVICE_DIR, f'combined_percentile_service_{service}_{country}_{scenario}.tif')
                task_graph.add_task(
                    func=intersection_op,
                    args=(
                        top_10th_percentile_service_dspop_path,
                        top_10th_percentile_service_road_path,
                        combined_percentile_service_path),
                    target_path_list=[combined_percentile_service_path],
                    task_name=f'combined service {service} {country} {scenario}')

                service_area_km2 = calculate_pixel_area_km2(
                    combined_percentile_service_path, projection_epsg)
                row_data = {
                    'country': country,
                    'country area km^2': country_area_km2,
                    'scenario': scenario,
                    'service': service,
                    'summary': 'top 10th percentile combined beneficiaries km^2',
                    'value': service_area_km2,
                    'source_file': combined_percentile_service_path,
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

            task_graph.add_task(
                func=intersection_op,
                args=(
                    dspop_road_overlap_path, overlap_combo_service_path,
                    service_vs_overlap_path),
                target_path_list=[service_vs_overlap_path],
                task_name=(
                    f'overlap combos {country} {scenario} '
                    f'{OVERLAPPING_SERVICES_ID}'))

            service_area_km2 = calculate_pixel_area_km2(
                service_vs_overlap_path, projection_epsg)
            row_data = {
                'country': country,
                'country area km^2': country_area_km2,
                'scenario': scenario,
                'service': service,
                'summary': r'coverage of service top 10% on top 10% overlapping services',
                'value': service_area_km2,
                'source_file': service_vs_overlap_path,
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
            for scenario in [RESTORATION_SCENARIO, CONSERVATION_SCENARIO]:
                overlap_combo_service_path = os.path.join(
                    OVERLAP_DIR, f'overlap_combos_top_10_{country}_{scenario}_{each_or_other}.tif')
                service_area_km2 = calculate_pixel_area_km2(
                    overlap_combo_service_path, projection_epsg)
                row_data = {
                    'country': country,
                    'country area km^2': country_area_km2,
                    'scenario': scenario,
                    'summary': f'top 10th percentile {each_or_other} km^2',
                    'value': service_area_km2,
                    'source_file': overlap_combo_service_path,
                }
                row_df = pandas.DataFrame([row_data])
                result_df = pandas.concat([result_df, row_df], ignore_index=True)

    result_df.to_csv('analysis.csv', index=False, na_rep='')
    LOGGER.debug('finished analysis, exitin')


def main():
    task_graph = taskgraph.TaskGraph(WORKING_DIR, -1)
    top_10_percent_maps = [
        ('PH', CONSERVATION_SCENARIO,),
        ('PH', RESTORATION_SCENARIO,),
        ('IDN', CONSERVATION_SCENARIO,),
        ('IDN', RESTORATION_SCENARIO,),
    ]

    overlapping_services = [
        ((), (SEDIMENT_SERVICE, FLOOD_MITIGATION_SERVICE), 2, operator.eq, 'sed/flood'),
        ((), (FLOOD_MITIGATION_SERVICE, RECHARGE_SERVICE), 2, operator.eq, 'flood/recharge'),
        ((), (SEDIMENT_SERVICE, RECHARGE_SERVICE), 2, operator.eq, 'sed/recharge'),
        ((CV_SERVICE,), (SEDIMENT_SERVICE, FLOOD_MITIGATION_SERVICE, RECHARGE_SERVICE), 1, operator.eq, 'cv/and one other service'),
        ((), (CV_SERVICE, SEDIMENT_SERVICE, FLOOD_MITIGATION_SERVICE, RECHARGE_SERVICE), 3, operator.eq, '3 service overlaps'),
        ((), (CV_SERVICE, SEDIMENT_SERVICE, FLOOD_MITIGATION_SERVICE, RECHARGE_SERVICE), 4, operator.eq, '4 service overlaps'),
    ]

    each_service = [
        ((), (SEDIMENT_SERVICE,), 1, operator.eq, "sediment"),
        ((), (FLOOD_MITIGATION_SERVICE,), 1, operator.eq, "flood"),
        ((), (RECHARGE_SERVICE,), 1, operator.eq, "recharge"),
        ((), (CV_SERVICE,), 1, operator.eq, 'coastal v'),
        ((), (SEDIMENT_SERVICE, FLOOD_MITIGATION_SERVICE, RECHARGE_SERVICE, CV_SERVICE), 2, operator.ge, '> 1 service overlap'),
    ]

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
                            task_graph.add_task(
                                func=overlap_dspop_road_op,
                                args=(
                                    top_10th_percentile_service_dspop_path,
                                    top_10th_percentile_service_road_path,
                                    f'top10_{country}_{scenario}',
                                    dspop_road_overlap_path),
                                target_path_list=[dspop_road_overlap_path],
                                task_name=f'dspop road {service} {country} {scenario}')
                        else:
                            # doesn't exist but we don't lose anything by just doing the dspop
                            dspop_road_overlap_path = top_10th_percentile_service_dspop_path
                        service_subset.append(dspop_road_overlap_path)

                overlap_sets.append((required_service_subset, optional_service_subset, overlap_threshold, comparitor_op))

            overlap_combo_service_path = os.path.join(
                OVERLAP_DIR, f'overlap_combos_top_10_{country}_{scenario}_{service_set_title}.tif')

            task_graph.add_task(
                func=overlap_combos_op,
                args=(
                    task_graph,
                    overlap_sets,
                    f'{country}_{scenario}',
                    overlap_combo_service_path),
                target_path_list=[overlap_combo_service_path],
                task_name=f'top 10% of combo priorities {country} {scenario}')
            LOGGER.debug(overlap_combo_service_path)

            figure_title = f'Top 10% of priorities for {service_set_title} ({scenario})'
            color_map = overlap_colormap(service_set_title)
            style_rasters(
                COUNTRY_OUTLINE_PATH[country],
                [overlap_combo_service_path],
                [category_list],
                country == 'IDN',
                color_map,
                ['categorical'],
                GLOBAL_FIG_SIZE,
                os.path.join(FIG_DIR, f'top_10p_overlap_{country}_{scenario}_{service_set_title}_{GLOBAL_DPI}.png'),
                figure_title, [None], GLOBAL_DPI)

    four_panel_tuples = [
        (SEDIMENT_SERVICE, 'PH', CONSERVATION_SCENARIO, 'Sediment retention (Conservation)'),
        (SEDIMENT_SERVICE, 'IDN', CONSERVATION_SCENARIO, 'Sediment retention (Conservation)'),
        (FLOOD_MITIGATION_SERVICE, 'IDN', CONSERVATION_SCENARIO, 'Flood mitigation (Conservation)'),
        (FLOOD_MITIGATION_SERVICE, 'PH', CONSERVATION_SCENARIO, 'Flood mitigation (Conservation)'),
        (FLOOD_MITIGATION_SERVICE, 'IDN', RESTORATION_SCENARIO, 'Flood mitigation (Restoration)'),
        (SEDIMENT_SERVICE, 'IDN', RESTORATION_SCENARIO, 'Sediment retention (Restoration)'),
        (FLOOD_MITIGATION_SERVICE, 'PH', RESTORATION_SCENARIO, 'Flood mitigation (Restoration)'),
        (SEDIMENT_SERVICE, 'PH', RESTORATION_SCENARIO, 'Sediment retention (Restoration)'),
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
            task_graph.add_task(
                func=overlap_dspop_road_op,
                args=(
                    top_10th_percentile_service_dspop_path,
                    top_10th_percentile_service_road_path,
                    f'fourpanel_{service}_{country}_{scenario}',
                    combined_percentile_service_path),
                target_path_list=[combined_percentile_service_path],
                task_name=f'combined service {service} {country} {scenario}')

            fig_1_title = f'Biophysical supply of {service}'
            fig_2_title = f'{service} for downstream people'
            fig_3_title = f'{service} for downstream roads'
            fig_4_title = f'Top 10% of priorities for {service} for downstream beneficiaries'

            style_rasters(
                COUNTRY_OUTLINE_PATH[country],
                [diff_path,
                 service_dspop_path,
                 service_road_path,
                 combined_percentile_service_path],
                [[f'{LOW_PERCENTILE}th percentile',
                  '50th percentile',
                 f'{HIGH_PERCENTILE}th percentile']] * 3 +
                [['none', 'benefiting roads only', 'benefiting people only',
                 'benefiting both']],
                country == 'IDN',
                [overlap_colormap(service),
                 overlap_colormap(service),
                 overlap_colormap(service),
                 overlap_colormap(ROAD_AND_PEOPLE_BENFICIARIES_ID)],
                [(LOW_PERCENTILE, HIGH_PERCENTILE),
                 (LOW_PERCENTILE, HIGH_PERCENTILE),
                 (LOW_PERCENTILE, HIGH_PERCENTILE),
                 'categorical'],
                GLOBAL_FIG_SIZE,
                os.path.join(FIG_DIR, f'{service}_{country}_{scenario}.png'),
                figure_title, [
                    fig_1_title,
                    fig_2_title,
                    fig_3_title,
                    fig_4_title,], GLOBAL_DPI)
            print(f'done with {service}_{country}_{scenario}.png')

        except Exception:
            LOGGER.error(f'{service} {country} {scenario}')
            raise

    three_panel_no_road_tuple = [
        (RECHARGE_SERVICE, 'IDN', CONSERVATION_SCENARIO, 'Water recharge (Conservation)'),
        (RECHARGE_SERVICE, 'PH', CONSERVATION_SCENARIO, 'Water recharge (Conservation)'),
        (RECHARGE_SERVICE, 'PH', RESTORATION_SCENARIO, 'Water recharge (Restoration)'),
        (RECHARGE_SERVICE, 'IDN', RESTORATION_SCENARIO, 'Water recharge (Restoration)'),
    ]

    for service, country, scenario, figure_title in three_panel_no_road_tuple:
        try:
            diff_path = FILENAMES[country][scenario][service]['diff']
            service_dspop_path = FILENAMES[country][scenario][service]['service_dspop']
            top_10th_percentile_service_dspop_path = FILENAMES[country][scenario][service]['top_10th_percentile_service_dspop']
            if any([not os.path.exists(path) for path in [diff_path, service_dspop_path, service_road_path, top_10th_percentile_service_dspop_path, top_10th_percentile_service_road_path]]):
                LOGGER.error('missing!')

            fig_1_title = 'Biophysical supply of water recharge'
            fig_2_title = 'Water recharge for downstream people'
            fig_3_title = None
            fig_4_title = 'Top 10% of priorities'

            style_rasters(
                COUNTRY_OUTLINE_PATH[country],
                [diff_path,
                 service_dspop_path, None,
                 top_10th_percentile_service_dspop_path],
                [[f'{LOW_PERCENTILE}th percentile',
                  '50th percentile',
                 f'{HIGH_PERCENTILE}th percentile']] * 3 +
                [['none', 'benefiting roads only', 'benefiting people only',
                 'benefiting both']],
                country == 'IDN',
                [overlap_colormap(service),
                 overlap_colormap(service),
                 None,
                 overlap_colormap(PEOPLE_ONLY_BENEFICIARIES_ID),],
                [(LOW_PERCENTILE, HIGH_PERCENTILE),
                 (LOW_PERCENTILE, HIGH_PERCENTILE),
                 (LOW_PERCENTILE, HIGH_PERCENTILE),
                 'categorical'],
                GLOBAL_FIG_SIZE,
                os.path.join(FIG_DIR, f'{service}_{country}_{scenario}.png'),
                figure_title, [
                    fig_1_title,
                    fig_2_title,
                    fig_3_title,
                    fig_4_title,], GLOBAL_DPI)
        except Exception:
            LOGGER.error(f'{service} {country} {scenario}')
            raise

    three_panel_no_diff_tuple = [
        (CV_SERVICE, 'IDN', CONSERVATION_SCENARIO, 'Coastal protection (Conservation)'),
        (CV_SERVICE, 'PH', CONSERVATION_SCENARIO, 'Coastal protection (Conservation)'),
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

            task_graph.add_task(
                func=overlap_dspop_road_op,
                args=(
                    top_10th_percentile_service_dspop_path,
                    top_10th_percentile_service_road_path,
                    f'3panel_{service}_{country}_{scenario}',
                    combined_percentile_service_path),
                target_path_list=[combined_percentile_service_path],
                task_name=f'combined service {service} {country} {scenario}')

            fig_1_title = None
            fig_2_title = 'Coastal protection for coastal people'
            fig_3_title = 'Coastal protection for coastal roads'
            fig_4_title = 'Top 10% of priorities for coastal protection for coastal beneficiaries'

            style_rasters(
                COUNTRY_OUTLINE_PATH[country],
                [None, service_dspop_path,
                 service_road_path,
                 combined_percentile_service_path],
                [[f'{LOW_PERCENTILE}th percentile',
                  '50th percentile',
                 f'{HIGH_PERCENTILE}th percentile']] * 3 +
                [['none', 'benefiting roads only', 'benefiting people only',
                 'benefiting both']],
                country == 'IDN',
                [overlap_colormap(service),
                 overlap_colormap(service),
                 overlap_colormap(service),
                 overlap_colormap(ROAD_AND_PEOPLE_BENFICIARIES_ID),],
                [(LOW_PERCENTILE, HIGH_PERCENTILE),
                 (LOW_PERCENTILE, HIGH_PERCENTILE),
                 (LOW_PERCENTILE, HIGH_PERCENTILE),
                 'categorical'],
                GLOBAL_FIG_SIZE,
                os.path.join(FIG_DIR, f'{service}_{country}_{scenario}.png'),
                figure_title, [
                    fig_1_title,
                    fig_2_title,
                    fig_3_title,
                    fig_4_title,], GLOBAL_DPI,
                pixel_coarsen_factor=50)
        except Exception:
            LOGGER.error(f'{service} {country} {scenario}')
            raise

    task_graph.close()
    task_graph.join()


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
    main()

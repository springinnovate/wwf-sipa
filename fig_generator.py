import csv
import glob
import itertools
import logging
import numpy
import os
import sys
from pathlib import Path

from ecoshard import geoprocessing
from ecoshard import taskgraph
from matplotlib.colors import LinearSegmentedColormap
from osgeo import gdal
from osgeo import osr
from pyproj import Transformer
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pyproj

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
COARSE_CV_SERVICE_DIR = os.path.join(WORKING_DIR, 'coarse_cv_service')
for dir_path in [
        WORKING_DIR, FIG_DIR, ALGINED_DIR, OVERLAP_DIR, SCALED_DIR,
        COARSE_CV_SERVICE_DIR]:
    os.makedirs(dir_path, exist_ok=True)

ROOT_DATA_DIR = r'D:\repositories\wwf-sipa\post_processing_results_no_road_recharge'

CV_COARSEN_SCALE = 20
LOW_PERCENTILE = 10
HIGH_PERCENTILE = 90
BASE_FONT_SIZE = 12
GLOBAL_FIG_SIZE = 10
GLOBAL_DPI = 400
SAMPLING_METHOD = 'near'
NODATA_COLOR = '#ffffff'
COLOR_LIST = {
    '1_element': [NODATA_COLOR, '#e41a1c'],
    '3_element': [NODATA_COLOR, '#7fc97f', '#beaed4', '#fdc086'],
    '5_element': [NODATA_COLOR, '#fdcdac', '#cbd5e8', '#f4cae4', '#e6f5c9', '#984ea3'],
    '7_element': [NODATA_COLOR, '#7fc97f', '#beaed4', '#fdc086', '#ffff99', '#386cb0', '#f0027f', '#e41a1c'],
    '8_element': [NODATA_COLOR, '#7fc97f', '#beaed4', '#fdc086', '#ffff99', '#386cb0', '#f0027f', '#bf5b17', '#e41a1c'],
    '9_element': [NODATA_COLOR, '#7fc97f', '#beaed4', '#fdc086', '#ffff99', '#386cb0', '#f0027f', '#bf5b17', '#666666', '#e41a1c'],
}

FLOOD_MITIGATION_SERVICE = 'flood mitigation'
RECHARGE_SERVICE = 'recharge'
SEDIMENT_SERVICE = 'sediment'
CV_SERVICE = 'coastal vulnerability'
RESTORATION_SCENARIO = 'restoration'
CONSERVATION_SCENARIO = 'conservation'

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


def style_rasters(raster_paths, category_list, stack_vertical, color_map_or_list, percentile_or_categorical, fig_size, fig_path, overall_title, subfigure_title_list, dpi):
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

    for idx, (base_raster_path, categories, color_map) in enumerate(zip(raster_paths, category_list, colormap_list)):
        if base_raster_path is None:
            axs[idx].axis('off')
            continue
        raster_info = geoprocessing.get_raster_info(base_raster_path)
        target_pixel_size = scale_pixel_size(raster_info['raster_size'], n_pixels, raster_info['pixel_size'])
        scaled_path = os.path.join(
            SCALED_DIR,
            f'scaled_for_fig_{os.path.basename(base_raster_path)}')

        LOGGER.info(f'scaling {scaled_path}')
        geoprocessing.warp_raster(
            base_raster_path, target_pixel_size, scaled_path,
            SAMPLING_METHOD)
        LOGGER.info('scaled!')

        base_array = gdal.OpenEx(scaled_path, gdal.OF_RASTER).ReadAsArray()

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
        axs[idx].imshow(styled_array, origin='upper')
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

    fontsize_for_suptitle = adjust_suptitle_fontsize(
        fig, BASE_FONT_SIZE)
    fig.suptitle(overall_title, fontsize=fontsize_for_suptitle)

    adjust_font_size(axs[idx], fig, BASE_FONT_SIZE)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(fig_path, dpi=dpi)
    plt.close(fig)


def overlap_dspop_road_op(raster_a_path, raster_b_path, unique_prefix, target_path):
    def _overlap_dspop_road_op(array_a, array_b):
        result = array_a + (2 * array_b)
        return result

    aligned_rasters = [
        os.path.join(ALGINED_DIR, f'aligned_{unique_prefix}_{os.path.basename(path)}')
        for path in [raster_a_path, raster_b_path]]
    LOGGER.debug(f'for {raster_a_path} does it exist: {os.path.exists(raster_a_path)}')
    pixel_size = geoprocessing.get_raster_info(raster_a_path)['pixel_size']
    geoprocessing.align_and_resize_raster_stack(
        [raster_a_path, raster_b_path], aligned_rasters, [SAMPLING_METHOD]*2,
        pixel_size, 'intersection')
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
        for required_path_list, optional_path_list, _ in overlap_combo_list
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
        next_optional_index, next_required_index, overlap_threshold = (
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
                        (local_overlap >= overlap_threshold)] = service_index
                    local_overlap[:] = 0
                    local_max_overlap = 0
                    service_index += 1
                    next_optional_index, next_required_index, overlap_threshold = (
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
    for required_path_list, optional_path_list, overlap_threshold in overlap_combo_list:
        index_list.append((
            next_offset+len(required_path_list),
            next_offset+len(required_path_list)+len(optional_path_list),
            overlap_threshold))
        next_offset = index_list[-1][1]
    index_list.pop(0)

    pixel_size = geoprocessing.get_raster_info(
        flat_path_list[0])['pixel_size']

    task_graph.add_task(
        func=geoprocessing.align_and_resize_raster_stack,
        args=(
            unique_path_list, aligned_rasters,
            [SAMPLING_METHOD] * len(unique_path_list),
            pixel_size, 'intersection'),
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


def main():
    task_graph = taskgraph.TaskGraph(WORKING_DIR, -1)
    # coarsen CV so it shows up better
    for country, scenario in itertools.product(
            ['IDN', 'PH'], [RESTORATION_SCENARIO, CONSERVATION_SCENARIO]):
        cv_set = FILENAMES[country][scenario][CV_SERVICE]
        for key, base_raster_path in cv_set.items():
            coarsened_raster_path = os.path.join(
                COARSE_CV_SERVICE_DIR, os.path.basename(base_raster_path))
            base_pixel_size = numpy.array(
                geoprocessing.get_raster_info(
                    base_raster_path)['pixel_size'])
            task_graph.add_task(
                func=geoprocessing.warp_raster,
                args=(
                    base_raster_path, tuple(base_pixel_size * CV_COARSEN_SCALE),
                    coarsened_raster_path, 'max'),
                target_path_list=[coarsened_raster_path],
                task_name=f'coarsening CV for {key}')
            FILENAMES[country][scenario][CV_SERVICE][key] = (
                coarsened_raster_path)
    task_graph.join()
    top_10_percent_maps = [
        ('PH', CONSERVATION_SCENARIO,),
        ('PH', RESTORATION_SCENARIO,),
        ('IDN', CONSERVATION_SCENARIO,),
        ('IDN', RESTORATION_SCENARIO,),
    ]

    overlapping_services = [
        ((), (SEDIMENT_SERVICE, FLOOD_MITIGATION_SERVICE), 2, 'sed/flood'),
        ((), (FLOOD_MITIGATION_SERVICE, RECHARGE_SERVICE), 2, 'flood/recharge'),
        ((), (SEDIMENT_SERVICE, RECHARGE_SERVICE), 2, 'sed/recharge'),
        ((CV_SERVICE,), (SEDIMENT_SERVICE, FLOOD_MITIGATION_SERVICE, RECHARGE_SERVICE), 1, 'cv/at least one other service'),
        ((), (SEDIMENT_SERVICE, FLOOD_MITIGATION_SERVICE, RECHARGE_SERVICE), 3, 'sed/flood/recharge'),
        ((), (CV_SERVICE, FLOOD_MITIGATION_SERVICE, RECHARGE_SERVICE), 3, 'cv/flood/recharge'),
        ((), (SEDIMENT_SERVICE, CV_SERVICE, RECHARGE_SERVICE), 3, 'sed/cv/recharge'),
        ((), (SEDIMENT_SERVICE, FLOOD_MITIGATION_SERVICE, CV_SERVICE), 3, 'sed/flood/cv'),
        ((), (SEDIMENT_SERVICE, FLOOD_MITIGATION_SERVICE, RECHARGE_SERVICE, CV_SERVICE), 4, 'sed/flood/recharge/cv'),
    ]

    each_service = [
        ((), (SEDIMENT_SERVICE,), 1, "sediment"),
        ((), (FLOOD_MITIGATION_SERVICE,), 1, "flood"),
        ((), (RECHARGE_SERVICE,), 1, "recharge"),
        ((), (CV_SERVICE,), 1, 'coastal v'),
        ((), (SEDIMENT_SERVICE, FLOOD_MITIGATION_SERVICE, RECHARGE_SERVICE, CV_SERVICE), 2, '> 1 service overlap'),
    ]

    for country, scenario in top_10_percent_maps:
        for service_set, service_set_title in [
                (each_service, 'each ecosystem service'),
                (overlapping_services, 'overlapping services'),
                ]:
            figure_title = f'Overlaps between top 10% of priorities for each ecosystem service ({scenario})'
            overlap_sets = []
            category_list = ['none']
            for required_service_tuple, optional_service_tuple, overlap_threshold, legend_category in service_set:
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

                overlap_sets.append((required_service_subset, optional_service_subset, overlap_threshold))

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
            color_map = overlap_colormap(f'{len(overlap_sets)}_element')
            style_rasters(
                [overlap_combo_service_path],
                [category_list],
                country == 'IDN',
                color_map,
                'categorical',
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
                [diff_path,
                 service_dspop_path,
                 service_road_path,
                 combined_percentile_service_path],
                [[f'{LOW_PERCENTILE}th percentile',
                 f'{HIGH_PERCENTILE}th percentile']] * 3 +
                ['benefiting roads only', 'benefiting people only',
                 'benefiting both'],
                country == 'IDN',
                [plt.get_cmap('turbo'),
                 plt.get_cmap('turbo'),
                 plt.get_cmap('turbo'),
                 overlap_colormap('3_element')],
                (LOW_PERCENTILE, HIGH_PERCENTILE),
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
                [diff_path,
                 service_dspop_path, None,
                 top_10th_percentile_service_dspop_path],
                [[f'{LOW_PERCENTILE}th percentile',
                 f'{HIGH_PERCENTILE}th percentile']] * 3 +
                ['benefiting roads only', 'benefiting people only',
                 'benefiting both'],
                country == 'IDN',
                [plt.get_cmap('turbo'),
                 plt.get_cmap('turbo'),
                 plt.get_cmap('turbo'),
                 overlap_colormap('3_element'),],
                (LOW_PERCENTILE, HIGH_PERCENTILE),
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
                [None, service_dspop_path,
                 service_road_path,
                 combined_percentile_service_path],
                [[f'{LOW_PERCENTILE}th percentile',
                 f'{HIGH_PERCENTILE}th percentile']] * 3 +
                ['benefiting roads only', 'benefiting people only',
                 'benefiting both'],
                country == 'IDN',
                [plt.get_cmap('turbo'),
                 plt.get_cmap('turbo'),
                 plt.get_cmap('turbo'),
                 overlap_colormap('3_element'),],
                (LOW_PERCENTILE, HIGH_PERCENTILE),
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


    # TODO: still do the analysis
    # How much do services overlap with each other? Which one overlaps the least?
    # Need to know:

    ph_epsg_projection = 3121
    ph_vector_path = "data/admin_boundaries/PH_outline.gpkg"

    idn_epsg_projection = 23830
    idn_vector_path = "data/admin_boundaries/IDN_outline.gpkg"

    for projection_epsg, vector_path in [
            (ph_epsg_projection, ph_vector_path),
            (idn_epsg_projection, idn_vector_path)]:
        # Total area of the country
        area_km2 = calculate_vector_area_km2(vector_path, projection_epsg)
        # Total area of top 10% solutions overlap map
        # 'top_10p_overlap_IDN_restoration_overlapping_services_400.png'
        # 'top_10p_overlap_IDN_conservation_inf_each_ecosystem_service_400.png'
        # 'top_10p_overlap_IDN_conservation_inf_overlapping_services_400.png'
        # 'top_10p_overlap_IDN_restoration_each_ecosystem_service_400.png'
        # Total area of each service’s top_10th_percentile_service_dspop or _road
        # Area of top 10% solutions overlap = 3, 5, 6
        # Area where each service’s panel D map is = 3
        # top_10th_percentile_service_dspop_[service]_[country]_[scenario]
        # Area of overlap between the top 10% overlap map and each service’s
        # top_10th_percentile_service_dspop or _road

    country_list = ['PH', 'IDN']
    scenario_list = [CONSERVATION_SCENARIO, RESTORATION_SCENARIO]
    service_list = [SEDIMENT_SERVICE, RECHARGE_SERVICE, FLOOD_MITIGATION_SERVICE, CV_SERVICE]

    for country, scenario, service in itertools.product(
            country_list, scenario_list, service_list):
        pass

    task_graph.close()
    task_graph.join()
    return


def calculate_pixel_area_km2(base_raster_path, target_epsg):
    source_raster = gdal.OpenEx(base_raster_path, gdal.OF_RASTER)
    target_srs = osr.SpatialReference()
    target_srs.ImportFromEPSG(target_epsg)

    reprojected_raster = gdal.Warp(
        '', source_raster, format='MEM', dstSRS=target_srs)
    reprojected_band = reprojected_raster.GetRasterBand(1)
    reprojected_data = reprojected_band.ReadAsArray()
    reprojected_geotransform = reprojected_raster.GetGeoTransform()
    reprojected_pixel_width, reprojected_pixel_height = (
        reprojected_geotransform[1], abs(reprojected_geotransform[5]))

    # Calculate the area of pixels with values > 1
    pixel_area = reprojected_pixel_width * reprojected_pixel_height
    count = (reprojected_data > 0).sum()
    total_area = count * pixel_area / 1e6  # covert to km2

    return total_area


def calculate_vector_area_km2(vector_path, target_epsg):
    source = gdal.OpenEx(vector_path, gdal.OF_VECTOR)
    layer = source.GetLayer()
    target_srs = pyproj.CRS(f'EPSG:{target_epsg}')
    source_srs = layer.GetSpatialRef()
    transformer = Transformer.from_crs(source_srs, target_srs, always_xy=True)
    area_km2 = 0

    for feature in layer:
        geom = feature.GetGeometryRef()
        geom.Transform(transformer.transform)
        area_m2 = geom.GetArea()
        area_km2 = area_m2 / 1e6

    return area_km2


if __name__ == '__main__':
    main()

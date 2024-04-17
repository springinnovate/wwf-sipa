import csv
import glob
import logging
import numpy
import os
import sys

from ecoshard import geoprocessing
from ecoshard import taskgraph
from matplotlib.colors import LinearSegmentedColormap
from osgeo import gdal
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

CUSTOM_STYLE_DIR = 'custom_styles'
WORKING_DIR = 'raster_styler_working_dir'
FIG_DIR = os.path.join(WORKING_DIR, 'fig_dir')
os.makedirs(FIG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.DEBUG,
    stream=sys.stdout,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(levelname)s %(name)s'
        ' [%(pathname)s.%(funcName)s:%(lineno)d] %(message)s'))
LOGGER = logging.getLogger(__name__)


WORKSPACE_DIR = 'fig_generator_workspace'

root_dir = r'D:\repositories\wwf-sipa\post_processing_results_no_road_recharge'

filenames = {
    'PH': {
        'restoration': {
            'flood_mitigation': {
                'diff': 'diff_flood_mitigation_PH_restoration.tif',
                'service_dspop': 'service_dspop_flood_mitigation_PH_restoration.tif',
                'service_road': 'service_road_flood_mitigation_PH_restoration.tif',
                'top_10th_percentile_service_dspop': 'top_10th_percentile_service_dspop_flood_mitigation_PH_restoration.tif',
                'top_10th_percentile_service_road': 'top_10th_percentile_service_road_flood_mitigation_PH_restoration.tif',
            },
            'recharge': {
                'diff': 'diff_recharge_PH_restoration.tif',
                'service_dspop': 'service_dspop_recharge_PH_restoration.tif',
                'top_10th_percentile_service_dspop': 'top_10th_percentile_service_dspop_recharge_PH_restoration.tif',
            },
            'sediment': {
                'diff': 'diff_sediment_PH_restoration.tif',
                'service_dspop': 'service_dspop_sediment_PH_restoration.tif',
                'service_road': 'service_road_sediment_PH_restoration.tif',
                'top_10th_percentile_service_dspop': 'top_10th_percentile_service_dspop_sediment_PH_restoration.tif',
                'top_10th_percentile_service_road': 'top_10th_percentile_service_road_sediment_PH_restoration.tif',
            },
            'cv': {
                'service_dspop': 'service_dspop_cv_ph_restoration_result.tif',
                'service_road': 'service_road_cv_ph_restoration_result.tif',
                'top_10th_percentile_service_dspop': 'top_10th_percentile_service_dspop_cv_ph_restoration_result.tif',
                'top_10th_percentile_service_road': 'top_10th_percentile_service_road_cv_ph_restoration_result.tif',
            },
        },
        'conservation_inf': {
            'flood_mitigation': {
                'diff': 'diff_flood_mitigation_PH_conservation_inf.tif',
                'service_dspop': 'service_dspop_flood_mitigation_PH_conservation_inf.tif',
                'service_road': 'service_road_flood_mitigation_PH_conservation_inf.tif',
                'top_10th_percentile_service_dspop': 'top_10th_percentile_service_dspop_flood_mitigation_PH_conservation_inf.tif',
                'top_10th_percentile_service_road': 'top_10th_percentile_service_road_flood_mitigation_PH_conservation_inf.tif',
            },
            'recharge': {
                'diff': 'diff_recharge_PH_conservation_inf.tif',
                'service_dspop': 'service_dspop_recharge_PH_conservation_inf.tif',
                'top_10th_percentile_service_dspop': 'top_10th_percentile_service_dspop_recharge_PH_conservation_inf.tif',
            },
            'sediment': {
                'diff': 'diff_sediment_PH_conservation_inf.tif',
                'service_dspop': 'service_dspop_sediment_PH_conservation_inf.tif',
                'service_road': 'service_road_sediment_PH_conservation_inf.tif',
                'top_10th_percentile_service_road': 'top_10th_percentile_service_road_sediment_PH_conservation_inf.tif',
                'top_10th_percentile_service_dspop': 'top_10th_percentile_service_dspop_cv_ph_conservation_inf_result.tif',
            },
            'cv': {
                'top_10th_percentile_service_dspop': 'top_10th_percentile_service_dspop_sediment_PH_conservation_inf.tif',
                'service_dspop': 'service_dspop_cv_ph_conservation_inf_result.tif',
                'service_road': 'service_road_cv_ph_conservation_inf_result.tif',
                'top_10th_percentile_service_road': 'top_10th_percentile_service_road_cv_ph_conservation_inf_result.tif',
            },
        }
    },
    'IDN': {
        'restoration': {
            'flood_mitigation': {
                'diff': 'diff_flood_mitigation_IDN_restoration.tif',
                'service_dspop': 'service_dspop_flood_mitigation_IDN_restoration.tif',
                'service_road': 'service_road_flood_mitigation_IDN_restoration.tif',
                'top_10th_percentile_service_dspop': 'top_10th_percentile_service_dspop_flood_mitigation_IDN_restoration.tif',
                'top_10th_percentile_service_road': 'top_10th_percentile_service_road_flood_mitigation_IDN_restoration.tif',
            },
            'recharge': {
                'diff': 'diff_recharge_IDN_restoration.tif',
                'service_dspop': 'service_dspop_recharge_IDN_restoration.tif',
                'top_10th_percentile_service_dspop': 'top_10th_percentile_service_dspop_recharge_IDN_restoration.tif',
            },
            'sediment': {
                'diff': 'diff_sediment_IDN_restoration.tif',
                'service_dspop': 'service_dspop_sediment_IDN_restoration.tif',
                'service_road': 'service_road_sediment_IDN_restoration.tif',
                'top_10th_percentile_service_dspop': 'top_10th_percentile_service_dspop_sediment_IDN_restoration.tif',
                'top_10th_percentile_service_road': 'top_10th_percentile_service_road_sediment_IDN_restoration.tif',
            },
            'cv': {
                'service_dspop': 'service_dspop_cv_idn_restoration_result.tif',
                'service_road': 'service_road_cv_idn_restoration_result.tif',
                'top_10th_percentile_service_dspop': 'top_10th_percentile_service_dspop_cv_idn_restoration_result.tif',
                'top_10th_percentile_service_road': 'top_10th_percentile_service_road_cv_idn_restoration_result.tif',
            },
        },
        'conservation_inf': {
            'flood_mitigation': {
                'diff': 'diff_flood_mitigation_IDN_conservation_inf.tif',
                'service_dspop': 'service_dspop_flood_mitigation_IDN_conservation_inf.tif',
                'service_road': 'service_road_flood_mitigation_IDN_conservation_inf.tif',
                'top_10th_percentile_service_dspop': 'top_10th_percentile_service_dspop_flood_mitigation_IDN_conservation_inf.tif',
                'top_10th_percentile_service_road': 'top_10th_percentile_service_road_flood_mitigation_IDN_conservation_inf.tif',
            },
            'recharge': {
                'diff': 'diff_recharge_IDN_conservation_inf.tif',
                'service_dspop': 'service_dspop_recharge_IDN_conservation_inf.tif',
                'top_10th_percentile_service_dspop': 'top_10th_percentile_service_dspop_recharge_IDN_conservation_inf.tif',
            },
            'sediment': {
                'diff': 'diff_sediment_IDN_conservation_inf.tif',
                'service_dspop': 'service_dspop_sediment_IDN_conservation_inf.tif',
                'service_road': 'service_road_sediment_IDN_conservation_inf.tif',
                'top_10th_percentile_service_road': 'top_10th_percentile_service_road_sediment_IDN_conservation_inf.tif',
                'top_10th_percentile_service_dspop': 'top_10th_percentile_service_dspop_cv_idn_conservation_inf_result.tif',
            },
            'cv': {
                'top_10th_percentile_service_dspop': 'top_10th_percentile_service_dspop_sediment_IDN_conservation_inf.tif',
                'service_dspop': 'service_dspop_cv_idn_conservation_inf_result.tif',
                'service_road': 'service_road_cv_idn_conservation_inf_result.tif',
                'top_10th_percentile_service_road': 'top_10th_percentile_service_road_cv_idn_conservation_inf_result.tif',
            },
        }
    }
}


COLOR_LIST = {
    '5_element': ['#aaaaaa', '#fdcdac', '#cbd5e8', '#e6f5c9', '#e6f5c9', '#984ea3'],
}


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
                raster_dict[current_raster_name]['transparency'] = [int(x) for x in row[1:]  if x != '']
    return raster_dict


def hex_to_rgba(hex_code, transparency):
    hex_code = hex_code.lstrip('#')
    rgb = tuple(int(hex_code[i:i+2], 16)/255.0 for i in (0, 2, 4))
    alpha = transparency / 100  # Convert 0-100 scale to 0-1 scale
    return rgb + (alpha,)


CUSTOM_STYLES = {}
for style_file_path in glob.glob(os.path.join(CUSTOM_STYLE_DIR, '*.csv')):
    CUSTOM_STYLES.update(read_raster_csv(style_file_path))


# def interpolated_colormap(cmap_name, N=100):
#     try:
#         # Get the original colormap from matplotlib
#         original_cmap = plt.colormaps[cmap_name]
#     except:
#         # Handle custom colormap dictionary
#         custom_cmap_info = CUSTOM_STYLES[cmap_name]
#         positions = custom_cmap_info['position']
#         hex_colors = custom_cmap_info['color']
#         transparency = custom_cmap_info['transparency']

#         # Convert hex to RGBA format and include transparency
#         rgba_colors = [
#             hex_to_rgba(hex_colors[i], transparency[i])
#             for i in range(len(hex_colors))]

#         # positions are from 0..100 in the csv
#         original_cmap = LinearSegmentedColormap.from_list(
#             'custom_cmap', list(zip(np.array(positions)/100, rgba_colors)))

#     # Use linspace to get N interpolated colors from the original colormap
#     colors = original_cmap(np.linspace(0, 1, N))
#     # Create a new colormap from these colors
#     new_cmap = LinearSegmentedColormap.from_list(
#         cmap_name + "_interp", colors, N=N)
#     return new_cmap


def root_filename(path):
    return os.path.splitext(os.path.basename(path))[0]


def interpolated_colormap(cmap):
    # This function should create a matplotlib colormap based on your specifications
    return plt.get_cmap(cmap)


def calculate_figsize(aspect_ratio, grid_size, subplot_size, dpi):
    rows, cols = grid_size
    subplot_width, subplot_height = subplot_size

    # Ensure the subplot height aligns with the desired aspect ratio
    subplot_width = subplot_height * aspect_ratio

    # Calculate total figure size in inches
    total_width = cols * subplot_width
    total_height = rows * subplot_height

    return (total_width, total_height)


def style_rasters(raster_paths, stack_vertical, cmap, min_percentile, max_percentile, fig_size, fig_path, overall_title, subfigure_title_list, dpi):
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
        aspect_ratio, (rows, columns), (fig_size, fig_size), dpi)


    fig, axs = plt.subplots(rows, columns, figsize=(fig_width, fig_height))  # Set up a 2x2 grid of plots
    n_pixels = fig_width/2*dpi
    if not single_raster_mode:
        axs = axs.flatten()  # Flatten the 2D array of axes for easier iteration
    else:
        axs = [axs]

    for idx, base_raster_path in enumerate(raster_paths):
        if base_raster_path is None:
            continue
        raster_info = geoprocessing.get_raster_info(base_raster_path)
        target_pixel_size = scale_pixel_size(raster_info['raster_size'], n_pixels, raster_info['pixel_size'])
        scaled_path = os.path.join(
            WORKSPACE_DIR,
            f'scaled_for_fig_{os.path.basename(base_raster_path)}')

        LOGGER.info(f'scaling {scaled_path}')
        geoprocessing.warp_raster(
            base_raster_path, target_pixel_size, scaled_path,
            'near')
        LOGGER.info(f'scaled!')

        base_array = gdal.OpenEx(scaled_path, gdal.OF_RASTER).ReadAsArray()

        # Create a color gradient
        cm = interpolated_colormap(cmap)
        no_data_color = [0, 0, 0, 0]  # Assuming a black NoData color with full transparency

        nodata = geoprocessing.get_raster_info(scaled_path)['nodata'][0]
        nodata_mask = (base_array == nodata) | np.isnan(base_array)
        styled_array = np.empty(base_array.shape + (4,), dtype=float)
        valid_base_array = base_array[~nodata_mask]
        #base_min, base_max = np.min(valid_base_array), np.max(valid_base_array)
        base_min = np.percentile(valid_base_array, min_percentile)
        base_max = np.percentile(valid_base_array, max_percentile)
        LOGGER.debug(f'******{base_raster_path}: {base_min}, {base_max}')
        normalized_array = (valid_base_array - base_min) / (base_max - base_min)

        styled_array[~nodata_mask] = cm(normalized_array)
        styled_array[nodata_mask] = no_data_color

        # Define bounding box for each raster
        bounding_box = geoprocessing.get_raster_info(scaled_path)['bounding_box']
        extend_bb = [bounding_box[i] for i in (0, 2, 1, 3)]

        subfigure_title = subfigure_title_list[idx]
        if subfigure_title is not None:
            axs[idx].set_title(subfigure_title_list[idx], wrap=True)
        axs[idx].imshow(styled_array, extent=extend_bb, origin='upper')
        axs[idx].axis('off')  # Turn off axis labels

    fig.suptitle(overall_title, fontsize=16)  # Set the overall title for the figure
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust the layout to make space for the overall title
    plt.savefig(fig_path)


def scale_op(raster_a_path, raster_b_path, target_path):
    def _scale_op(array_a, array_b):
        result = array_a+2*array_b
        return result

    aligned_rasters = [
        os.path.join(WORKSPACE_DIR, f'aligned_{os.path.basename(path)}')
        for path in [raster_a_path, raster_b_path]]
    pixel_size = geoprocessing.get_raster_info(raster_a_path)['pixel_size']
    geoprocessing.align_and_resize_raster_stack(
        [raster_a_path, raster_b_path], aligned_rasters, ['near']*2,
        pixel_size, 'intersection')
    geoprocessing.raster_calculator(
        [(path, 1) for path in aligned_rasters], _scale_op, target_path,
        gdal.GDT_Int16, None, allow_different_blocksize=True)


def overlap_op(task_graph, service_top_10_pair_path_list, target_path):
    """Format of service_top_10_pair_path_list is
        [service_a_dspop, service_a_road, service_b_dspop, service_b_road, ....]
    """
    def _overlap_op(*array_list):
        overlap_count = numpy.zeros(array_list[0].shape, dtype=int)
        result = numpy.zeros(array_list[0].shape, dtype=int)
        list_iter = iter(array_list)
        for index, (service_a, service_b) in enumerate(
                zip(list_iter, list_iter)):
            service_index = index // 2 + 1 # we get pairs of services
            valid_mask = (service_a > 0) | (service_b > 0)
            result[valid_mask] = service_index
            overlap_count[valid_mask] += 1
        result[overlap_count > 1] = len(array_list)//2 + 2 # the 2 puts it over the top
        return result

    aligned_rasters = [
        os.path.join(WORKSPACE_DIR, f'aligned_{index}_{os.path.basename(path)}')
        for index, path in enumerate(service_top_10_pair_path_list)]
    pixel_size = geoprocessing.get_raster_info(
        service_top_10_pair_path_list[0])['pixel_size']
    task_graph.add_task(
        func=geoprocessing.align_and_resize_raster_stack,
        args=(
            service_top_10_pair_path_list, aligned_rasters, ['near']*len(aligned_rasters),
            pixel_size, 'intersection'),
        target_path_list=aligned_rasters,
        task_name='alignining in overlap op')
    task_graph.join()
    geoprocessing.raster_calculator(
        [(path, 1) for path in aligned_rasters], _overlap_op, target_path,
        gdal.GDT_Int16, None, allow_different_blocksize=True)


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


def main():
    task_graph = taskgraph.TaskGraph(WORKSPACE_DIR, -1)

    # First map(s) [for each scenario, country] - Title: “Top 10% of priorities for each ecosystem service (Conservation)” “Top 10% of priorities for each ecosystem service (Restoration)”
    #     Only sediment “Sediment retention”
    #     Only flood “Flood mitigation”
    #     Only recharge “Water recharge”
    #     Only cv “Only coastal protection”
    #     Any overlap “Overlaps between services”

    top_10_percent_maps = [
        ('PH', 'conservation_inf',),
        ('IDN', 'conservation_inf',),
        ('IDN', 'conservation_inf',),
        ('PH', 'conservation_inf',),
        ('IDN', 'restoration',),
        ('IDN', 'restoration',),
        ('PH', 'restoration',),
        ('PH', 'restoration',),
    ]

    all_services = [
        'sediment',
        'flood_mitigation',
        'recharge',
        'cv']

    for country, scenario in top_10_percent_maps:

        figure_title = 'Top 10% of priorities for each ecosystem service (scenario)'

        # Only sediment “Sediment retention” -
        # Only flood “Flood mitigation”
        # Only recharge “Water recharge”
        # Only cv “Only coastal protection”
        # Any overlap “Overlaps between services”

        top_10_percent_map_list = []
        for service in all_services:
            top_10_percent_map_list.append(os.path.join(root_dir, filenames[country][scenario][service]['top_10th_percentile_service_dspop']))
            if 'top_10th_percentile_service_road' in filenames[country][scenario][service]:
                top_10_percent_map_list.append(os.path.join(root_dir, filenames[country][scenario][service]['top_10th_percentile_service_road']))
            else:
                # doesn't exist but we don't lose anything by just doing the dspop
                top_10_percent_map_list.append(top_10_percent_map_list[-1])

        overlap_percentile_service_path = os.path.join(
            WORKSPACE_DIR, f'overlap_top_10_{country}_{scenario}.tif')

        task_graph.add_task(
            func=overlap_op,
            args=(
                task_graph,
                top_10_percent_map_list,
                overlap_percentile_service_path),
            target_path_list=[overlap_percentile_service_path],
            task_name=f'top 10% of priorities {country} {scenario}')

        figure_title = f'Top 10% of priorities for each ecosystem service ({scenario}) - {service}'
        cm = overlap_colormap('5_element')
        fig_size = 20
        style_rasters(
            [overlap_percentile_service_path],
            country == 'IDN',
            cm,
            0, 100,
            fig_size,
            os.path.join(FIG_DIR, f'top_10p_overlap_{country}_{scenario}.png'),
            figure_title, [None], 100)
        return



    four_panel_tuples = [
        ('sediment', 'PH', 'conservation_inf', 'Sediment retention (Conservation)'),
        ('sediment', 'IDN', 'conservation_inf', 'Sediment retention (Conservation)'),
        ('flood_mitigation', 'IDN', 'conservation_inf', 'Flood mitigation (Conservation)'),
        ('flood_mitigation', 'PH', 'conservation_inf', 'Flood mitigation (Conservation)'),
        ('flood_mitigation', 'IDN', 'restoration', 'Flood mitigation (Restoration)'),
        ('sediment', 'IDN', 'restoration', 'Sediment retention (Restoration)'),
        ('flood_mitigation', 'PH', 'restoration', 'Flood mitigation (Restoration)'),
        ('sediment', 'PH', 'restoration', 'Sediment retention (Restoration)'),
    ]

    for service, country, scenario, figure_title in four_panel_tuples:
        try:
            diff_path = os.path.join(root_dir, filenames[country][scenario][service]['diff'])
            service_dspop_path = os.path.join(root_dir, filenames[country][scenario][service]['service_dspop'])
            service_road_path = os.path.join(root_dir, filenames[country][scenario][service]['service_road'])
            top_10th_percentile_service_dspop_path = os.path.join(root_dir, filenames[country][scenario][service]['top_10th_percentile_service_dspop'])
            top_10th_percentile_service_road_path = os.path.join(root_dir, filenames[country][scenario][service]['top_10th_percentile_service_road'])
            if any([not os.path.exists(path) for path in [diff_path, service_dspop_path, service_road_path, top_10th_percentile_service_dspop_path, top_10th_percentile_service_road_path]]):
                LOGGER.error('missing!')

            combined_percentile_service_path = os.path.join(
                WORKSPACE_DIR, f'combined_percentile_service_{service}_{country}_{scenario}.tif')
            task_graph.add_task(
                func=scale_op,
                args=(
                    top_10th_percentile_service_dspop_path,
                    top_10th_percentile_service_road_path,
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
                country == 'IDN',
                'turbo',
                os.path.join(FIG_DIR, f'{service}_{country}_{scenario}.png'),
                figure_title, [
                    fig_1_title,
                    fig_2_title,
                    fig_3_title,
                    fig_4_title,
                    ])
            print(f'done with {service}_{country}_{scenario}.png')
            return

        except Exception:
            LOGGER.error(f'{service} {country} {scenario}')
            raise
    three_panel_no_road_tuple = [
        ('recharge', 'IDN', 'conservation_inf', 'Water recharge (Conservation)'),
        ('recharge', 'PH', 'conservation_inf', 'Water recharge (Conservation)'),
        ('recharge', 'PH', 'restoration', 'Water recharge (Restoration)'),
        ('recharge', 'IDN', 'restoration', 'Water recharge (Restoration)'),
    ]
    for service, country, scenario, figure_title in three_panel_no_road_tuple:
        try:
            diff_path = os.path.join(root_dir, filenames[country][scenario][service]['diff'])
            service_dspop_path = os.path.join(root_dir, filenames[country][scenario][service]['service_dspop'])
            top_10th_percentile_service_dspop_path = os.path.join(root_dir, filenames[country][scenario][service]['top_10th_percentile_service_dspop'])
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
                country == 'IDN',
                'turbo',
                os.path.join(FIG_DIR, f'{service}_{country}_{scenario}.png'),
                figure_title, [
                    fig_1_title,
                    fig_2_title,
                    fig_3_title,
                    fig_4_title,
                    ])
        except Exception:
            LOGGER.error(f'{service} {country} {scenario}')
            raise

    three_panel_no_diff_tuple = [
        ('cv', 'IDN', 'conservation_inf', 'Coastal protection (Conservation)'),
        ('cv', 'PH', 'conservation_inf', 'Coastal protection (Conservation)'),
        ('cv', 'PH', 'restoration', 'Coastal protection (Restoration)'),
        ('cv', 'IDN', 'restoration', 'Coastal protection (Restoration)'),
    ]

    for service, country, scenario, figure_title in three_panel_no_diff_tuple:
        try:
            service_dspop_path = os.path.join(root_dir, filenames[country][scenario][service]['service_dspop'])
            service_road_path = os.path.join(root_dir, filenames[country][scenario][service]['service_road'])
            top_10th_percentile_service_dspop_path = os.path.join(root_dir, filenames[country][scenario][service]['top_10th_percentile_service_dspop'])
            top_10th_percentile_service_road_path = os.path.join(root_dir, filenames[country][scenario][service]['top_10th_percentile_service_road'])
            if any([not os.path.exists(path) for path in [service_dspop_path, service_road_path, top_10th_percentile_service_dspop_path, top_10th_percentile_service_road_path]]):
                LOGGER.error('missing!')
            combined_percentile_service_path = os.path.join(
                WORKSPACE_DIR, f'combined_percentile_service_{service}_{country}_{scenario}.tif')
            task_graph.add_task(
                func=scale_op,
                args=(
                    top_10th_percentile_service_dspop_path,
                    top_10th_percentile_service_road_path,
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
                country == 'IDN',
                'turbo',
                os.path.join(FIG_DIR, f'{service}_{country}_{scenario}.png'),
                figure_title, [
                    fig_1_title,
                    fig_2_title,
                    fig_3_title,
                    fig_4_title,
                    ])
        except Exception:
            LOGGER.error(f'{service} {country} {scenario}')
            raise
    task_graph.close()
    task_graph.join()


if __name__ == '__main__':
    main()

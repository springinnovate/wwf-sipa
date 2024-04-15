import logging
import numpy
import os
import sys

from ecoshard import geoprocessing
from ecoshard import taskgraph
from osgeo import gdal

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

def scale_op(raster_a_path, raster_b_path, target_path):

    def _scale_op(array_a, array_b, nodata):
        result = numpy.fill(array_a.shape, nodata)
        valid_mask_a = (array_a != nodata)
        result[valid_mask_a] = array_a[valid_mask_a]
        valid_mask_b = (array_b != nodata)
        result[valid_mask_b] += 2*array_a[valid_mask_b]
        return result
    nodata = geoprocessing.get_raster_info(raster_a_path)['nodata'][0]
    geoprocessing.raster_calculator(
        [(raster_a_path, 1), (raster_b_path, 1)], _scale_op, target_path,
        gdal.GDT_Int, nodata)


def main():
    task_graph = taskgraph.TaskGraph(WORKSPACE_DIR, -1)
    four_panel_tuples = [
        ('flood_mitigation', 'IDN', 'conservation_inf'),
        ('sediment', 'IDN', 'conservation_inf'),
        ('flood_mitigation', 'PH', 'conservation_inf'),
        ('sediment', 'PH', 'conservation_inf'),
        ('flood_mitigation', 'IDN', 'restoration'),
        ('sediment', 'IDN', 'restoration'),
        ('flood_mitigation', 'PH', 'restoration'),
        ('sediment', 'PH', 'restoration'),
    ]

    for service, country, scenario in four_panel_tuples:
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
        except Exception:
            LOGGER.error(f'{service} {country} {scenario}')
            raise

    three_panel_no_road_tuple = [
        ('recharge', 'IDN', 'conservation_inf'),
        ('recharge', 'PH', 'conservation_inf'),
        ('recharge', 'PH', 'restoration'),
        ('recharge', 'IDN', 'restoration'),
    ]
    for service, country, scenario in three_panel_no_road_tuple:
        try:
            diff_path = os.path.join(root_dir, filenames[country][scenario][service]['diff'])
            service_dspop_path = os.path.join(root_dir, filenames[country][scenario][service]['service_dspop'])
            top_10th_percentile_service_dspop_path = os.path.join(root_dir, filenames[country][scenario][service]['top_10th_percentile_service_dspop'])
            if any([not os.path.exists(path) for path in [diff_path, service_dspop_path, service_road_path, top_10th_percentile_service_dspop_path, top_10th_percentile_service_road_path]]):
                LOGGER.error('missing!')
        except Exception:
            LOGGER.error(f'{service} {country} {scenario}')
            raise

    three_panel_no_diff_tuple = [
        ('cv', 'IDN', 'conservation_inf'),
        ('cv', 'PH', 'conservation_inf'),
        ('cv', 'PH', 'restoration'),
        ('cv', 'IDN', 'restoration'),
    ]

    for service, country, scenario in three_panel_no_diff_tuple:
        try:
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
        except Exception:
            LOGGER.error(f'{service} {country} {scenario}')
            raise
    task_graph.close()
    task_graph.join()


if __name__ == '__main__':
    main()

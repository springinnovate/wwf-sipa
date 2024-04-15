import os
import re


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
        #diff_[service]_[country]_[scenario]
        # service_dspop_[service]_[country]_[scenario]
        # service_road_[service]_[country]_[scenario]
        # top_10th_percentile_service_dspop_[service]_[country]_[scenario] + 2*
        # top_10th_percentile_service_road_[service]_[country]_[scenario]
        diff_path = os.path.join(root_dir, filenames[country][scenario][service]['diff'])
        service_dspop_path = os.path.join(root_dir, filenames[country][scenario][service]['service_dspop'])
        service_road_path = os.path.join(root_dir, filenames[country][scenario][service]['service_road'])
        top_10th_percentile_service_dspop_path = os.path.join(root_dir, filenames[country][scenario][service]['top_10th_percentile_service_dspop'])
        top_10th_percentile_service_road_path = os.path.join(root_dir, filenames[country][scenario][service]['top_10th_percentile_service_road'])
        if any([not os.path.exists(path) for path in [diff_path, service_dspop_path, service_road_path, top_10th_percentile_service_dspop_path, top_10th_percentile_service_road_path]]):
            print('missing!')
    except Exception as e:
        print(f'{service} {country} {scenario}')
        raise

three_panel_no_road_tuple = [
    ('recharge', 'IDN', 'conservation_inf'),
    ('recharge', 'PH', 'conservation_inf'),
    ('recharge', 'PH', 'restoration'),
    ('recharge', 'IDN', 'restoration'),
]
for service, country, scenario in three_panel_no_road_tuple:
    try:
        #diff_[service]_[country]_[scenario]
        # service_dspop_[service]_[country]_[scenario]
        # service_road_[service]_[country]_[scenario]
        # top_10th_percentile_service_dspop_[service]_[country]_[scenario] + 2*
        # top_10th_percentile_service_road_[service]_[country]_[scenario]
        diff_path = os.path.join(root_dir, filenames[country][scenario][service]['diff'])
        service_dspop_path = os.path.join(root_dir, filenames[country][scenario][service]['service_dspop'])
        top_10th_percentile_service_dspop_path = os.path.join(root_dir, filenames[country][scenario][service]['top_10th_percentile_service_dspop'])
        if any([not os.path.exists(path) for path in [diff_path, service_dspop_path, service_road_path, top_10th_percentile_service_dspop_path, top_10th_percentile_service_road_path]]):
            print('missing!')
    except Exception as e:
        print(f'{service} {country} {scenario}')
        raise

three_panel_no_diff_tuple = [
    ('cv', 'IDN', 'conservation_inf'),
    ('cv', 'PH', 'conservation_inf'),
    ('cv', 'PH', 'restoration'),
    ('cv', 'IDN', 'restoration'),
]

for service, country, scenario in three_panel_no_diff_tuple:
    try:
        #diff_[service]_[country]_[scenario]
        # service_dspop_[service]_[country]_[scenario]
        # service_road_[service]_[country]_[scenario]
        # top_10th_percentile_service_dspop_[service]_[country]_[scenario] + 2*
        # top_10th_percentile_service_road_[service]_[country]_[scenario]
        service_dspop_path = os.path.join(root_dir, filenames[country][scenario][service]['service_dspop'])
        service_road_path = os.path.join(root_dir, filenames[country][scenario][service]['service_road'])
        top_10th_percentile_service_dspop_path = os.path.join(root_dir, filenames[country][scenario][service]['top_10th_percentile_service_dspop'])
        top_10th_percentile_service_road_path = os.path.join(root_dir, filenames[country][scenario][service]['top_10th_percentile_service_road'])
        if any([not os.path.exists(path) for path in [diff_path, service_dspop_path, service_road_path, top_10th_percentile_service_dspop_path, top_10th_percentile_service_road_path]]):
            print('missing!')
    except Exception as e:
        print(f'{service} {country} {scenario}')
        raise
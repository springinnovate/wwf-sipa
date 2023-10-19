var default_vector_style = {
    color: '000000',
    width: 0.1,
    fillColor: '00000000',
}
var aggregate_polygons = {
    '(*clear*)': '',
    'Philippines Administrative Level 3': 'projects/ecoshard-202922/assets/gadm41_PHL_3',
    'Indonesia Administrative Level 3': 'projects/ecoshard-202922/assets/gadm41_IDN_3',
    'Indonesia Administrative Level 4': 'projects/ecoshard-202922/assets/gadm41_IDN_4',
}

var datasets = {
    '(*clear*)': '',
    '10_IDN_conservation_inf_dspop__service_overlap_count': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/10_IDN_conservation_inf_dspop__service_overlap_count.tif',
    '10_IDN_conservation_inf_dspop_ssp245__service_overlap_count': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/10_IDN_conservation_inf_dspop_ssp245__service_overlap_count.tif',
    '10_IDN_conservation_inf_road__service_overlap_count': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/10_IDN_conservation_inf_road__service_overlap_count.tif',
    '10_IDN_conservation_inf_road_ssp245__service_overlap_count': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/10_IDN_conservation_inf_road_ssp245__service_overlap_count.tif',
    '10_IDN_restoration_dspop__service_overlap_count': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/10_IDN_restoration_dspop__service_overlap_count.tif',
    '10_IDN_restoration_dspop_ssp245__service_overlap_count': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/10_IDN_restoration_dspop_ssp245__service_overlap_count.tif',
    '10_IDN_restoration_road__service_overlap_count': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/10_IDN_restoration_road__service_overlap_count.tif',
    '10_IDN_restoration_road_ssp245__service_overlap_count': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/10_IDN_restoration_road_ssp245__service_overlap_count.tif',
    '10_PH_conservation_inf_dspop__service_overlap_count': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/10_PH_conservation_inf_dspop__service_overlap_count.tif',
    '10_PH_conservation_inf_dspop_ssp245__service_overlap_count': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/10_PH_conservation_inf_dspop_ssp245__service_overlap_count.tif',
    '10_PH_conservation_inf_road__service_overlap_count': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/10_PH_conservation_inf_road__service_overlap_count.tif',
    '10_PH_conservation_inf_road_ssp245__service_overlap_count': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/10_PH_conservation_inf_road_ssp245__service_overlap_count.tif',
    '10_PH_restoration_dspop__service_overlap_count': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/10_PH_restoration_dspop__service_overlap_count.tif',
    '10_PH_restoration_dspop_ssp245__service_overlap_count': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/10_PH_restoration_dspop_ssp245__service_overlap_count.tif',
    '10_PH_restoration_road__service_overlap_count': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/10_PH_restoration_road__service_overlap_count.tif',
    '10_PH_restoration_road_ssp245__service_overlap_count': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/10_PH_restoration_road_ssp245__service_overlap_count.tif',
    '25_IDN_conservation_inf_dspop__service_overlap_count': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/25_IDN_conservation_inf_dspop__service_overlap_count.tif',
    '25_IDN_conservation_inf_dspop_ssp245__service_overlap_count': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/25_IDN_conservation_inf_dspop_ssp245__service_overlap_count.tif',
    '25_IDN_conservation_inf_road__service_overlap_count': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/25_IDN_conservation_inf_road__service_overlap_count.tif',
    '25_IDN_conservation_inf_road_ssp245__service_overlap_count': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/25_IDN_conservation_inf_road_ssp245__service_overlap_count.tif',
    '25_IDN_restoration_dspop__service_overlap_count': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/25_IDN_restoration_dspop__service_overlap_count.tif',
    '25_IDN_restoration_dspop_ssp245__service_overlap_count': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/25_IDN_restoration_dspop_ssp245__service_overlap_count.tif',
    '25_IDN_restoration_road__service_overlap_count': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/25_IDN_restoration_road__service_overlap_count.tif',
    '25_IDN_restoration_road_ssp245__service_overlap_count': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/25_IDN_restoration_road_ssp245__service_overlap_count.tif',
    '25_PH_conservation_inf_dspop__service_overlap_count': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/25_PH_conservation_inf_dspop__service_overlap_count.tif',
    '25_PH_conservation_inf_dspop_ssp245__service_overlap_count': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/25_PH_conservation_inf_dspop_ssp245__service_overlap_count.tif',
    '25_PH_conservation_inf_road__service_overlap_count': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/25_PH_conservation_inf_road__service_overlap_count.tif',
    '25_PH_conservation_inf_road_ssp245__service_overlap_count': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/25_PH_conservation_inf_road_ssp245__service_overlap_count.tif',
    '25_PH_restoration_dspop__service_overlap_count': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/25_PH_restoration_dspop__service_overlap_count.tif',
    '25_PH_restoration_dspop_ssp245__service_overlap_count': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/25_PH_restoration_dspop_ssp245__service_overlap_count.tif',
    '25_PH_restoration_road__service_overlap_count': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/25_PH_restoration_road__service_overlap_count.tif',
    '25_PH_restoration_road_ssp245__service_overlap_count': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/25_PH_restoration_road_ssp245__service_overlap_count.tif',
    'diff_flood_mitigation_IDN_conservation_inf': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/diff_flood_mitigation_IDN_conservation_inf.tif',
    'diff_flood_mitigation_IDN_conservation_inf_ssp245': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/diff_flood_mitigation_IDN_conservation_inf_ssp245.tif',
    'diff_flood_mitigation_IDN_restoration': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/diff_flood_mitigation_IDN_restoration.tif',
    'diff_flood_mitigation_IDN_restoration_ssp245': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/diff_flood_mitigation_IDN_restoration_ssp245.tif',
    'diff_flood_mitigation_PH_conservation_inf': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/diff_flood_mitigation_PH_conservation_inf.tif',
    'diff_flood_mitigation_PH_conservation_inf_ssp245': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/diff_flood_mitigation_PH_conservation_inf_ssp245.tif',
    'diff_flood_mitigation_PH_restoration': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/diff_flood_mitigation_PH_restoration.tif',
    'diff_flood_mitigation_PH_restoration_ssp245': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/diff_flood_mitigation_PH_restoration_ssp245.tif',
    'diff_quickflow_IDN_conservation_inf': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/diff_quickflow_IDN_conservation_inf.tif',
    'diff_quickflow_IDN_conservation_inf_ssp245': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/diff_quickflow_IDN_conservation_inf_ssp245.tif',
    'diff_quickflow_IDN_restoration': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/diff_quickflow_IDN_restoration.tif',
    'diff_quickflow_IDN_restoration_ssp245': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/diff_quickflow_IDN_restoration_ssp245.tif',
    'diff_quickflow_PH_conservation_inf': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/diff_quickflow_PH_conservation_inf.tif',
    'diff_quickflow_PH_conservation_inf_ssp245': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/diff_quickflow_PH_conservation_inf_ssp245.tif',
    'diff_quickflow_PH_restoration': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/diff_quickflow_PH_restoration.tif',
    'diff_quickflow_PH_restoration_ssp245': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/diff_quickflow_PH_restoration_ssp245.tif',
    'diff_recharge_IDN_conservation_inf': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/diff_recharge_IDN_conservation_inf.tif',
    'diff_recharge_IDN_conservation_inf_ssp245': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/diff_recharge_IDN_conservation_inf_ssp245.tif',
    'diff_recharge_IDN_restoration': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/diff_recharge_IDN_restoration.tif',
    'diff_recharge_IDN_restoration_ssp245': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/diff_recharge_IDN_restoration_ssp245.tif',
    'diff_recharge_PH_conservation_inf': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/diff_recharge_PH_conservation_inf.tif',
    'diff_recharge_PH_conservation_inf_ssp245': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/diff_recharge_PH_conservation_inf_ssp245.tif',
    'diff_recharge_PH_restoration': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/diff_recharge_PH_restoration.tif',
    'diff_recharge_PH_restoration_ssp245': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/diff_recharge_PH_restoration_ssp245.tif',
    'diff_sediment_IDN_conservation_inf': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/diff_sediment_IDN_conservation_inf.tif',
    'diff_sediment_IDN_conservation_inf_ssp245': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/diff_sediment_IDN_conservation_inf_ssp245.tif',
    'diff_sediment_IDN_restoration': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/diff_sediment_IDN_restoration.tif',
    'diff_sediment_IDN_restoration_ssp245': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/diff_sediment_IDN_restoration_ssp245.tif',
    'diff_sediment_PH_conservation_inf': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/diff_sediment_PH_conservation_inf.tif',
    'diff_sediment_PH_conservation_inf_ssp245': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/diff_sediment_PH_conservation_inf_ssp245.tif',
    'diff_sediment_PH_restoration': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/diff_sediment_PH_restoration.tif',
    'diff_sediment_PH_restoration_ssp245': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/diff_sediment_PH_restoration_ssp245.tif',
    'flood_mitigation_IDN_baseline_historical_climate': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/flood_mitigation_IDN_baseline_historical_climate.tif',
    'flood_mitigation_PH_baseline_historical_climate': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/flood_mitigation_PH_baseline_historical_climate.tif',
    'service_dspop_flood_mitigation_IDN_conservation_inf': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/service_dspop_flood_mitigation_IDN_conservation_inf.tif',
    'service_dspop_flood_mitigation_IDN_conservation_inf_ssp245': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/service_dspop_flood_mitigation_IDN_conservation_inf_ssp245.tif',
    'service_dspop_flood_mitigation_IDN_restoration': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/service_dspop_flood_mitigation_IDN_restoration.tif',
    'service_dspop_flood_mitigation_IDN_restoration_ssp245': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/service_dspop_flood_mitigation_IDN_restoration_ssp245.tif',
    'service_dspop_flood_mitigation_PH_conservation_inf': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/service_dspop_flood_mitigation_PH_conservation_inf.tif',
    'service_dspop_flood_mitigation_PH_conservation_inf_ssp245': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/service_dspop_flood_mitigation_PH_conservation_inf_ssp245.tif',
    'service_dspop_flood_mitigation_PH_restoration': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/service_dspop_flood_mitigation_PH_restoration.tif',
    'service_dspop_flood_mitigation_PH_restoration_ssp245': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/service_dspop_flood_mitigation_PH_restoration_ssp245.tif',
    'service_dspop_recharge_IDN_conservation_inf': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/service_dspop_recharge_IDN_conservation_inf.tif',
    'service_dspop_recharge_IDN_conservation_inf_ssp245': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/service_dspop_recharge_IDN_conservation_inf_ssp245.tif',
    'service_dspop_recharge_IDN_restoration': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/service_dspop_recharge_IDN_restoration.tif',
    'service_dspop_recharge_IDN_restoration_ssp245': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/service_dspop_recharge_IDN_restoration_ssp245.tif',
    'service_dspop_recharge_PH_conservation_inf': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/service_dspop_recharge_PH_conservation_inf.tif',
    'service_dspop_recharge_PH_conservation_inf_ssp245': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/service_dspop_recharge_PH_conservation_inf_ssp245.tif',
    'service_dspop_recharge_PH_restoration': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/service_dspop_recharge_PH_restoration.tif',
    'service_dspop_recharge_PH_restoration_ssp245': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/service_dspop_recharge_PH_restoration_ssp245.tif',
    'service_dspop_sediment_IDN_conservation_inf': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/service_dspop_sediment_IDN_conservation_inf.tif',
    'service_dspop_sediment_IDN_conservation_inf_ssp245': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/service_dspop_sediment_IDN_conservation_inf_ssp245.tif',
    'service_dspop_sediment_IDN_restoration': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/service_dspop_sediment_IDN_restoration.tif',
    'service_dspop_sediment_IDN_restoration_ssp245': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/service_dspop_sediment_IDN_restoration_ssp245.tif',
    'service_dspop_sediment_PH_conservation_inf': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/service_dspop_sediment_PH_conservation_inf.tif',
    'service_dspop_sediment_PH_conservation_inf_ssp245': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/service_dspop_sediment_PH_conservation_inf_ssp245.tif',
    'service_dspop_sediment_PH_restoration': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/service_dspop_sediment_PH_restoration.tif',
    'service_dspop_sediment_PH_restoration_ssp245': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/service_dspop_sediment_PH_restoration_ssp245.tif',
    'service_road_flood_mitigation_IDN_conservation_inf': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/service_road_flood_mitigation_IDN_conservation_inf.tif',
    'service_road_flood_mitigation_IDN_conservation_inf_ssp245': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/service_road_flood_mitigation_IDN_conservation_inf_ssp245.tif',
    'service_road_flood_mitigation_IDN_restoration': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/service_road_flood_mitigation_IDN_restoration.tif',
    'service_road_flood_mitigation_IDN_restoration_ssp245': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/service_road_flood_mitigation_IDN_restoration_ssp245.tif',
    'service_road_flood_mitigation_PH_conservation_inf': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/service_road_flood_mitigation_PH_conservation_inf.tif',
    'service_road_flood_mitigation_PH_conservation_inf_ssp245': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/service_road_flood_mitigation_PH_conservation_inf_ssp245.tif',
    'service_road_flood_mitigation_PH_restoration': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/service_road_flood_mitigation_PH_restoration.tif',
    'service_road_flood_mitigation_PH_restoration_ssp245': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/service_road_flood_mitigation_PH_restoration_ssp245.tif',
    'service_road_recharge_IDN_conservation_inf': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/service_road_recharge_IDN_conservation_inf.tif',
    'service_road_recharge_IDN_conservation_inf_ssp245': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/service_road_recharge_IDN_conservation_inf_ssp245.tif',
    'service_road_recharge_IDN_restoration': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/service_road_recharge_IDN_restoration.tif',
    'service_road_recharge_IDN_restoration_ssp245': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/service_road_recharge_IDN_restoration_ssp245.tif',
    'service_road_recharge_PH_conservation_inf': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/service_road_recharge_PH_conservation_inf.tif',
    'service_road_recharge_PH_conservation_inf_ssp245': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/service_road_recharge_PH_conservation_inf_ssp245.tif',
    'service_road_recharge_PH_restoration': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/service_road_recharge_PH_restoration.tif',
    'service_road_recharge_PH_restoration_ssp245': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/service_road_recharge_PH_restoration_ssp245.tif',
    'service_road_sediment_IDN_conservation_inf': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/service_road_sediment_IDN_conservation_inf.tif',
    'service_road_sediment_IDN_conservation_inf_ssp245': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/service_road_sediment_IDN_conservation_inf_ssp245.tif',
    'service_road_sediment_IDN_restoration': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/service_road_sediment_IDN_restoration.tif',
    'service_road_sediment_IDN_restoration_ssp245': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/service_road_sediment_IDN_restoration_ssp245.tif',
    'service_road_sediment_PH_conservation_inf': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/service_road_sediment_PH_conservation_inf.tif',
    'service_road_sediment_PH_conservation_inf_ssp245': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/service_road_sediment_PH_conservation_inf_ssp245.tif',
    'service_road_sediment_PH_restoration': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/service_road_sediment_PH_restoration.tif',
    'service_road_sediment_PH_restoration_ssp245': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/service_road_sediment_PH_restoration_ssp245.tif',
    'top_10th_percentile_service_dspop_flood_mitigation_IDN_conservation_inf': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/top_10th_percentile_service_dspop_flood_mitigation_IDN_conservation_inf.tif',
    'top_10th_percentile_service_dspop_flood_mitigation_IDN_conservation_inf_ssp245': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/top_10th_percentile_service_dspop_flood_mitigation_IDN_conservation_inf_ssp245.tif',
    'top_10th_percentile_service_dspop_flood_mitigation_IDN_restoration': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/top_10th_percentile_service_dspop_flood_mitigation_IDN_restoration.tif',
    'top_10th_percentile_service_dspop_flood_mitigation_IDN_restoration_ssp245': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/top_10th_percentile_service_dspop_flood_mitigation_IDN_restoration_ssp245.tif',
    'top_10th_percentile_service_dspop_flood_mitigation_PH_conservation_inf': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/top_10th_percentile_service_dspop_flood_mitigation_PH_conservation_inf.tif',
    'top_10th_percentile_service_dspop_flood_mitigation_PH_conservation_inf_ssp245': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/top_10th_percentile_service_dspop_flood_mitigation_PH_conservation_inf_ssp245.tif',
    'top_10th_percentile_service_dspop_flood_mitigation_PH_restoration': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/top_10th_percentile_service_dspop_flood_mitigation_PH_restoration.tif',
    'top_10th_percentile_service_dspop_flood_mitigation_PH_restoration_ssp245': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/top_10th_percentile_service_dspop_flood_mitigation_PH_restoration_ssp245.tif',
    'top_10th_percentile_service_dspop_recharge_IDN_conservation_inf': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/top_10th_percentile_service_dspop_recharge_IDN_conservation_inf.tif',
    'top_10th_percentile_service_dspop_recharge_IDN_conservation_inf_ssp245': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/top_10th_percentile_service_dspop_recharge_IDN_conservation_inf_ssp245.tif',
    'top_10th_percentile_service_dspop_recharge_IDN_restoration': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/top_10th_percentile_service_dspop_recharge_IDN_restoration.tif',
    'top_10th_percentile_service_dspop_recharge_IDN_restoration_ssp245': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/top_10th_percentile_service_dspop_recharge_IDN_restoration_ssp245.tif',
    'top_10th_percentile_service_dspop_recharge_PH_conservation_inf': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/top_10th_percentile_service_dspop_recharge_PH_conservation_inf.tif',
    'top_10th_percentile_service_dspop_recharge_PH_conservation_inf_ssp245': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/top_10th_percentile_service_dspop_recharge_PH_conservation_inf_ssp245.tif',
    'top_10th_percentile_service_dspop_recharge_PH_restoration': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/top_10th_percentile_service_dspop_recharge_PH_restoration.tif',
    'top_10th_percentile_service_dspop_recharge_PH_restoration_ssp245': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/top_10th_percentile_service_dspop_recharge_PH_restoration_ssp245.tif',
    'top_10th_percentile_service_dspop_sediment_IDN_conservation_inf': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/top_10th_percentile_service_dspop_sediment_IDN_conservation_inf.tif',
    'top_10th_percentile_service_dspop_sediment_IDN_conservation_inf_ssp245': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/top_10th_percentile_service_dspop_sediment_IDN_conservation_inf_ssp245.tif',
    'top_10th_percentile_service_dspop_sediment_IDN_restoration': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/top_10th_percentile_service_dspop_sediment_IDN_restoration.tif',
    'top_10th_percentile_service_dspop_sediment_IDN_restoration_ssp245': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/top_10th_percentile_service_dspop_sediment_IDN_restoration_ssp245.tif',
    'top_10th_percentile_service_dspop_sediment_PH_conservation_inf': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/top_10th_percentile_service_dspop_sediment_PH_conservation_inf.tif',
    'top_10th_percentile_service_dspop_sediment_PH_conservation_inf_ssp245': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/top_10th_percentile_service_dspop_sediment_PH_conservation_inf_ssp245.tif',
    'top_10th_percentile_service_dspop_sediment_PH_restoration': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/top_10th_percentile_service_dspop_sediment_PH_restoration.tif',
    'top_10th_percentile_service_dspop_sediment_PH_restoration_ssp245': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/top_10th_percentile_service_dspop_sediment_PH_restoration_ssp245.tif',
    'top_10th_percentile_service_road_flood_mitigation_IDN_conservation_inf': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/top_10th_percentile_service_road_flood_mitigation_IDN_conservation_inf.tif',
    'top_10th_percentile_service_road_flood_mitigation_IDN_conservation_inf_ssp245': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/top_10th_percentile_service_road_flood_mitigation_IDN_conservation_inf_ssp245.tif',
    'top_10th_percentile_service_road_flood_mitigation_IDN_restoration': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/top_10th_percentile_service_road_flood_mitigation_IDN_restoration.tif',
    'top_10th_percentile_service_road_flood_mitigation_IDN_restoration_ssp245': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/top_10th_percentile_service_road_flood_mitigation_IDN_restoration_ssp245.tif',
    'top_10th_percentile_service_road_flood_mitigation_PH_conservation_inf': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/top_10th_percentile_service_road_flood_mitigation_PH_conservation_inf.tif',
    'top_10th_percentile_service_road_flood_mitigation_PH_conservation_inf_ssp245': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/top_10th_percentile_service_road_flood_mitigation_PH_conservation_inf_ssp245.tif',
    'top_10th_percentile_service_road_flood_mitigation_PH_restoration': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/top_10th_percentile_service_road_flood_mitigation_PH_restoration.tif',
    'top_10th_percentile_service_road_flood_mitigation_PH_restoration_ssp245': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/top_10th_percentile_service_road_flood_mitigation_PH_restoration_ssp245.tif',
    'top_10th_percentile_service_road_recharge_IDN_conservation_inf': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/top_10th_percentile_service_road_recharge_IDN_conservation_inf.tif',
    'top_10th_percentile_service_road_recharge_IDN_conservation_inf_ssp245': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/top_10th_percentile_service_road_recharge_IDN_conservation_inf_ssp245.tif',
    'top_10th_percentile_service_road_recharge_IDN_restoration': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/top_10th_percentile_service_road_recharge_IDN_restoration.tif',
    'top_10th_percentile_service_road_recharge_IDN_restoration_ssp245': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/top_10th_percentile_service_road_recharge_IDN_restoration_ssp245.tif',
    'top_10th_percentile_service_road_recharge_PH_conservation_inf': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/top_10th_percentile_service_road_recharge_PH_conservation_inf.tif',
    'top_10th_percentile_service_road_recharge_PH_conservation_inf_ssp245': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/top_10th_percentile_service_road_recharge_PH_conservation_inf_ssp245.tif',
    'top_10th_percentile_service_road_recharge_PH_restoration': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/top_10th_percentile_service_road_recharge_PH_restoration.tif',
    'top_10th_percentile_service_road_recharge_PH_restoration_ssp245': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/top_10th_percentile_service_road_recharge_PH_restoration_ssp245.tif',
    'top_10th_percentile_service_road_sediment_IDN_conservation_inf': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/top_10th_percentile_service_road_sediment_IDN_conservation_inf.tif',
    'top_10th_percentile_service_road_sediment_IDN_conservation_inf_ssp245': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/top_10th_percentile_service_road_sediment_IDN_conservation_inf_ssp245.tif',
    'top_10th_percentile_service_road_sediment_IDN_restoration': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/top_10th_percentile_service_road_sediment_IDN_restoration.tif',
    'top_10th_percentile_service_road_sediment_IDN_restoration_ssp245': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/top_10th_percentile_service_road_sediment_IDN_restoration_ssp245.tif',
    'top_10th_percentile_service_road_sediment_PH_conservation_inf': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/top_10th_percentile_service_road_sediment_PH_conservation_inf.tif',
    'top_10th_percentile_service_road_sediment_PH_conservation_inf_ssp245': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/top_10th_percentile_service_road_sediment_PH_conservation_inf_ssp245.tif',
    'top_10th_percentile_service_road_sediment_PH_restoration': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/top_10th_percentile_service_road_sediment_PH_restoration.tif',
    'top_10th_percentile_service_road_sediment_PH_restoration_ssp245': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/top_10th_percentile_service_road_sediment_PH_restoration_ssp245.tif',
    'top_25th_percentile_service_dspop_flood_mitigation_IDN_conservation_inf': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/top_25th_percentile_service_dspop_flood_mitigation_IDN_conservation_inf.tif',
    'top_25th_percentile_service_dspop_flood_mitigation_IDN_conservation_inf_ssp245': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/top_25th_percentile_service_dspop_flood_mitigation_IDN_conservation_inf_ssp245.tif',
    'top_25th_percentile_service_dspop_flood_mitigation_IDN_restoration': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/top_25th_percentile_service_dspop_flood_mitigation_IDN_restoration.tif',
    'top_25th_percentile_service_dspop_flood_mitigation_IDN_restoration_ssp245': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/top_25th_percentile_service_dspop_flood_mitigation_IDN_restoration_ssp245.tif',
    'top_25th_percentile_service_dspop_flood_mitigation_PH_conservation_inf': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/top_25th_percentile_service_dspop_flood_mitigation_PH_conservation_inf.tif',
    'top_25th_percentile_service_dspop_flood_mitigation_PH_conservation_inf_ssp245': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/top_25th_percentile_service_dspop_flood_mitigation_PH_conservation_inf_ssp245.tif',
    'top_25th_percentile_service_dspop_flood_mitigation_PH_restoration': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/top_25th_percentile_service_dspop_flood_mitigation_PH_restoration.tif',
    'top_25th_percentile_service_dspop_flood_mitigation_PH_restoration_ssp245': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/top_25th_percentile_service_dspop_flood_mitigation_PH_restoration_ssp245.tif',
    'top_25th_percentile_service_dspop_recharge_IDN_conservation_inf': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/top_25th_percentile_service_dspop_recharge_IDN_conservation_inf.tif',
    'top_25th_percentile_service_dspop_recharge_IDN_conservation_inf_ssp245': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/top_25th_percentile_service_dspop_recharge_IDN_conservation_inf_ssp245.tif',
    'top_25th_percentile_service_dspop_recharge_IDN_restoration': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/top_25th_percentile_service_dspop_recharge_IDN_restoration.tif',
    'top_25th_percentile_service_dspop_recharge_IDN_restoration_ssp245': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/top_25th_percentile_service_dspop_recharge_IDN_restoration_ssp245.tif',
    'top_25th_percentile_service_dspop_recharge_PH_conservation_inf': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/top_25th_percentile_service_dspop_recharge_PH_conservation_inf.tif',
    'top_25th_percentile_service_dspop_recharge_PH_conservation_inf_ssp245': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/top_25th_percentile_service_dspop_recharge_PH_conservation_inf_ssp245.tif',
    'top_25th_percentile_service_dspop_recharge_PH_restoration': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/top_25th_percentile_service_dspop_recharge_PH_restoration.tif',
    'top_25th_percentile_service_dspop_recharge_PH_restoration_ssp245': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/top_25th_percentile_service_dspop_recharge_PH_restoration_ssp245.tif',
    'top_25th_percentile_service_dspop_sediment_IDN_conservation_inf': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/top_25th_percentile_service_dspop_sediment_IDN_conservation_inf.tif',
    'top_25th_percentile_service_dspop_sediment_IDN_conservation_inf_ssp245': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/top_25th_percentile_service_dspop_sediment_IDN_conservation_inf_ssp245.tif',
    'top_25th_percentile_service_dspop_sediment_IDN_restoration': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/top_25th_percentile_service_dspop_sediment_IDN_restoration.tif',
    'top_25th_percentile_service_dspop_sediment_IDN_restoration_ssp245': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/top_25th_percentile_service_dspop_sediment_IDN_restoration_ssp245.tif',
    'top_25th_percentile_service_dspop_sediment_PH_conservation_inf': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/top_25th_percentile_service_dspop_sediment_PH_conservation_inf.tif',
    'top_25th_percentile_service_dspop_sediment_PH_conservation_inf_ssp245': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/top_25th_percentile_service_dspop_sediment_PH_conservation_inf_ssp245.tif',
    'top_25th_percentile_service_dspop_sediment_PH_restoration': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/top_25th_percentile_service_dspop_sediment_PH_restoration.tif',
    'top_25th_percentile_service_dspop_sediment_PH_restoration_ssp245': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/top_25th_percentile_service_dspop_sediment_PH_restoration_ssp245.tif',
    'top_25th_percentile_service_road_flood_mitigation_IDN_conservation_inf': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/top_25th_percentile_service_road_flood_mitigation_IDN_conservation_inf.tif',
    'top_25th_percentile_service_road_flood_mitigation_IDN_conservation_inf_ssp245': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/top_25th_percentile_service_road_flood_mitigation_IDN_conservation_inf_ssp245.tif',
    'top_25th_percentile_service_road_flood_mitigation_IDN_restoration': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/top_25th_percentile_service_road_flood_mitigation_IDN_restoration.tif',
    'top_25th_percentile_service_road_flood_mitigation_IDN_restoration_ssp245': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/top_25th_percentile_service_road_flood_mitigation_IDN_restoration_ssp245.tif',
    'top_25th_percentile_service_road_flood_mitigation_PH_conservation_inf': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/top_25th_percentile_service_road_flood_mitigation_PH_conservation_inf.tif',
    'top_25th_percentile_service_road_flood_mitigation_PH_conservation_inf_ssp245': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/top_25th_percentile_service_road_flood_mitigation_PH_conservation_inf_ssp245.tif',
    'top_25th_percentile_service_road_flood_mitigation_PH_restoration': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/top_25th_percentile_service_road_flood_mitigation_PH_restoration.tif',
    'top_25th_percentile_service_road_flood_mitigation_PH_restoration_ssp245': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/top_25th_percentile_service_road_flood_mitigation_PH_restoration_ssp245.tif',
    'top_25th_percentile_service_road_recharge_IDN_conservation_inf': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/top_25th_percentile_service_road_recharge_IDN_conservation_inf.tif',
    'top_25th_percentile_service_road_recharge_IDN_conservation_inf_ssp245': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/top_25th_percentile_service_road_recharge_IDN_conservation_inf_ssp245.tif',
    'top_25th_percentile_service_road_recharge_IDN_restoration': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/top_25th_percentile_service_road_recharge_IDN_restoration.tif',
    'top_25th_percentile_service_road_recharge_IDN_restoration_ssp245': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/top_25th_percentile_service_road_recharge_IDN_restoration_ssp245.tif',
    'top_25th_percentile_service_road_recharge_PH_conservation_inf': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/top_25th_percentile_service_road_recharge_PH_conservation_inf.tif',
    'top_25th_percentile_service_road_recharge_PH_conservation_inf_ssp245': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/top_25th_percentile_service_road_recharge_PH_conservation_inf_ssp245.tif',
    'top_25th_percentile_service_road_recharge_PH_restoration': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/top_25th_percentile_service_road_recharge_PH_restoration.tif',
    'top_25th_percentile_service_road_recharge_PH_restoration_ssp245': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/top_25th_percentile_service_road_recharge_PH_restoration_ssp245.tif',
    'top_25th_percentile_service_road_sediment_IDN_conservation_inf': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/top_25th_percentile_service_road_sediment_IDN_conservation_inf.tif',
    'top_25th_percentile_service_road_sediment_IDN_conservation_inf_ssp245': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/top_25th_percentile_service_road_sediment_IDN_conservation_inf_ssp245.tif',
    'top_25th_percentile_service_road_sediment_IDN_restoration': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/top_25th_percentile_service_road_sediment_IDN_restoration.tif',
    'top_25th_percentile_service_road_sediment_IDN_restoration_ssp245': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/top_25th_percentile_service_road_sediment_IDN_restoration_ssp245.tif',
    'top_25th_percentile_service_road_sediment_PH_conservation_inf': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/top_25th_percentile_service_road_sediment_PH_conservation_inf.tif',
    'top_25th_percentile_service_road_sediment_PH_conservation_inf_ssp245': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/top_25th_percentile_service_road_sediment_PH_conservation_inf_ssp245.tif',
    'top_25th_percentile_service_road_sediment_PH_restoration': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/top_25th_percentile_service_road_sediment_PH_restoration.tif',
    'top_25th_percentile_service_road_sediment_PH_restoration_ssp245': 'gs://ecoshard-root/wwf_sipa_final_viewer_cogs/top_25th_percentile_service_road_sediment_PH_restoration_ssp245.tif'
};


var legend_styles = {
  'black_to_red': ['000000', '005aff', '43c8c8', 'fff700', 'ff0000'],
  'blue_to_green': ['440154', '414287', '218e8d', '5ac864', 'fde725'],
  'cividis': ['00204d', '414d6b', '7c7b78', 'b9ac70', 'ffea46'],
  'viridis': ['440154', '355e8d', '20928c', '70cf57', 'fde725'],
  'blues': ['f7fbff', 'c6dbef', '6baed6', '2171b5', '08306b'],
  'reds': ['fff5f0', 'fcbba1', 'fb6a4a', 'cb181d', '67000d'],
  'turbo': ['321543', '2eb4f2', 'affa37', 'f66c19', '7a0403'],
};
var default_legend_style = 'blue_to_green';

function changeColorScheme(key, active_context) {
  active_context.visParams.palette = legend_styles[key];
  active_context.build_legend_panel();
  active_context.updateVisParams();
}

function updateRasterLayer(key, active_context) {
  if (key !== null) {
    active_context.raster = ee.Image.loadGeoTIFF(datasets[key]);
    if (active_context.aoi_polygon != null) {
      active_context.raster = active_context.raster.clipToCollection(active_context.aoi_polygon);
    }
    active_context.last_layer = active_context.map.addLayer(
      active_context.raster, active_context.visParams);
  }
}

var linkedMap = ui.Map();
// Center on Philippines
Map.setCenter(122.695894, 4, 5);
var linker = ui.Map.Linker([ui.root.widgets().get(0), linkedMap]);
// Create a SplitPanel which holds the linked maps side-by-side.
var splitPanel = ui.SplitPanel({
  firstPanel: linker.get(0),
  secondPanel: linker.get(1),
  orientation: 'horizontal',
  wipe: true,
  style: {stretch: 'both'}
});
ui.root.widgets().reset([splitPanel]);

var panel_list = [];
[[Map, 'left'], [linkedMap, 'right']].forEach(function(mapside, index) {
    var active_context = {
      'last_layer': null,
      'last_aoi': null,
      'raster': null,
      'point_val': null,
      'last_point_layer': null,
      'map': mapside[0],
      'legend_panel': null,
      'visParams': null,
      'raster_key': null,
      'polygon_key': null,
    };

    active_context.map.style().set('cursor', 'crosshair');
    active_context.visParams = {
      min: 0.0,
      max: 100.0,
      palette: legend_styles[default_legend_style],
    };

    var panel = ui.Panel({
      layout: ui.Panel.Layout.flow('vertical'),
      style: {
        'position': "middle-"+mapside[1],
        'backgroundColor': 'rgba(255, 255, 255, 0.4)'
      }
    });

    var default_control_text = mapside[1]+' controls';
    var controls_label = ui.Label({
      value: default_control_text,
      style: {
        backgroundColor: 'rgba(0, 0, 0, 0)',
      }
    });
    var select = ui.Select({
      items: Object.keys(datasets),
      onChange: function(raster_key, self) {
          active_context.raster_key = raster_key;
          self.setDisabled(true);
          var original_value = self.getValue();
          self.setPlaceholder('loading ...');
          self.setValue(null, false);
          if (active_context.last_layer !== null) {
            active_context.map.remove(active_context.last_layer);
            active_context.min_val.setDisabled(true);
            active_context.max_val.setDisabled(true);
          }
          if (datasets[raster_key] == '') {
            self.setValue(original_value, false);
            self.setDisabled(false);
            return
          }

          updateRasterLayer(raster_key, active_context);
          var mean_reducer = ee.Reducer.percentile([10, 90], ['p10', 'p90']);
          var meanDictionary = active_context.raster.reduceRegion({
            reducer: mean_reducer,
            geometry: active_context.map.getBounds(true),
            bestEffort: true,
          });

          ee.data.computeValue(meanDictionary, function (val) {
            if (val['B0_p10'] != val['B0_p90']) {
              active_context.visParams = {
                min: val['B0_p10'],
                max: val['B0_p90'],
                palette: active_context.visParams.palette,
              };
            } else {
              active_context.visParams = {
                min: 0,
                max: val['B0_p90'],
                palette: active_context.visParams.palette,
              };
            }
            active_context.min_val.setValue(active_context.visParams.min, false);
            active_context.max_val.setValue(active_context.visParams.max, false);
            active_context.min_val.setDisabled(false);
            active_context.max_val.setDisabled(false);
            self.setValue(original_value, false);
            self.setDisabled(false);
          });
      }
    });

    var polygon_aggregate = ui.Select({
        items: Object.keys(aggregate_polygons),
      onChange: function(polygon_key, self) {
          self.setDisabled(true);
          var original_value = self.getValue();
          self.setPlaceholder('loading ...');
          self.setValue(null, false);
          if (active_context.last_aoi !== null) {
            active_context.map.remove(active_context.last_aoi);
          }
          if (active_context.last_layer !== null) {
            active_context.map.remove(active_context.last_layer);
          }
          if (aggregate_polygons[polygon_key] == '') {
            // nothing selected
            self.setDisabled(false);
            return
          }

          //TODO: in this spot remove the layer and clip it with the new polygon
          active_context.aoi_polygon = ee.FeatureCollection(aggregate_polygons[polygon_key]);
          updateRasterLayer(active_context.raster_key, active_context);

          // Add the feature to the map with styling
          active_context.last_aoi = active_context.map.addLayer(
            active_context.aoi_polygon.style(default_vector_style));

          self.setValue(original_value, false);
          self.setDisabled(false);
      }
    });

    active_context.min_val = ui.Textbox(
      0, 0, function (value) {
        active_context.visParams.min = +(value);
        updateVisParams();
      });
    active_context.min_val.setDisabled(true);

    active_context.max_val = ui.Textbox(
      100, 100, function (value) {
        active_context.visParams.max = +(value);
        updateVisParams();
      });
    active_context.max_val.setDisabled(true);

    active_context.point_val = ui.Textbox('nothing clicked');
    function updateVisParams() {
      if (active_context.last_layer !== null) {
        active_context.last_layer.setVisParams(active_context.visParams);
      }
    }
    active_context.updateVisParams = updateVisParams;
    select.setPlaceholder('Choose a dataset...');
    var range_button = ui.Button(
      'Detect Range', function (self) {
        self.setDisabled(true);
        var base_label = self.getLabel();
        self.setLabel('Detecting...');
        var mean_reducer = ee.Reducer.percentile([10, 90], ['p10', 'p90']);
        var meanDictionary = active_context.raster.reduceRegion({
          reducer: mean_reducer,
          geometry: active_context.map.getBounds(true),
          bestEffort: true,
        });
        ee.data.computeValue(meanDictionary, function (val) {
          active_context.min_val.setValue(val['B0_p10'], false);
          active_context.max_val.setValue(val['B0_p90'], true);
          self.setLabel(base_label)
          self.setDisabled(false);
        });
      });

    panel.add(controls_label);
    panel.add(select);
    polygon_aggregate.setPlaceholder('Filter through Administrative Levels');
    panel.add(polygon_aggregate);
    panel.add(ui.Label({
        value: 'min',
        style:{'backgroundColor': 'rgba(0, 0, 0, 0)'}
      }));
    panel.add(active_context.min_val);
    panel.add(ui.Label({
        value: 'max',
        style:{'backgroundColor': 'rgba(0, 0, 0, 0)'}
      }));
    panel.add(active_context.max_val);
    panel.add(range_button);
    panel.add(ui.Label({
      value: 'picked point',
      style: {'backgroundColor': 'rgba(0, 0, 0, 0)'}
    }));
    panel.add(active_context.point_val);
    panel_list.push([panel, active_context.min_val, active_context.max_val, active_context]);
    active_context.map.add(panel);

    function build_legend_panel() {
      var makeRow = function(color, name) {
        var colorBox = ui.Label({
          style: {
            backgroundColor: '#' + color,
            padding: '4px 25px 4px 25px',
            margin: '0 0 0px 0',
            position: 'bottom-center',
          }
        });
        var description = ui.Label({
          value: name,
          style: {
            margin: '0 0 0px 0px',
            position: 'top-center',
            fontSize: '10px',
            padding: 0,
            border: 0,
            textAlign: 'center',
            backgroundColor: 'rgba(0, 0, 0, 0)',
          }
        });

        return ui.Panel({
          widgets: [colorBox, description],
          layout: ui.Panel.Layout.Flow('vertical'),
          style: {
            backgroundColor: 'rgba(0, 0, 0, 0)',
          }
        });
      };

      var names = ['Low', '', '', '', 'High'];
      if (active_context.legend_panel !== null) {
        active_context.legend_panel.clear();
      } else {
        active_context.legend_panel = ui.Panel({
          layout: ui.Panel.Layout.Flow('horizontal'),
          style: {
            position: 'top-center',
            padding: '0px',
            backgroundColor: 'rgba(255, 255, 255, 0.4)'
          }
        });
        active_context.legend_select = ui.Select({
          items: Object.keys(legend_styles),
          placeholder: default_legend_style,
          onChange: function(key, self) {
            changeColorScheme(key, active_context);
        }});
        active_context.map.add(active_context.legend_panel);
      }
      active_context.legend_panel.add(active_context.legend_select);
      // Add color and and names
      for (var i = 0; i<5; i++) {
        var row = makeRow(active_context.visParams.palette[i], names[i]);
        active_context.legend_panel.add(row);
      }
    }

    active_context.map.setControlVisibility(false);
    active_context.map.setControlVisibility({"mapTypeControl": true});
    build_legend_panel();
    active_context.build_legend_panel = build_legend_panel;
});

var clone_to_right = ui.Button(
  'Use this range in both windows', function () {
      panel_list[1][1].setValue(panel_list[0][1].getValue(), false)
      panel_list[1][2].setValue(panel_list[0][2].getValue(), true)
});
var clone_to_left = ui.Button(
  'Use this range in both windows', function () {
      panel_list[0][1].setValue(panel_list[1][1].getValue(), false)
      panel_list[0][2].setValue(panel_list[1][2].getValue(), true)
});

//panel_list.push([panel, min_val, max_val, map, active_context]);
panel_list.forEach(function (panel_array) {
  var map = panel_array[3].map;
  map.onClick(function (obj) {
    var point = ee.Geometry.Point([obj.lon, obj.lat]);
    [panel_list[0][3], panel_list[1][3]].forEach(function (active_context) {
      if (active_context.last_layer !== null) {
        active_context.point_val.setValue('sampling...')
        var point_sample = active_context.raster.sampleRegions({
          collection: point,
          //scale: 10,
          //geometries: true
        });
        ee.data.computeValue(point_sample, function (val) {
          if (val.features.length > 0) {
            active_context.point_val.setValue(val.features[0].properties.B0.toString());
            if (active_context.last_point_layer !== null) {
              active_context.map.remove(active_context.last_point_layer);
            }
            active_context.last_point_layer = active_context.map.addLayer(
              point, {'color': '#FF00FF'});
          } else {
            active_context.point_val.setValue('nodata');
          }
        });
      }
    });
  });
});

panel_list[0][0].add(clone_to_right);
panel_list[1][0].add(clone_to_left);

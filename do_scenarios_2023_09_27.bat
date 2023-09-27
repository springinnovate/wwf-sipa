:: * convert base landcover to C factor
::
:: * create base mask that shouldn't be converted
::     python ../spring/raster_calculations/create_mask_from_values.py data/landcover_rasters/ph_clip.tif --value_list 3 7 --target_raster_path data/landcover_rasters/ph_clip_unchanged_land.tif --invert
::
:: * apply continuous change scenario to that C factor
::     python continuous_change_scenario_from_infra.py data/landcover_rasters/ph_clip_c_factor.tif data/PH_infrastructure_future_continuous_c_factor_clip_test.csv --convert_mask_path data/landcover_rasters/ph_clip_unchanged_land.tif
::
::     python continuous_change_scenario_from_infra.py data/landcover_rasters/ph_clip_c_factor.tif data/PH_infrastructure_future_continuous_c_factor_clip_test.csv
::
::
::
:: And reminder (to myself as much as to you because I had already forgotten!), when we remake the scenarios and rerun the models USE THE NEW INDONESIA LULC!!! "D:\repositories\wwf-sipa\data\landcover_rasters\ID_LULC_restoration_all_islands_corr_md5_a66e06.tif" -- I guess that's a step 2.5 is create new parameter rasters using "D:\repositories\raster_calculations\reclassify_by_table.py" with that new IDN lulc

:: create new parameter rasters using reclassify_by_table ->

CALL python ../spring/raster_calculations/reclassify_by_table.py "D:\repositories\wwf-sipa\data\landcover_rasters\PH_restoration_compressed_md5_41d7ea.tif" "D:\repositories\wwf-sipa\data\biophysical_template_PH_revised.csv" lulc_id affected_by_infra,int usle_c,float CN_A,float CN_B,float CN_C,float CN_D,float root_depth,float Kc_1,float Kc_2,float Kc_3,float Kc_4,float Kc_5,float Kc_6,float Kc_7,float Kc_8,float Kc_9,float Kc_10,float Kc_11,float Kc_12,float  --output_dir data/parameter_rasters
CALL python ../spring/raster_calculations/reclassify_by_table.py "D:\repositories\wwf-sipa\data\landcover_rasters\ID_LULC_restoration_all_islands_corr_md5_a66e06.tif" "D:\repositories\wwf-sipa\data\biophysical_template_IDN_revised.csv" lulc_id affected_by_infra,int usle_c,float CN_A,float CN_B,float CN_C,float CN_D,float root_depth,float Kc_1,float Kc_2,float Kc_3,float Kc_4,float Kc_5,float Kc_6,float Kc_7,float Kc_8,float Kc_9,float Kc_10,float Kc_11,float Kc_12,float  --output_dir data/parameter_rasters

:: PH future
:: PH scenario
:: IDN future
:: IDN scenario


::python continuous_change_scenario_from_infra.py
::
::"D:\repositories\wwf-sipa\data\IDN_infrastructure_future_continuous_c_factor.csv"
::"D:\repositories\wwf-sipa\data\IDN_infrastructure_future_continuous_CN_A.csv"
::"D:\repositories\wwf-sipa\data\IDN_infrastructure_future_continuous_CN_B.csv"
::"D:\repositories\wwf-sipa\data\IDN_infrastructure_future_continuous_CN_C.csv"
::"D:\repositories\wwf-sipa\data\IDN_infrastructure_future_continuous_CN_D.csv"
::"D:\repositories\wwf-sipa\data\IDN_infrastructure_scenario_continuous_c_factor.csv"
::"D:\repositories\wwf-sipa\data\IDN_infrastructure_scenario_continuous_CN_A.csv"
::"D:\repositories\wwf-sipa\data\IDN_infrastructure_scenario_continuous_CN_B.csv"
::"D:\repositories\wwf-sipa\data\IDN_infrastructure_scenario_continuous_CN_C.csv"
::"D:\repositories\wwf-sipa\data\IDN_infrastructure_scenario_continuous_CN_D.csv"
::
::"D:\repositories\wwf-sipa\data\PH_infrastructure_future_continuous_c_factor.csv"
::"D:\repositories\wwf-sipa\data\PH_infrastructure_future_continuous_CN_A.csv"
::"D:\repositories\wwf-sipa\data\PH_infrastructure_future_continuous_CN_B.csv"
::"D:\repositories\wwf-sipa\data\PH_infrastructure_future_continuous_CN_C.csv"
::"D:\repositories\wwf-sipa\data\PH_infrastructure_future_continuous_CN_D.csv"
::"D:\repositories\wwf-sipa\data\PH_infrastructure_scenario_continuous_c_factor.csv"
::"D:\repositories\wwf-sipa\data\PH_infrastructure_scenario_continuous_CN_A.csv"
::"D:\repositories\wwf-sipa\data\PH_infrastructure_scenario_continuous_CN_B.csv"
::"D:\repositories\wwf-sipa\data\PH_infrastructure_scenario_continuous_CN_C.csv"
::"D:\repositories\wwf-sipa\data\PH_infrastructure_scenario_continuous_CN_D.csv"

:: apply continuous change to c_factor, CN_A, CN_B, CN_C, CN_D for all 4 possibilities
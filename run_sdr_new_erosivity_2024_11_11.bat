@echo off
:: run on 11/11/2024 817am, this is the pipeline to run "worst case", fixes issue with overview corruption, and runs watersheds of all sizes
set ORIGINAL_DIR=%cd%

cd /d D:\repositories\ndr_sdr_global

call python run_ndr_sdr_pipeline.py .\pipeline_config_files\wwf_IDN_baseline_historical_climate.ini > log_wwf_IDN_baseline_historical_climate_2024_11_11.txt
call python run_ndr_sdr_pipeline.py .\pipeline_config_files\wwf_IDN_baseline_ssp245_climate.ini > log_wwf_IDN_baseline_ssp245_climate_2024_11_11.txt
call python run_ndr_sdr_pipeline.py .\pipeline_config_files\wwf_IDN_infra_historical_climate.ini > log_wwf_IDN_infra_historical_climate_2024_11_11.txt
call python run_ndr_sdr_pipeline.py .\pipeline_config_files\wwf_IDN_infra_ssp245_climate.ini > log_wwf_IDN_infra_ssp245_climate_2024_11_11.txt
call python run_ndr_sdr_pipeline.py .\pipeline_config_files\wwf_IDN_restoration_historical_climate.ini > log_wwf_IDN_restoration_historical_climate_2024_11_11.txt
call python run_ndr_sdr_pipeline.py .\pipeline_config_files\wwf_IDN_restoration_ssp245_climate.ini > log_wwf_IDN_restoration_ssp245_climate_2024_11_11.txt
call python run_ndr_sdr_pipeline.py .\pipeline_config_files\wwf_IDN_worstcase_historical_climate.ini > log_wwf_IDN_worstcase_historical_climate_2024_11_11.txt
call python run_ndr_sdr_pipeline.py .\pipeline_config_files\wwf_IDN_worstcase_ssp245_climate.ini > log_wwf_IDN_worstcase_ssp245_climate_2024_11_11.txt
call python run_ndr_sdr_pipeline.py .\pipeline_config_files\wwf_PH_baseline_historical_climate.ini > log_wwf_PH_baseline_historical_climate_2024_11_11.txt
call python run_ndr_sdr_pipeline.py .\pipeline_config_files\wwf_PH_baseline_ssp245_climate.ini > log_wwf_PH_baseline_ssp245_climate_2024_11_11.txt
call python run_ndr_sdr_pipeline.py .\pipeline_config_files\wwf_PH_infra_historical_climate.ini > log_wwf_PH_infra_historical_climate_2024_11_11.txt
call python run_ndr_sdr_pipeline.py .\pipeline_config_files\wwf_PH_infra_ssp245_climate.ini > log_wwf_PH_infra_ssp245_climate_2024_11_11.txt
call python run_ndr_sdr_pipeline.py .\pipeline_config_files\wwf_PH_restoration_historical_climate.ini > log_wwf_PH_restoration_historical_climate_2024_11_11.txt
call python run_ndr_sdr_pipeline.py .\pipeline_config_files\wwf_PH_restoration_ssp245_climate.ini > log_wwf_PH_restoration_ssp245_climate_2024_11_11.txt
call python run_ndr_sdr_pipeline.py .\pipeline_config_files\wwf_PH_worstcase_historical_climate.ini > log_wwf_PH_worstcase_historical_climate_2024_11_11.txt
call python run_ndr_sdr_pipeline.py .\pipeline_config_files\wwf_PH_worstcase_ssp245_climate.ini > log_wwf_PH_worstcase_ssp245_climate_2024_11_11.txt

:: Is there something like the following that should be run for SWY?
rem cd /d D:\repositories\swy_global
rem call python run_swy_global.py pipeline_config_files\wwf_IDN_worstcase_historical_climate.ini > wwf_IDN_worstcase_historical_climate_log.txt
rem call python run_swy_global.py pipeline_config_files\wwf_PH_worstcase_historical_climate.ini > wwf_PH_worstcase_historical_climate_log.txt
:: call python D:\repositories\swy_global\run_swy_global.py D:\repositories\swy_global\pipeline_config_files\wwf_IDN_baseline_historical_climate.ini > D:\repositories\swy_global\wwf_IDN_baseline_historical_climate_log.txt
:: call python D:\repositories\swy_global\run_swy_global.py D:\repositories\swy_global\pipeline_config_files\wwf_IDN_baseline_ssp245_climate10.ini > D:\repositories\swy_global\wwf_IDN_baseline_ssp245_climate10_log.txt
:: call python D:\repositories\swy_global\run_swy_global.py D:\repositories\swy_global\pipeline_config_files\wwf_IDN_baseline_ssp245_climate90.ini > D:\repositories\swy_global\wwf_IDN_baseline_ssp245_climate90_log.txt
:: call python D:\repositories\swy_global\run_swy_global.py D:\repositories\swy_global\pipeline_config_files\wwf_IDN_infra_historical_climate.ini > D:\repositories\swy_global\wwf_IDN_infra_historical_climate_log.txt
:: call python D:\repositories\swy_global\run_swy_global.py D:\repositories\swy_global\pipeline_config_files\wwf_IDN_infra_ssp245_climate10.ini > D:\repositories\swy_global\wwf_IDN_infra_ssp245_climate10_log.txt
:: call python D:\repositories\swy_global\run_swy_global.py D:\repositories\swy_global\pipeline_config_files\wwf_IDN_infra_ssp245_climate90.ini > D:\repositories\swy_global\wwf_IDN_infra_ssp245_climate90_log.txt
:: call python D:\repositories\swy_global\run_swy_global.py D:\repositories\swy_global\pipeline_config_files\wwf_IDN_restoration_historical_climate.ini > D:\repositories\swy_global\wwf_IDN_restoration_historical_climate_log.txt
:: call python D:\repositories\swy_global\run_swy_global.py D:\repositories\swy_global\pipeline_config_files\wwf_IDN_restoration_ssp245_climate10.ini > D:\repositories\swy_global\wwf_IDN_restoration_ssp245_climate10_log.txt
:: call python D:\repositories\swy_global\run_swy_global.py D:\repositories\swy_global\pipeline_config_files\wwf_IDN_restoration_ssp245_climate90.ini > D:\repositories\swy_global\wwf_IDN_restoration_ssp245_climate90_log.txt
:: call python D:\repositories\swy_global\run_swy_global.py D:\repositories\swy_global\pipeline_config_files\wwf_PH_baseline_historical_climate.ini > D:\repositories\swy_global\wwf_PH_baseline_historical_climate_log.txt
:: call python D:\repositories\swy_global\run_swy_global.py D:\repositories\swy_global\pipeline_config_files\wwf_PH_baseline_ssp245_climate10.ini > D:\repositories\swy_global\wwf_PH_baseline_ssp245_climate10_log.txt
:: call python D:\repositories\swy_global\run_swy_global.py D:\repositories\swy_global\pipeline_config_files\wwf_PH_baseline_ssp245_climate90.ini > D:\repositories\swy_global\wwf_PH_baseline_ssp245_climate90_log.txt
:: call python D:\repositories\swy_global\run_swy_global.py D:\repositories\swy_global\pipeline_config_files\wwf_PH_infra_historical_climate.ini > D:\repositories\swy_global\wwf_PH_infra_historical_climate_log.txt
:: call python D:\repositories\swy_global\run_swy_global.py D:\repositories\swy_global\pipeline_config_files\wwf_PH_infra_ssp245_climate10.ini > D:\repositories\swy_global\wwf_PH_infra_ssp245_climate10_log.txt
:: call python D:\repositories\swy_global\run_swy_global.py D:\repositories\swy_global\pipeline_config_files\wwf_PH_infra_ssp245_climate90.ini > D:\repositories\swy_global\wwf_PH_infra_ssp245_climate90_log.txt
:: call python D:\repositories\swy_global\run_swy_global.py D:\repositories\swy_global\pipeline_config_files\wwf_PH_restoration_historical_climate.ini > D:\repositories\swy_global\wwf_PH_restoration_historical_climate_log.txt
:: call python D:\repositories\swy_global\run_swy_global.py D:\repositories\swy_global\pipeline_config_files\wwf_PH_restoration_ssp245_climate10.ini > D:\repositories\swy_global\wwf_PH_restoration_ssp245_climate10_log.txt
:: call python D:\repositories\swy_global\run_swy_global.py D:\repositories\swy_global\pipeline_config_files\wwf_PH_restoration_ssp245_climate90.ini > D:\repositories\swy_global\wwf_PH_restoration_ssp245_climate90_log.txt

cd /d %ORIGINAL_DIR%

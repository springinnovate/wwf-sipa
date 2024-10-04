@echo off
set ORIGINAL_DIR=%cd%

cd /d D:\repositories\ndr_sdr_global
call python run_ndr_sdr_pipeline.py pipeline_config_files\wwf_IDN_worstcase_historical_climate.ini
call python run_ndr_sdr_pipeline.py pipeline_config_files\wwf_PH_worstcase_historical_climate.ini

cd /d D:\repositories\swy_global
call python run_swy_global.py pipeline_config_files\wwf_IDN_worstcase_historical_climate.ini > wwf_IDN_worstcase_historical_climate_log.txt
call python run_swy_global.py pipeline_config_files\wwf_PH_worstcase_historical_climate.ini > wwf_PH_worstcase_historical_climate_log.txt
cd /d %ORIGINAL_DIR%

rem call python D:\repositories\ndr_sdr_global\run_ndr_sdr_pipeline.py D:\repositories\ndr_sdr_global\pipeline_config_files\wwf_IDN_worstcase_historical_climate.ini
rem call python D:\repositories\ndr_sdr_global\run_ndr_sdr_pipeline.py D:\repositories\ndr_sdr_global\pipeline_config_files\wwf_PH_worstcase_historical_climate.ini
rem call python D:\repositories\swy_global\run_swy_global.py D:\repositories\swy_global\pipeline_config_files\wwf_IDN_worstcase_historical_climate.ini > D:\repositories\swy_global\wwf_IDN_worstcase_historical_climate_log.txt
rem call python D:\repositories\swy_global\run_swy_global.py D:\repositories\swy_global\pipeline_config_files\wwf_PH_worstcase_historical_climate.ini > D:\repositories\swy_global\wwf_PH_worstcase_historical_climate_log.txt

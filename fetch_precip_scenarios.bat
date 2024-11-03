set delayedexpansion true

set startYear=1985
set endYear=2014
set scenario=historical
set eventThreshold=10
set datasetScale=1000

REM note that i only have the percentiles set to 50 for a short run, should be 10 50 90 for the full run

FOR %%A IN (PH_large_aoi.gpkg IDN_large_aoi.gpkg) DO (
    call python precip_calculator.py --aoi_vector_path D:/repositories/wwf-sipa/province_dependence_workspace/%%A --scenario_id %scenario% --date_range %startYear% %endYear% --percentile 50 --dataset_scale %datasetScale%
    call python n_events_calculator.py --aoi_vector_path D:/repositories/wwf-sipa/province_dependence_workspace/%%A --scenario_id %scenario% --date_range %startYear% %endYear% --threshold %eventThreshold% --percentile 50 --dataset_scale %datasetScale%
)

set startYear=2040
set endYear=2060
set scenario=ssp245

FOR %%A IN (PH_large_aoi.gpkg IDN_large_aoi.gpkg) DO (
    call python precip_calculator.py --aoi_vector_path D:/repositories/wwf-sipa/province_dependence_workspace/%%A --scenario_id %scenario% --date_range %startYear% %endYear% --percentile 50 --dataset_scale %datasetScale%
    call python n_events_calculator.py --aoi_vector_path D:/repositories/wwf-sipa/province_dependence_workspace/%%A --scenario_id %scenario% --date_range %startYear% %endYear% --threshold %eventThreshold% --percentile 50 --dataset_scale %datasetScale%
)

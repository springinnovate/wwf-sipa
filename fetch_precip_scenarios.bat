set delayedexpansion true

REM set startYear=2040
REM set endYear=2060
REM set scenario=ssp245
set startYear=1985
set endYear=2014
set scenario=historical
set eventThreshold=10
set datasetScale=1000

FOR %%A IN (PH_outline.gpkg IDN_outline.gpkg) DO (
    ECHO call python precip_calculator.py --aoi_vector_path ./data/admin_boundaries/%%A --scenario_id %scenario% --date_range %startYear% %endYear% --percentile 10 50 90 --dataset_scale datasetScale
    ECHO call python n_events_calculator.py --aoi_vector_path ./data/admin_boundaries/%%A --scenario_id %scenario% --date_range %startYear% %endYear% --threshold %eventThreshold% --percentile 10 50 90 --dataset_scale datasetScale
)

set startYear=2040
set endYear=2060
set scenario=ssp245

FOR %%A IN (PH_outline.gpkg IDN_outline.gpkg) DO (
    ECHO call python precip_calculator.py --aoi_vector_path ./data/admin_boundaries/%%A --scenario_id %scenario% --date_range %startYear% %endYear% --percentile 10 50 90 --dataset_scale datasetScale
    ECHO call python n_events_calculator.py --aoi_vector_path ./data/admin_boundaries/%%A --scenario_id %scenario% --date_range %startYear% %endYear% --threshold %eventThreshold% --percentile 10 50 90 --dataset_scale datasetScale
)

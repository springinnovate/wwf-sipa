set delayedexpansion true
set startYear=2035
set endYear=2035
set scenario=ssp245
set eventThreshold=10

REM call python precip_calculator.py --aoi_vector_path ./data/admin_boundaries/PH_outline.gpkg --scenario_id %scenario% --date_range %startYear% %endYear% --percentile 10 50 90
call python n_events_calculator.py --aoi_vector_path ./data/admin_boundaries/PH_outline.gpkg --scenario_id %scenario% --date_range %startYear% %endYear% --threshold %eventThreshold% --percentile 10 50 90

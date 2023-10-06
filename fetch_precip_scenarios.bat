set delayedexpansion true

FOR %%A IN (PH_outline.gpkg IDN_outline.gpkg) DO (
    set startYear=2040
    set endYear=2060
    set scenario=ssp245
    set eventThreshold=10

    call python precip_calculator.py --aoi_vector_path ./data/admin_boundaries/%%A --scenario_id %scenario% --date_range %startYear% %endYear% --percentile 10 50 90 --dataset_scale 1000
    call python n_events_calculator.py --aoi_vector_path ./data/admin_boundaries/%%A --scenario_id %scenario% --date_range %startYear% %endYear% --threshold %eventThreshold% --percentile 10 50 90 --dataset_scale 1000

    set startYear=1985
    set endYear=2014
    set scenario=historical
    set eventThreshold=10

    call python precip_calculator.py --aoi_vector_path ./data/admin_boundaries/%%A --scenario_id %scenario% --date_range %startYear% %endYear% --percentile 10 50 90 --dataset_scale 1000
    call python n_events_calculator.py --aoi_vector_path ./data/admin_boundaries/%%A --scenario_id %scenario% --date_range %startYear% %endYear% --threshold %eventThreshold% --percentile 10 50 90 --dataset_scale 1000
)

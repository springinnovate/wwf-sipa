SIPA Processing pipeline

Preprocessing
-------------
* `rasterize_roads.py` - used to rasterize line road geometry into buffered regions onto a raster
* `rasterize_landuse_polygons.py` - used to originally rasterize the landuse polygons we got from Indonesia


Data generators for scenarios
-----------------------------
* `erosivity_calculator_cmip6.py` - used to generate monthly and annual erosivity rasters from CMIP6 NEX dataset.
* `precip_cmip6.py` - used to create monthly and annual precipitation rasters from CMIP6 dataset.
* `precip_events_cmip6.py` - used to calculate monthly precip events from CMIP6 dataset.


Analysis
--------
* `downstream_beneficiaries_aggregator.py` - used in conjunction with `downstream_*.ini` files to calculate the count of the beneficiaries downstream from a mask such as priority areas.

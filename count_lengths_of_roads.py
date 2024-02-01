import os
import sys

from ecoshard import geoprocessing
from osgeo import gdal
from osgeo import ogr
from osgeo import osr
import geopandas as gpd


WORK = {
    'IDN': {
        'road_vector_path': r"D:\repositories\wwf-sipa\data\infrastructure_polygons\IDN_All_Roads_Merged.gpkg",
        'conservation_rasters': [
            r"D:\repositories\wwf-sipa\workspace_postprocess_number_of_people_benefitting\working_dir\coastal_benefit_areas_idn_dem_10_IDN_conservation_inf_service_overlap_count.tif",
            r"D:\repositories\wwf-sipa\workspace_postprocess_number_of_people_benefitting\working_dir\10_IDN_conservation_inf_service_overlap_count_downstream_mask.tif",
            ],
        'restoration_rasters': [
            r"D:\repositories\wwf-sipa\workspace_postprocess_number_of_people_benefitting\working_dir\coastal_benefit_areas_idn_dem_10_IDN_restoration_service_overlap_count.tif",
            r"D:\repositories\wwf-sipa\workspace_postprocess_number_of_people_benefitting\working_dir\10_IDN_restoration_service_overlap_count_downstream_mask.tif",
            ],
    },
    'PH': {
        'road_vector_path': r"D:\repositories\wwf-sipa\data\infrastructure_polygons\PH_All_Roads_Merged.gpkg",
        'conservation_rasters': [
            r"D:\repositories\wwf-sipa\workspace_postprocess_number_of_people_benefitting\working_dir\coastal_benefit_areas_ph_dem_10_PH_conservation_inf_service_overlap_count.tif",
            r"D:\repositories\wwf-sipa\workspace_postprocess_number_of_people_benefitting\working_dir\10_PH_conservation_inf_service_overlap_count_downstream_mask.tif",
            ],
        'restoration_rasters': [
            r"D:\repositories\wwf-sipa\workspace_postprocess_number_of_people_benefitting\working_dir\coastal_benefit_areas_ph_dem_10_PH_restoration_service_overlap_count.tif",
            r"D:\repositories\wwf-sipa\workspace_postprocess_number_of_people_benefitting\working_dir\10_PH_restoration_service_overlap_count_downstream_mask.tif",
            ],
    },
}

WORKING_DIR = 'road_length_workspace'
os.makedirs(WORKING_DIR, exist_ok=True)


def _merge_masks_op(a, b):
    return (a >= 1) | (b >= 1)


def progress_callback(complete, message, _):
    """
    A callback function that displays the progress.

    Parameters:
    - complete: A float representing the percentage of the task completed.
    - message: A string message (not always provided).
    - _: An ignored user data parameter.
    """
    print(f"Progress: {complete*100:.2f}%", end="\r")
    sys.stdout.flush()


def main():
    for country_id in ['PH', 'IDN']:
        for raster_list_id in ['conservation_rasters', 'restoration_rasters']:
            raster_list = WORK[country_id][raster_list_id]
            mask_raster_path = os.path.join(WORKING_DIR, f'{raster_list_id}.tif')

            geoprocessing.raster_calculator(
                [(path, 1) for path in raster_list], _merge_masks_op,
                mask_raster_path,
                gdal.GDT_Byte, 0)

            # Open the raster file
            raster = gdal.Open(mask_raster_path)
            band = raster.GetRasterBand(1)

            srs = osr.SpatialReference()
            srs.ImportFromWkt(raster.GetProjection())

            # Create the destination data source
            target_layername = f'{country_id}_{raster_list_id}'
            driver = ogr.GetDriverByName("GPKG")
            overlap_polygon_path = os.path.join(
                WORKING_DIR, f'{target_layername}.gpkg')
            target_vector = driver.CreateDataSource(overlap_polygon_path)
            target_layer = target_vector.CreateLayer(
                target_layername, srs=srs)

            # Add a field
            fd = ogr.FieldDefn("DN", ogr.OFTInteger)
            target_layer.CreateField(fd)
            dst_field = 0

            # Polygonize
            gdal.Polygonize(
                band, None, target_layer, dst_field, [],
                callback=progress_callback)
            target_layer = None
            target_vector = None
            band = None
            raster = None

            # Load the line vector layer
            line_gdf = gpd.read_file(WORK[country_id]['road_vector_path'])

            # Load the polygon vector layer (used for clipping)
            polygon_gdf = gpd.read_file('path_to_your_polygon_layer.shp')

            # Perform the clipping operation
            clipped_lines = gpd.clip(line_gdf, polygon_gdf)

            # Save the clipped output
            clipped_lines.to_file('path_to_save_clipped_lines.shp')

            return


if __name__ == '__main__':
    main()

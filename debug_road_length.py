from ecoshard.geoprocessing.geoprocessing_core import DEFAULT_OSR_AXIS_MAPPING_STRATEGY
from osgeo import osr
from osgeo import gdal


ELLIPSOID_EPSG = 6933
def length(vector_path):
    target_projection = osr.SpatialReference()
    target_projection.ImportFromEPSG(ELLIPSOID_EPSG)
    target_projection.SetAxisMappingStrategy(DEFAULT_OSR_AXIS_MAPPING_STRATEGY)

    vector = gdal.OpenEx(vector_path)
    clipped_lines_layer = vector.GetLayer()
    total_length = 0
    transform = osr.CreateCoordinateTransformation(
        clipped_lines_layer.GetSpatialRef(), target_projection)
    for line_feature in clipped_lines_layer:
        line_geometry = line_feature.GetGeometryRef()
        if transform is not None:
            err_code = line_geometry.Transform(transform)
            if err_code != 0:
                raise RuntimeError(f'{line_geometry.ExportToWkt()}')
        total_length += line_geometry.Length() / 1000  # convert to km
    return total_length


if __name__ == '__main__':
    for path in ['D:/repositories/wwf-sipa/province_dependence_workspace_2025_02_03/road_vector_debug/downstream_clipped_roads_Jambi_conservation_all_top10_service_downstream_local_coverage.gpkg',
                 'D:/repositories/wwf-sipa/province_dependence_workspace_2025_02_03/road_vector_debug/base_clipped_roads_Jambi.gpkg']:
        print(length(path))

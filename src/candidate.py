import random
import geopandas as gpd
from shapely import Point
from src.network_utils import simplify_nodes


def allocate_poi_by_area(polygon_gdf, total_poi_num):
    if polygon_gdf.crs.is_geographic:
        polygon_gdf = polygon_gdf.to_crs(epsg=3857)
    polygon_gdf['area'] = polygon_gdf.geometry.area
    total_area = polygon_gdf['area'].sum()
    polygon_gdf['poi_num'] = ((polygon_gdf['area'] / total_area) * total_poi_num).round().astype(int)

    diff = total_poi_num - polygon_gdf['poi_num'].sum()
    for _ in range(abs(diff)):
        idx = polygon_gdf['poi_num'].idxmax() if diff > 0 else polygon_gdf['poi_num'].idxmin()
        polygon_gdf.at[idx, 'poi_num'] += 1 if diff > 0 else -1

    return dict(zip(polygon_gdf['poly_id'], polygon_gdf['poi_num'])), polygon_gdf


def generate_candidates(polygon_gdf, poi_per_polygon, uncovered_nodes):
    candidates = []
    for idx, row in polygon_gdf.iterrows():
        poly = row.geometry
        N = poi_per_polygon.get(row.poly_id, 0)
        minx, miny, maxx, maxy = poly.bounds
        for _ in range(N):
            while True:
                p = Point(random.uniform(minx, maxx), random.uniform(miny, maxy))
                if poly.contains(p):
                    candidates.append(p)
                    break

    candidate_gdf = gpd.GeoDataFrame({'node_id': range(len(candidates))}, geometry=candidates, crs=polygon_gdf.crs)
    demand_gdf = simplify_nodes(uncovered_nodes, buffer_distance=20).to_crs(epsg=3857)

    return candidate_gdf.to_crs(epsg=3857), demand_gdf


def join_centrality(candidate_gdf, primal_2_gdf):
    joined = gpd.sjoin_nearest(candidate_gdf, primal_2_gdf[['betweenness', 'geometry']], how='left',
                               distance_col='dist_to_edge')
    joined['betweenness'] = joined['betweenness'].fillna(0)
    bet = joined['betweenness'].values.astype(float)
    joined['betweenness_norm'] = (bet - bet.min()) / (bet.max() - bet.min() + 1e-9)
    return joined
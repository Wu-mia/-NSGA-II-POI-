import networkx as nx
import pandas as pd
import geopandas as gpd
from shapely import LineString, Point
from src.network_utils import get_nearest_node, simplify_nodes


def calculate_reachable(G, tree, nodes, points, distance, edges):
    poi_nodes = get_nearest_node(points, tree, nodes)
    reachable_nodes = set()

    for n in poi_nodes:
        length = nx.single_source_dijkstra_path_length(G, n, cutoff=distance, weight='length')
        reachable_nodes.update(length.keys())

    edges_list = []
    for u, v, data in G.edges(data=True):
        geom = data.get('geometry')
        if geom is None:
            geom = LineString([(G.nodes[u]['x'], G.nodes[u]['y']), (G.nodes[v]['x'], G.nodes[v]['y'])])
        edges_list.append({'u': u, 'v': v, 'geometry': geom})

    edges_gdf = gpd.GeoDataFrame(edges_list, geometry='geometry', crs=edges.crs)
    reachable_edges = edges_gdf[edges_gdf['u'].isin(reachable_nodes) & edges_gdf['v'].isin(reachable_nodes)]

    return reachable_edges


def get_uncovered_polygon(edges, reachable_edges):
    uncovered_edges = edges[
        ~edges['geometry'].apply(lambda g: g.wkb in reachable_edges['geometry'].apply(lambda x: x.wkb).values)].copy()
    uncovered_edges['u'] = uncovered_edges['geometry'].apply(lambda g: Point(g.coords[0]))
    uncovered_edges['v'] = uncovered_edges['geometry'].apply(lambda g: Point(g.coords[-1]))

    uncovered_nodes = pd.concat([uncovered_edges["u"], uncovered_edges["v"]], ignore_index=True)
    uncovered_nodes = uncovered_nodes.drop_duplicates().reset_index(drop=True)
    uncovered_nodes_gdf = gpd.GeoDataFrame(geometry=uncovered_nodes, crs=uncovered_edges.crs)

    buffer_area = uncovered_edges.buffer(20)
    polygons = list(buffer_area.unary_union.geoms)
    polygons_gdf = gpd.GeoDataFrame({'poly_id': range(len(polygons))}, geometry=polygons, crs=edges.crs)

    return polygons_gdf, uncovered_nodes_gdf
import geopandas as gpd
from shapely import Point
import momepy
from shapely.ops import unary_union
import numpy as np
from scipy.spatial import cKDTree


def prepare_edges(edges_path):
    edges = gpd.read_file(edges_path)
    edges = edges.copy()
    edges['u'] = edges['geometry'].apply(lambda g: Point(g.coords[0]))
    edges['v'] = edges['geometry'].apply(lambda g: Point(g.coords[-1]))
    return edges


def build_graph_and_kdtree(edges):
    G = momepy.gdf_to_nx(edges, approach='primal', length='length')
    nodes = list(G.nodes())
    nodes_coords = np.array([[G.nodes[i]['x'], G.nodes[i]['y']] for i in nodes])
    tree = cKDTree(nodes_coords)
    return G, nodes, tree


def get_nearest_node(points, tree, nodes):
    poi_point = []
    for pt in points.geometry:
        dist, idx = tree.query([pt.x, pt.y])
        poi_point.append(nodes[idx])
    return poi_point


def simplify_nodes(nodes_gdf, buffer_distance):
    buffers = nodes_gdf.geometry.buffer(buffer_distance)
    merged = unary_union(buffers)

    if merged.geom_type == 'Polygon':
        polys = [merged]
    elif merged.geom_type == "MultiPolygon":
        polys = list(merged.geoms)
    else:
        polys = []

    rep_points = [p.representative_point() for p in polys]
    return gpd.GeoDataFrame(geometry=rep_points, crs=nodes_gdf.crs)


def calc_betweenness(G):
    primal_2 = momepy.betweenness_centrality(G, name='betweenness', mode='edges', weight='length')
    return momepy.nx_to_gdf(primal_2, points=False)
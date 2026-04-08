import numpy as np
import networkx as nx
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.operators.crossover.pntx import SinglePointCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.termination import get_termination
from pymoo.optimize import minimize
from src.network_utils import get_nearest_node


def run_NSGA2(candidate_gdf, demand_gdf, G, current_poi, R, target_cov, tree, nodes, lambda_bc=0.5):
    cand_nodes = get_nearest_node(candidate_gdf, tree, nodes)
    dem_nodes = get_nearest_node(demand_gdf, tree, nodes)
    rest_nodes = get_nearest_node(current_poi, tree, nodes)

    dem_node_to_idx = {node: idx for idx, node in enumerate(dem_nodes)}
    n_c, n_d, n_r = len(cand_nodes), len(dem_nodes), len(rest_nodes)

    dist_net_cand_dem = np.full((n_c, n_d), np.inf, dtype=float)
    for i, c_node in enumerate(cand_nodes):
        lengths = nx.single_source_dijkstra_path_length(G, c_node, cutoff=R, weight='length')
        for node, dist in lengths.items():
            if node in dem_node_to_idx:
                dist_net_cand_dem[i, dem_node_to_idx[node]] = min(dist, dist_net_cand_dem[i, dem_node_to_idx[node]])

    dist_net_rest_dem = np.full((n_r, n_d), np.inf, dtype=float)
    for i, r_node in enumerate(rest_nodes):
        lengths = nx.single_source_dijkstra_path_length(G, r_node, cutoff=R, weight='length')
        for node, dist in lengths.items():
            if node in dem_node_to_idx:
                dist_net_rest_dem[i, dem_node_to_idx[node]] = min(dist, dist_net_rest_dem[i, dem_node_to_idx[node]])

    already_covered = np.any(dist_net_rest_dem <= R, axis=0)
    betweenness_norm = candidate_gdf['betweenness_norm'].fillna(0).values.astype(float)

    class POISelectionNetProblem(ElementwiseProblem):
        def __init__(self):
            super().__init__(n_var=n_c, n_obj=2, n_constr=1, xl=0, xu=1, type_var=np.int32)

        def _evaluate(self, x, out, *args, **kwargs):
            chosen = (x == 1)
            f1 = chosen.sum() - lambda_bc * (betweenness_norm[chosen].sum() if chosen.any() else 0)

            covered_by_new = (dist_net_cand_dem[chosen, :].min(axis=0) <= R) if chosen.any() else np.zeros(n_d,
                                                                                                           dtype=bool)
            coverage_ratio = (already_covered | covered_by_new).sum() / n_d if n_d > 0 else 0.0

            out["F"] = np.array([f1, -float((already_covered | covered_by_new).sum())])
            out["G"] = np.array([target_cov - coverage_ratio])

    problem = POISelectionNetProblem()
    algorithm = NSGA2(pop_size=200, sampling=BinaryRandomSampling(), crossover=SinglePointCrossover(prob=0.9),
                      mutation=BitflipMutation(prob=0.05), eliminate_duplicate=True)
    res = minimize(problem, algorithm, get_termination("n_gen", 400), seed=1, verbose=True)

    cov_list = []
    for x in res.X:
        chosen = (x == 1)
        c_new = (dist_net_cand_dem[chosen, :].min(axis=0) <= R) if chosen.any() else np.zeros(n_d, dtype=bool)
        cov_list.append((already_covered | c_new).sum() / n_d)

    cov_list = np.array(cov_list)
    ok_idx = np.where(cov_list >= target_cov)[0]

    if len(ok_idx) == 0:
        print("未找到满足覆盖率的解。")
        return None, 0.0, res

    best_idx = ok_idx[np.argmin(res.X[ok_idx].sum(axis=1))]
    best_x = res.X[best_idx]

    return candidate_gdf[best_x == 1].copy(), cov_list[best_idx], res
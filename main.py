import os
import pandas as pd
import geopandas as gpd
import osmnx as ox
from src.config import EDGES_PATH, POI_PATHS, TARGET_COV, CANDIDATE_NUM, OUTPUT_DIR
from src.network_utils import prepare_edges, build_graph_and_kdtree, calc_betweenness
from src.coverage import calculate_reachable, get_uncovered_polygon
from src.candidate import allocate_poi_by_area, generate_candidates, join_centrality
from src.optimization import run_NSGA2
from src.visualization import plot_centrality, plot_pareto_front, plot_final_decision_map


def main():
    print("--- Urban-Optima 决策引擎启动 ---")

    # 1. 环境准备与底图加载
    print("\n[Step 1/5] 正在读取路网与环境数据...")
    if not os.path.exists(EDGES_PATH):
        print(f"❌ 错误：在 {EDGES_PATH} 未找到路网文件，请检查路径。")
        return

    edges = prepare_edges(EDGES_PATH)
    # 获取建筑数据作为可视化底图 (基于原脚本坐标)
    center_point = (43.77915, 11.28373)
    print(f"正在从 OSM 获取 {center_point} 附近的建筑数据...")
    buildings = ox.features_from_point(center_point, dist=1500, tags={'building': True})
    buildings = buildings.to_crs(epsg=3857)

    # 2. 构建拓扑网络与计算中心性
    print("\n[Step 2/5] 正在构建路网拓扑并计算中心性指标...")
    G, nodes, tree = build_graph_and_kdtree(edges)
    primal_2_gdf = calc_betweenness(G)

    # 保存中心性可视化结果
    plot_centrality(
        primal_2_gdf,
        'betweenness',
        "Road Network Betweenness Centrality",
        os.path.join(OUTPUT_DIR, "network_centrality_analysis.png")
    )

    all_selected_candidates = []

    # 3. 循环处理每一类 POI 服务设施
    print("\n[Step 3/5] 进入多目标优化迭代流程...")
    for poi_name, (path, radius) in POI_PATHS.items():
        if not os.path.exists(path):
            print(f"⚠️ 找不到 {poi_name} 的 POI 数据，跳过此项。")
            continue

        print(f"\n>>> 正在处理业务类型: {poi_name} (服务半径: {radius}m)")
        poi_gdf = gpd.read_file(path).to_crs(epsg=3857)

        # 4. 盲区识别与候选点生成
        print(f"   - 识别服务盲区...")
        reachable_edges_before = calculate_reachable(G, tree, nodes, poi_gdf, radius, edges)
        polygon_gdf, uncovered_nodes = get_uncovered_polygon(edges, reachable_edges_before)

        print(f"   - 生成候选空间点位...")
        poi_per_polygon, _ = allocate_poi_by_area(polygon_gdf, CANDIDATE_NUM)
        candidate_gdf, demand_gdf = generate_candidates(polygon_gdf, poi_per_polygon, uncovered_nodes)
        candidate_gdf = join_centrality(candidate_gdf, primal_2_gdf)

        # 5. 运行 NSGA-II 算法决策
        print(f"   - 启动 NSGA-II 多目标求解引擎...")
        selected, best_cov, res = run_NSGA2(
            candidate_gdf, demand_gdf, G, poi_gdf,
            radius, TARGET_COV, tree, nodes
        )

        if selected is not None:
            print(f"   ✅ 优化成功！[新增点数: {len(selected)} | 最终覆盖率: {best_cov:.2%}]")

            # --- 产出可视化决策报告 ---
            print(f"   - 正在生成 {poi_name} 的决策分析图表...")

            # 5a. 帕累托前沿分析
            plot_pareto_front(res, poi_name, os.path.join(OUTPUT_DIR, f"{poi_name}_pareto_front.png"))

            # 5b. 计算优化后的覆盖效果
            poi_after = pd.concat([poi_gdf[['geometry']], selected[['geometry']]], ignore_index=True)
            poi_after = gpd.GeoDataFrame(poi_after, geometry='geometry', crs=edges.crs)
            reachable_edges_after = calculate_reachable(G, tree, nodes, poi_after, radius, edges)

            # 5c. 生成最终决策地图
            stats_text = f"Type: {poi_name} | Radius: {radius}m\nOriginal Sites: {len(poi_gdf)} | New Sites: {len(selected)}\nTarget Coverage: {TARGET_COV} | Actual: {best_cov:.2%}"
            plot_final_decision_map(
                edges, buildings, candidate_gdf, selected,
                reachable_edges_after, stats_text,
                os.path.join(OUTPUT_DIR, f"{poi_name}_decision_map.png")
            )

            # 保存 GeoJSON 数据
            selected.to_file(os.path.join(OUTPUT_DIR, f"{poi_name}_optimized_sites.geojson"), driver='GeoJSON')
            all_selected_candidates.append(selected)

    # 6. 汇总与结束
    if all_selected_candidates:
        print("\n[Step 5/5] 正在汇总所有品类优化结果...")
        final_gdf = pd.concat(all_selected_candidates, ignore_index=True)
        final_gdf = gpd.GeoDataFrame(final_gdf, geometry='geometry')
        final_gdf.to_file(os.path.join(OUTPUT_DIR, "all_optimized_sites_summary.geojson"), driver='GeoJSON')

    print("\n--- 流程全部执行完毕！请查看 data/output 目录下的可视化报告与数据文件 ---")


if __name__ == "__main__":
    main()
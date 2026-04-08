import matplotlib.pyplot as plt
import os
import geopandas as gpd


def plot_centrality(gdf, column, title, save_path):
    """可视化路网中心性 (Closeness/Betweenness)"""
    fig, ax = plt.subplots(figsize=(10, 10))
    gdf.plot(ax=ax, column=column, cmap='Spectral_r', k=15, alpha=1, legend=True)
    plt.title(title)
    plt.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_pareto_front(res, poi_name, save_path):
    """绘制多目标优化的帕累托前沿 (Pareto Front)"""
    F = res.F
    plt.figure(figsize=(10, 6))
    plt.scatter(-F[:, 0], -F[:, 1], edgecolor='k', s=50)
    plt.title(f'Pareto Front - {poi_name}')
    plt.xlabel('Efficiency Score (New Points & Centrality)')
    plt.ylabel('Maximum Coverage')
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()


def plot_final_decision_map(edges, buildings, candidate_gdf, selected_gdf, reachable_edges, stats_str, save_path):
    """生成最终的选址决策地图 (包含背景建筑、热力候选点、选中点及覆盖路网)"""
    fig, ax = plt.subplots(figsize=(12, 12))

    # 1. 绘制底图：路网与建筑
    edges.plot(ax=ax, edgecolor='gray', linewidth=0.5, alpha=0.5, zorder=1)
    buildings.plot(ax=ax, facecolor='lightgrey', edgecolor='none', alpha=0.4, zorder=2)

    # 2. 绘制优化后的覆盖范围 (红色路段)
    if not reachable_edges.empty:
        reachable_edges.plot(ax=ax, color='red', linewidth=1, alpha=0.7, zorder=3, label='Covered Network')

    # 3. 绘制候选点热力 (基于中心性权重)
    candidate_gdf.plot(ax=ax, column='betweenness_norm', cmap='viridis', markersize=20, alpha=0.6, zorder=4)

    # 4. 绘制最终选中的决策点 (大红点)
    selected_gdf.plot(ax=ax, color='red', edgecolor='black', markersize=60, alpha=1, zorder=5,
                      label='Optimized Locations')

    # 5. 添加统计信息文字说明
    plt.figtext(0.5, 0.02, stats_str, ha='center', fontsize=12, color='darkred',
                bbox=dict(facecolor='white', alpha=0.8))

    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
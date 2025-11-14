import os
import pickle
import shap
import matplotlib.pyplot as plt
import numpy as np

def get_watershed_name_mapping():
    return {
        '渭河上游': 'Upper Weihe River',
        '渭河中游': 'Middle Weihe River',
        '渭河下游': 'Lower Weihe River',
        '北洛河': 'North Lou River',
        '泾河': 'Jing River'
    }

def process_shap_data(shap_data):
    """处理SHAP数据"""
    shap_values = shap_data['shap_values']
    feature_names = shap_data['feature_names']
    features = shap_data['features']

    # 定义需要的特征
    target_features = ['NDVI', 'SM', 'TMP', 'ET', 'PRE', 'TMN', 'TMX']
    # 创建布尔掩码：只保留目标特征
    mask = []
    valid_indices = []
    for i, name in enumerate(feature_names):
        if name in target_features:
            mask.append(True)
            valid_indices.append(i)
        else:
            mask.append(False)

    print(f"  过滤特征: {[feature_names[i] for i in valid_indices]}")

    # 应用掩码提取数据
    shap_vals = shap_values[:, valid_indices]
    feature_names_filtered = [feature_names[i] for i in valid_indices]
    features_filtered = features[:, valid_indices]

    # 处理维度
    if len(shap_vals.shape) == 3:
        shap_vals = shap_vals.squeeze(axis=-1) if shap_vals.shape[-1] == 1 else shap_vals.mean(axis=1)
    if len(features_filtered.shape) == 3:
        features_filtered = features_filtered.squeeze(axis=1) if features_filtered.shape[
                                                                     1] == 1 else features_filtered.mean(axis=1)

    # 计算重要性并排序
    importance = np.abs(shap_vals).mean(axis=0)
    sorted_idx = np.argsort(importance)[::-1]

    print(f"  最终SHAP维度: {shap_vals.shape}")
    print(f"  特征重要性排序: {[feature_names_filtered[i] for i in sorted_idx]}")

    return (shap_vals[:, sorted_idx],
            features_filtered[:, sorted_idx],
            [feature_names_filtered[i] for i in sorted_idx],
            importance[sorted_idx])

def plot_watershed_shap(ax_bee, shap_vals, features, feature_names, importances, watershed_name, cmap_name="RdYlBu"):
    """绘制单个流域的SHAP图"""
    importance_pct = importances / importances.sum() * 100

    # 设置beeswarm配色
    shap.plots._utils.colors.red_blue = cmap_name
    cmap_gradient = plt.get_cmap(cmap_name)

    explanation = shap.Explanation(values=shap_vals, data=features, feature_names=feature_names)
    shap.plots.beeswarm(explanation, ax=ax_bee, show=False, plot_size=None)

    # 获取y轴信息
    y_coords = ax_bee.get_yticks()
    y_labels = [tick.get_text() for tick in ax_bee.get_yticklabels()]
    y_order = y_labels if any(lbl != "" for lbl in y_labels) else feature_names

    importance_map = dict(zip(feature_names, importances))
    pct_map = dict(zip(feature_names, importance_pct))
    bar_widths = [importance_map.get(lbl, 0) for lbl in y_order]
    bar_pcts = [pct_map.get(lbl, 0) for lbl in y_order]

    # 创建柱状图坐标轴
    ax_bar = ax_bee.twiny()
    ax_bar.set_zorder(0)
    ax_bee.set_zorder(1)
    ax_bee.patch.set_alpha(0)

    # 渐变色柱状图
    colors = cmap_gradient(np.linspace(0.3, 0.9, len(y_coords)))
    ax_bar.barh(y=y_coords, width=bar_widths, height=0.65, alpha=0.35, color=colors, edgecolor="none", zorder=0)

    # 柱状图坐标轴设置
    bar_xlim = max(bar_widths) * 1.12
    ax_bar.set_xlim(0, bar_xlim)
    ax_bar.set_xticks(np.linspace(0, bar_xlim, 6))
    ax_bar.set_xticklabels([f"{x:.3f}" for x in np.linspace(0, bar_xlim, 6)], fontsize=9.5)
    ax_bar.set_xlabel("Mean(|SHAP value|)", fontsize=11, fontweight='medium', labelpad=8)
    ax_bar.set_yticks([])
    ax_bar.tick_params(axis='x', which='major', length=4, width=1.2, direction='in', pad=6)

    # beeswarm坐标轴设置
    shap_xlim = round(abs(shap_vals).max() * 1.15, 2)
    ax_bee.set_xlim(-shap_xlim, shap_xlim)
    ax_bee.set_xticks(np.linspace(-shap_xlim, shap_xlim, 7))
    ax_bee.set_xticklabels([f"{x:.2f}" for x in np.linspace(-shap_xlim, shap_xlim, 7)], fontsize=9.5)
    ax_bee.set_xlabel("SHAP value (impact on model output)", fontsize=11, fontweight='medium', labelpad=8)
    ax_bee.set_yticks(y_coords)
    ax_bee.set_yticklabels(y_order, fontsize=10)
    ax_bee.set_ylabel("")
    ax_bee.tick_params(axis='x', which='major', length=4, width=1.2, direction='in', pad=6)
    ax_bee.tick_params(axis='y', which='major', length=0)

    ax_bar.set_ylim(ax_bee.get_ylim())

    # 百分比标签
    for y, p in zip(y_coords, bar_pcts):
        ax_bar.text(0.015 * bar_xlim, y, f"({p:.1f}%)", va="center", ha="left",
                    fontsize=8.5, color="black", fontweight='medium')

    # 标题设置
    ax_bee.set_title(watershed_name, fontsize=12, fontweight='bold', pad=12, loc='center')

    # 添加网格线
    ax_bee.grid(axis='x', alpha=0.15, linestyle='--', linewidth=0.8, zorder=0)

    # 边框设置
    for spine in ax_bee.spines.values():
        spine.set_linewidth(1.2)
    for spine in ax_bar.spines.values():
        spine.set_linewidth(1.2)

def visualize_shap_results():
    """生成五个流域的SHAP可视化图"""
    shap_folder_path = './模型保存/shap_results'

    # 检查路径是否存在
    if not os.path.exists(shap_folder_path):
        print(f"错误：SHAP结果文件夹不存在: {shap_folder_path}")
        print("请先运行训练代码生成SHAP结果")
        return

    shap_files = [f for f in os.listdir(shap_folder_path) if f.endswith('.pkl')]

    if not shap_files:
        print(f"错误：在{shap_folder_path}文件夹中未找到SHAP结果文件")
        return

    mapping = get_watershed_name_mapping()
    watersheds_data = []

    for shap_file in shap_files:
        watershed_name_cn = shap_file.replace('_shap.pkl', '')
        watershed_name_en = mapping.get(watershed_name_cn, watershed_name_cn)

        # 跳过summary文本文件
        if not shap_file.endswith('_shap.pkl'):
            continue

        with open(os.path.join(shap_folder_path, shap_file), 'rb') as f:
            shap_data = pickle.load(f)

        shap_vals, features, feature_names, importances = process_shap_data(shap_data)
        watersheds_data.append((watershed_name_en, shap_vals, features, feature_names, importances))
        print(f"已加载: {watershed_name_en}")

    # 创建大图
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['axes.linewidth'] = 1.2
    plt.rcParams['xtick.major.width'] = 1.2
    plt.rcParams['ytick.major.width'] = 1.2

    fig = plt.figure(figsize=(13, 16), dpi=300)
    fig.patch.set_facecolor('white')

    # 绘制每个流域
    for idx, (name, shap_vals, features, feature_names, importances) in enumerate(watersheds_data):
        ax = fig.add_subplot(5, 1, idx + 1)
        plot_watershed_shap(ax, shap_vals, features, feature_names, importances, name, cmap_name="RdYlBu")

    plt.tight_layout(pad=2.5, h_pad=3.0)
    output_path = "shap_visualization_combined.png"
    plt.savefig(output_path, dpi=600, format='png', bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"已保存: {output_path}")
    plt.show()

def main():
    print("开始SHAP可视化...")
    visualize_shap_results()
    print("SHAP可视化完成！")

if __name__ == "__main__":
    main()
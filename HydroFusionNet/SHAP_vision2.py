import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from scipy.stats import spearmanr

COLOR_PALETTE = {
    'background': '#FAFBFC',
    'zero_line': '#7F8C8D',
    'ci': '#5DADE2',
    'main': '#2C3E50',
    'tipping_point': '#E74C3C',
    'positive': '#27AE60',
    'negative': '#E67E22',
    'grid': '#E8EAED'
}


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

    target_features = ['NDVI', 'SM', 'TMP', 'ET', 'PRE', 'TMN', 'TMX']

    print(f"  原始SHAP维度: {shap_values.shape}")
    print(f"  原始特征列表: {feature_names}")

    valid_indices = [i for i, name in enumerate(feature_names) if name in target_features]

    shap_vals = shap_values[:, valid_indices]
    feature_names_filtered = [feature_names[i] for i in valid_indices]
    features_filtered = features[:, valid_indices]

    if len(shap_vals.shape) == 3:
        shap_vals = shap_vals.squeeze(axis=-1) if shap_vals.shape[-1] == 1 else shap_vals.mean(axis=1)
    if len(features_filtered.shape) == 3:
        features_filtered = features_filtered.squeeze(axis=1) if features_filtered.shape[
                                                                     1] == 1 else features_filtered.mean(axis=1)

    importance = np.abs(shap_vals).mean(axis=0)
    sorted_idx = np.argsort(importance)[::-1]

    print(f"  最终SHAP维度: {shap_vals.shape}")
    print(f"  特征重要性排序: {[feature_names_filtered[i] for i in sorted_idx]}")

    return (shap_vals[:, sorted_idx],
            features_filtered[:, sorted_idx],
            [feature_names_filtered[i] for i in sorted_idx],
            importance[sorted_idx])


def fit_smooth_curve(x, y, n_points=200):
    """拟合平滑曲线"""
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    y_sorted = y[sort_idx]

    try:
        data_variance = np.var(y_sorted)
        smoothing_factor = max(len(x) * 0.01, len(x) * data_variance * 0.005)

        spl = UnivariateSpline(x_sorted, y_sorted, s=smoothing_factor, k=3)
        x_smooth = np.linspace(x_sorted.min(), x_sorted.max(), n_points)
        y_smooth = spl(x_smooth)

        residuals = y_sorted - spl(x_sorted)
        std_residuals = np.std(residuals)
        ci_lower = y_smooth - 1.96 * std_residuals
        ci_upper = y_smooth + 1.96 * std_residuals
        ci = np.column_stack([ci_lower, ci_upper])

        return x_smooth, y_smooth, ci
    except Exception as e:
        print(f"拟合失败: {e}, 使用原始数据")
        return x, y, np.column_stack([y, y])


def find_tipping_points(x, y):
    """检测零点交叉"""
    tipping_points = []
    for i in range(len(y) - 1):
        if y[i] * y[i + 1] < 0:
            x_tip = x[i] + (x[i + 1] - x[i]) * abs(y[i]) / (abs(y[i]) + abs(y[i + 1]))
            tipping_points.append((x_tip, 0))
    return tipping_points if tipping_points else None


def plot_scientific_style(ax, XX, y_pred, ci, feature_name, tipping_points=None, p_value=None):
    """科研风格绘图"""
    ax.set_facecolor(COLOR_PALETTE['background'])

    # 添加网格线
    ax.grid(True, which='major', linestyle=':', linewidth=0.5, color=COLOR_PALETTE['grid'], alpha=0.6, zorder=1)

    # 零线
    ax.axhline(0, color=COLOR_PALETTE['zero_line'], linestyle='--', linewidth=1.5, alpha=0.85, zorder=2)

    # 置信区间
    ax.fill_between(XX.flatten(), ci[:, 0], ci[:, 1],
                    color=COLOR_PALETTE['ci'], alpha=0.25, zorder=3,
                    edgecolor='none', label='95% CI')

    # 主曲线
    ax.plot(XX, y_pred, color=COLOR_PALETTE['main'], linewidth=2.8, zorder=5, label='SHAP trend')

    # 填充正负区域
    pos_mask = y_pred > 0
    if any(pos_mask):
        ax.fill_between(XX.flatten(), 0, y_pred, where=pos_mask,
                        color=COLOR_PALETTE['positive'], alpha=0.18, zorder=0)
    neg_mask = y_pred <= 0
    if any(neg_mask):
        ax.fill_between(XX.flatten(), 0, y_pred, where=neg_mask,
                        color=COLOR_PALETTE['negative'], alpha=0.18, zorder=0)

    # 临界点标注
    if tipping_points:
        for idx, (x, y) in enumerate(tipping_points):
            ax.axvline(x, color=COLOR_PALETTE['tipping_point'],
                       linestyle='--', linewidth=1.8, alpha=0.75, zorder=4)
            ax.scatter(x, y, color=COLOR_PALETTE['tipping_point'],
                       s=90, zorder=6, edgecolor='white', linewidth=2, marker='o')
            ax.annotate(f'TP: {x:.2f}', xy=(x, y),
                        xytext=(10, 15 if idx % 2 == 0 else -20),
                        textcoords='offset points', fontsize=9, fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.4", facecolor='white',
                                  alpha=0.9, edgecolor=COLOR_PALETTE['tipping_point'], linewidth=1.5),
                        arrowprops=dict(arrowstyle='->', color=COLOR_PALETTE['tipping_point'],
                                        lw=1.2, alpha=0.8))

    # 坐标轴标签
    ax.set_xlabel(feature_name, labelpad=8, fontweight='bold', fontsize=11)
    ax.set_ylabel('SHAP Value', labelpad=8, fontweight='bold', fontsize=11)

    # 刻度参数
    ax.tick_params(axis='both', which='major', labelsize=9, width=1.2, length=5)
    ax.tick_params(axis='both', which='minor', labelsize=0, width=0, length=0)

    # 边框
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
        spine.set_color('#2C3E50')

    # p值标注
    if p_value is not None:
        if p_value < 0.001:
            p_text = "p < 0.001***"
        elif p_value < 0.01:
            p_text = f"p < 0.01**"
        elif p_value < 0.05:
            p_text = f"p < 0.05*"
        else:
            p_text = f"p = {p_value:.3f}"

        ax.text(0.97, 0.97, p_text, transform=ax.transAxes,
                ha='right', va='top', fontsize=10, fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.9, edgecolor='#34495E',
                          linewidth=1.5, pad=4, boxstyle='round,pad=0.5'))

    # 图例
    ax.legend(loc='upper left', frameon=True, framealpha=0.95,
              edgecolor='#BDC3C7', fontsize=9, fancybox=True, shadow=False)


def visualize_shap_dependence(top_n_features=3):
    """生成SHAP依赖图"""
    shap_folder_path = './模型保存/shap_results'
    shap_files = [f for f in os.listdir(shap_folder_path) if f.endswith('.pkl')]

    if not shap_files:
        print(f"错误：在{shap_folder_path}文件夹中未找到SHAP结果文件")
        return

    mapping = get_watershed_name_mapping()
    all_watersheds_data = []

    for shap_file in sorted(shap_files):
        watershed_name_cn = shap_file.split('.')[0].replace('_shap_values', '')
        watershed_name_en = mapping.get(watershed_name_cn, 'Unknown Watershed')

        with open(os.path.join(shap_folder_path, shap_file), 'rb') as f:
            shap_data = pickle.load(f)

        shap_vals, features, feature_names, importance = process_shap_data(shap_data)
        top_indices = list(range(min(top_n_features, len(feature_names))))

        print(f"\n{watershed_name_cn} ({watershed_name_en}) - Top {len(top_indices)} 重要特征:")
        for rank, idx in enumerate(top_indices, 1):
            print(f"  {rank}. {feature_names[idx]}: {importance[idx]:.4f}")
            all_watersheds_data.append((
                watershed_name_en,
                shap_vals[:, idx],
                features[:, idx],
                feature_names[idx],
                rank
            ))

    # 配置matplotlib
    plt.rcParams.update({
        'font.family': 'Times New Roman',
        'font.size': 10,
        'axes.linewidth': 1.5,
        'figure.dpi': 300,
        'savefig.dpi': 600,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1
    })

    n_cols = top_n_features
    n_rows = len(shap_files)

    fig = plt.figure(figsize=(6.5 * n_cols, 4.5 * n_rows), dpi=300)
    fig.patch.set_facecolor('white')

    for idx, (watershed_name, shap_vals, feature_vals, feature_name, rank) in enumerate(all_watersheds_data):
        ax = fig.add_subplot(n_rows, n_cols, idx + 1)

        x_smooth, y_smooth, ci = fit_smooth_curve(feature_vals, shap_vals)
        tipping_points = find_tipping_points(x_smooth, y_smooth)
        _, p_value = spearmanr(feature_vals, shap_vals)

        plot_scientific_style(ax, x_smooth, y_smooth, ci, feature_name, tipping_points, p_value)

        title = f"{watershed_name}\n(Feature Rank #{rank})"
        ax.set_title(title, fontsize=12, fontweight='bold', pad=12, color='#2C3E50')

    plt.tight_layout(pad=2.5, h_pad=3.5, w_pad=2.5)

    output_path = f"shap_dependence_top{top_n_features}.png"
    plt.savefig(output_path, dpi=600, format='png', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"\n图像已保存: {output_path}")
    plt.show()


def main():
    print("=" * 60)
    print("开始SHAP依赖图可视化...")
    print("=" * 60)
    visualize_shap_dependence(top_n_features=3)
    print("=" * 60)
    print("SHAP依赖图可视化完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()

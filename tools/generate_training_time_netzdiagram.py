import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.cm as cm
from matplotlib.ticker import FuncFormatter
from matplotlib.lines import Line2D

# Custom scale function for compressing the 0-60 range, expanding the 60-100 range
def custom_scale(y):
    return np.where(y <= 60, y * 0.3 / 60, 0.3 + (y - 60) * 0.7 / 40)

# 反向的自定义刻度函数 - 用于显示刻度标签
def inverse_custom_scale(y):
    return np.where(y <= 0.3, y * 60 / 0.3, 60 + (y - 0.3) * 40 / 0.7)

# 获取脚本目录
script_dir = os.path.dirname(os.path.abspath(__file__))
# 日志目录
logs_dir = os.path.join(script_dir, '..', 'logs', 'training_time')
# 输出路径
output_dir = os.path.join(script_dir, '..', 'output')
# 如果输出目录不存在，则创建
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 定义数据集信息（样本数和类别数）
dataset_info = {
    'visa': {'samples': 10281, 'categories': 12},
    'mvtec': {'samples': 5354, 'categories': 15},
    'mvtec3d': {'samples': 5312, 'categories': 10},
    'btech': {'samples': 2830, 'categories': 3},
    'kolektor': {'samples': 399, 'categories': 1}
}

# 按样本数量降序排列数据集
sorted_datasets = sorted(dataset_info.keys(), key=lambda x: dataset_info[x]['samples'], reverse=True)

# 数据集CSV文件路径
file_paths = {
    dataset: os.path.join(logs_dir, f"{dataset}.csv")
    for dataset in dataset_info.keys()
}

# 打印文件路径用于调试
for name, path in file_paths.items():
    print(f"Looking for {name} at: {os.path.abspath(path)}")

# 读取CSV文件
dfs = {name: pd.read_csv(path) for name, path in file_paths.items()}

# 获取所有唯一的模型名称
all_models = []
for df in dfs.values():
    all_models.extend(df['model'].tolist())
all_models = sorted(list(set(all_models)))

# 创建用于绘图的数据结构
auroc_values = {}  # 存储每个模型在每个数据集上的AUROC值

# 处理数据
for model in all_models:
    auroc_values[model] = []
    
    for dataset in sorted_datasets:
        df = dfs[dataset]
        if model in df['model'].values:
            # 获取该模型在当前数据集的AUROC值
            model_row = df[df['model'] == model].iloc[0]
            auroc = float(model_row['image auroc value'])
            
            auroc_values[model].append(auroc)
        else:
            # 如果该模型在当前数据集中不存在，填充None
            auroc_values[model].append(None)

# 计算每个模型的平均AUROC值
model_avg_auroc = {}
for model in all_models:
    valid_aurocs = [a for a in auroc_values[model] if a is not None]
    if valid_aurocs:
        model_avg_auroc[model] = sum(valid_aurocs) / len(valid_aurocs)
    else:
        model_avg_auroc[model] = 0

# 按平均AUROC值排序所有模型
sorted_models = sorted(model_avg_auroc.keys(), key=lambda m: model_avg_auroc[m], reverse=True)
print(f"Found {len(sorted_models)} models: {sorted_models}")

# 创建一个较大的图表以容纳所有15个模型
fig = plt.figure(figsize=(18, 16))
ax = fig.add_subplot(111, polar=True)

# 确定角度
angles = np.linspace(0, 2*np.pi, len(sorted_datasets), endpoint=False).tolist()
# 闭合图形
angles += angles[:1]

# 设置雷达图的方向，从上方开始
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)

# 设置每个点的标签
ax.set_xticks(angles[:-1])
ax.set_xticklabels([dataset.upper() for dataset in sorted_datasets], fontsize=14, fontweight='bold')

# 应用自定义刻度
scaled_rgrids = [custom_scale(val) for val in [0, 20, 40, 60, 70, 80, 90, 95, 100]]
label_rgrids = [0, 20, 40, 60, 70, 80, 90, 95, 100]

# 设置y轴范围，应用自定义刻度
ax.set_ylim(0, 1)  # 缩放后的范围是0-1
ax.set_rgrids(scaled_rgrids, labels=[f"{val}" for val in label_rgrids], fontsize=12)

# 绘制60的特殊环线，用虚线标记
sixty_circle = plt.Circle((0, 0), custom_scale(60), transform=ax.transData._b, 
                        fill=False, edgecolor='gray', linestyle='--', linewidth=1)
ax.add_artist(sixty_circle)

# 标记60分隔线
plt.text(0, custom_scale(60) + 0.02, "60", transform=ax.transData._b, 
        horizontalalignment='center', verticalalignment='bottom', 
        fontsize=12, color='gray')

# 使用HSV颜色空间创建一系列清晰可区分的颜色
hues = np.linspace(0, 1, len(sorted_models), endpoint=False)
colors = [plt.cm.hsv(h) for h in hues]

# 定义线型和标记符
line_styles = ['-', '--', '-.', ':']
markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'h', '8', '+', 'x', 'd', '|', '_', '.']

# 存储每个模型的图形元素，用于创建图例
legend_elements = []

# 绘制每个模型的雷达图
for i, model in enumerate(sorted_models):
    values = [v if v is not None else 0 for v in auroc_values[model]]
    # 应用自定义刻度
    scaled_values = [custom_scale(v) for v in values]
    # 闭合数据
    scaled_values += scaled_values[:1]
    
    # 确定线宽和线型
    line_width = 2.5
    line_style = line_styles[i % len(line_styles)]
    marker = markers[i % len(markers)]
    color = colors[i]
    
    # 绘制线条
    line = ax.plot(angles, scaled_values, 
            linewidth=line_width, 
            color=color, 
            linestyle=line_style, 
            marker=marker, 
            markersize=8,
            label=model)[0]
    
    # 为所有模型填充区域，使用相同的透明度
    ax.fill(angles, scaled_values, color=color, alpha=0.15)
    
    # 创建图例元素
    legend_elements.append(
        Line2D([0], [0], color=color, lw=line_width, linestyle=line_style, 
               marker=marker, markersize=8, 
               label=f"{model} (avg: {model_avg_auroc[model]:.1f})")
    )

# 创建单一图例，包含所有模型
ax.legend(handles=legend_elements, 
          loc='center left', 
          bbox_to_anchor=(1.05, 0.5),
          fontsize=13, 
          title="Models with Average AUROC", 
          title_fontsize=15)

# 添加标题
plt.title('AUROC Values for All Models Across Different Datasets', size=18, y=1.05)

# 紧凑布局
plt.tight_layout()

# 保存图表
output_path = os.path.join(output_dir, 'auroc_radar_chart_all_models.png')
plt.savefig(output_path, format='png', dpi=300, bbox_inches='tight')
print(f"All models radar chart saved to: {os.path.abspath(output_path)}")

# 保存为SVG
svg_path = os.path.join(output_dir, 'auroc_radar_chart_all_models.svg')
plt.savefig(svg_path, format='svg', bbox_inches='tight')
print(f"All models radar chart also saved as SVG to: {os.path.abspath(svg_path)}")

plt.close()

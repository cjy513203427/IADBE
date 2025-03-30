import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Custom scale function for training time
def custom_scale(y):
    return np.where(y <= 10000, y / 10000 * 0.5, 0.5 + (y - 10000) / 340000 * 0.5)

# Get the script directory
script_dir = os.path.dirname(os.path.abspath(__file__))
# Navigate to the logs directory
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

# Print the absolute paths for debugging
for name, path in file_paths.items():
    print(f"Looking for {name} at: {os.path.abspath(path)}")

# Read the CSV files
dfs = {name: pd.read_csv(path) for name, path in file_paths.items()}

# 获取所有唯一的模型名称
all_models = []
for df in dfs.values():
    all_models.extend(df['model'].tolist())
all_models = sorted(list(set(all_models)))

# 创建用于绘图的数据结构
training_times = {}  # 存储每个模型在每个数据集上的训练时间

# 处理数据
for model in all_models:
    training_times[model] = []
    
    for dataset in sorted_datasets:
        df = dfs[dataset]
        if model in df['model'].values:
            # 获取该模型在当前数据集的训练时间
            model_row = df[df['model'] == model].iloc[0]
            training_time = int(model_row['training time'].rstrip('s'))
            
            training_times[model].append(training_time)
        else:
            # 如果该模型在当前数据集中不存在，填充None
            training_times[model].append(None)

# Plotting
fig, ax = plt.subplots(figsize=(14, 10))

# 每个模型对应一种颜色
colors = plt.cm.tab20(np.linspace(0, 1, len(all_models)))
markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'h', '8', '+', 'x']

# 创建x轴位置
x = np.arange(len(sorted_datasets))

# 为每个模型绘制折线图
for i, model in enumerate(all_models):
    # 过滤掉None值
    valid_indices = [j for j, t in enumerate(training_times[model]) if t is not None]
    valid_x = [x[j] for j in valid_indices]
    valid_times = [training_times[model][j] for j in valid_indices]
    
    if not valid_times:
        continue
    
    # 绘制折线
    ax.plot(valid_x, valid_times, marker=markers[i % len(markers)], 
             markersize=10, color=colors[i], linewidth=2, label=model)

# 设置x轴标签为数据集名称和样本数量
ax.set_xticks(x)
x_labels = [f"{dataset.upper()}\n({dataset_info[dataset]['samples']} samples)" for dataset in sorted_datasets]
ax.set_xticklabels(x_labels)

# 设置y轴为对数刻度，更好地显示不同数量级的训练时间
ax.set_yscale('log')

# 添加标题和标签
ax.set_title('Training Time vs Datasets for Different Models', fontsize=16)
ax.set_xlabel('Datasets (Sorted by Sample Size)', fontsize=14)
ax.set_ylabel('Training Time (seconds, log scale)', fontsize=14)

# 添加图例，放在图外部右侧
ax.legend(title='Models', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)

# 添加网格线
ax.grid(True, linestyle='--', alpha=0.7)

# 紧凑布局
plt.tight_layout()

# Save plot as a PNG
output_path = os.path.join(output_dir, 'training_time_by_dataset.png')
plt.savefig(output_path, format='png', dpi=300, bbox_inches='tight')
print(f"Plot saved to: {os.path.abspath(output_path)}")

# 保存为SVG
svg_path = os.path.join(output_dir, 'training_time_by_dataset.svg')
plt.savefig(svg_path, format='svg', bbox_inches='tight')
print(f"Plot also saved as SVG to: {os.path.abspath(svg_path)}")

# 不显示图形，避免阻塞
# plt.show()

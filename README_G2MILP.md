# G2MILP二分图数据表示框架使用指南

![G2MILP Status](https://img.shields.io/badge/G2MILP-Complete-brightgreen)
![Python](https://img.shields.io/badge/Python-3.8+-blue)
![CVXPY](https://img.shields.io/badge/CVXPY-1.2+-orange)
![Graph Neural Networks](https://img.shields.io/badge/GNN-Ready-purple)

---

## 📋 项目概述

本项目实现了完整的G2MILP (Graph to Mixed Integer Linear Programming) 框架的二分图数据表示方法。该框架能够将混合整数线性规划（MILP）问题转换为二分图表示，为图神经网络在优化问题上的应用提供标准化的数据格式。

### 🎯 核心功能

1. **MILP到二分图转换** - 将CVXPY优化问题转换为标准二分图表示
2. **标准特征提取** - 实现约束节点、变量节点（9维）和边的特征计算  
3. **电力系统语义增强** - 针对电力系统优化问题的专门语义信息
4. **多格式导出** - 支持PyTorch Geometric、DGL等主流图神经网络框架
5. **批量处理** - 高效的批量二分图生成和统计分析
6. **完整集成** - 与现有MILP生成器无缝集成

---

## 🏗️ 核心架构

基于G2MILP论文的标准架构设计：

```
G2MILP二分图表示架构
├── MILP标准形式参数提取
│   ├── 目标函数系数向量 c
│   ├── 约束矩阵 A
│   ├── 右端项向量 b  
│   ├── 变量界限 l, u
│   └── 变量类型信息
├── 二分图节点特征
│   ├── 约束节点特征 (16维)
│   │   ├── 约束类型、右端项、约束方向
│   │   ├── 非零元素数、行密度
│   │   ├── 系数统计（均值、方差、范围）
│   │   ├── 拓扑特征（度数）
│   │   └── 电力系统语义（节点ID、时间段）
│   └── 变量节点特征 (标准9维)
│       ├── 变量类型 (连续/二进制/整数)
│       ├── 目标函数系数
│       ├── 变量界限（上界、下界）
│       ├── 变量度数
│       ├── 系数统计（均值、方差、最大值）
│       └── 归一化索引
├── 边特征 (8维)
│   ├── 原始系数值
│   ├── 系数绝对值和对数变换
│   ├── 行列归一化系数
│   └── 排名特征
└── 格式转换接口
    ├── PyTorch Geometric HeteroData
    ├── DGL异构图
    └── NumPy矩阵格式
```

---

## 📁 文件结构

```
G2MILP核心文件
├── src/models/
│   ├── g2milp_bipartite.py             # G2MILP二分图核心实现
│   ├── biased_milp_generator.py        # 扩展的MILP生成器(集成G2MILP)
│   └── post_disaster_dynamic_multi_period.py  # 基础MILP模型
├── test_g2milp_bipartite.py            # 完整测试套件
├── demo_g2milp_usage.py                # 使用演示脚本
└── README_G2MILP.md                    # 本文档
```

---

## 🚀 快速开始

### 1. 环境准备

```bash
# 确保Python环境已安装
python --version  # 需要 3.8+

# 安装核心依赖
pip install cvxpy numpy pandas scipy scikit-learn

# 可选：图神经网络框架
pip install torch torch-geometric  # PyTorch Geometric
pip install dgl                     # DGL
```

### 2. 基本使用

```python
from src.datasets.loader import load_system_data
from src.models.biased_milp_generator import BiasedMILPGenerator, PerturbationConfig
from src.models.g2milp_bipartite import create_g2milp_generator

# 1. 加载电力系统数据
system_data = load_system_data("data")

# 2. 创建MILP实例
milp_generator = BiasedMILPGenerator(base_system_data=system_data)
config = PerturbationConfig(load_noise_std=0.1, random_seed=42)
instance = milp_generator.generate_single_instance(config)

# 3. 生成G2MILP二分图表示  
success = instance.generate_bipartite_graph(include_power_system_semantics=True)

# 4. 访问二分图数据
bg = instance.bipartite_graph
print(f"约束节点: {bg.n_constraint_nodes}")
print(f"变量节点: {bg.n_variable_nodes}")
print(f"边数量: {bg.n_edges}")
print(f"变量9维特征形状: {bg.variable_feature_matrix.shape}")
```

### 3. 格式转换

```python
# 导出为PyTorch Geometric格式
pyg_data = instance.export_pytorch_geometric()
print(f"节点类型: {pyg_data.node_types}")
print(f"边类型: {pyg_data.edge_types}")

# 导出为DGL格式
dgl_graph = instance.export_dgl_graph()
print(f"约束节点数: {dgl_graph.num_nodes('constraint')}")
print(f"变量节点数: {dgl_graph.num_nodes('variable')}")

# 保存为NumPy格式
import numpy as np
np.savez('g2milp_features.npz',
         constraint_features=bg.constraint_feature_matrix,
         variable_features=bg.variable_feature_matrix,
         edge_features=bg.edge_feature_matrix)
```

---

## 💡 详细使用示例

### 示例1: 单个实例的完整流程

```python
import numpy as np
from src.datasets.loader import load_system_data
from src.models.biased_milp_generator import BiasedMILPGenerator, PerturbationConfig

# 加载数据
system_data = load_system_data("data")

# 创建生成器
generator = BiasedMILPGenerator(base_system_data=system_data)

# 配置扰动参数
config = PerturbationConfig(
    load_perturbation_type="gaussian",
    load_noise_std=0.1,                    # 负荷10%扰动
    generator_noise_std=0.05,              # 发电机5%扰动
    pv_noise_std=0.15,                     # 光伏15%扰动
    perturbation_intensity=1.0,            # 100%扰动强度
    random_seed=42                         # 可重现
)

# 生成MILP实例
instance = generator.generate_single_instance(
    perturbation_config=config,
    instance_id="example_instance",
    n_periods=21,
    save_to_file=True
)

# 生成二分图表示
success = instance.generate_bipartite_graph(include_power_system_semantics=True)

if success:
    bg = instance.bipartite_graph
    
    # 分析变量节点9维特征
    variable_features = bg.variable_feature_matrix
    feature_names = [
        "变量类型", "目标系数", "下界", "上界", "度数",
        "系数均值", "系数标准差", "系数最大值", "索引归一化"
    ]
    
    for i, name in enumerate(feature_names):
        values = variable_features[:, i]
        print(f"{name}: 均值={np.mean(values):.3f}, "
              f"范围=[{np.min(values):.3f}, {np.max(values):.3f}]")
    
    # 保存二分图
    instance.save_bipartite_graph("output/example_bipartite.pkl")
```

### 示例2: 批量处理和统计分析

```python
from src.models.biased_milp_generator import create_scenario_perturbation_configs

# 创建多种扰动场景
scenario_configs = create_scenario_perturbation_configs()

# 选择几个场景
selected_scenarios = {
    "负荷高峰": scenario_configs["load_peak"],
    "光伏不稳定": scenario_configs["pv_unstable"],
    "设备故障": scenario_configs["equipment_failure"]
}

# 批量生成MILP实例
scenario_instances = generator.generate_scenario_instances(
    scenario_configs=selected_scenarios,
    save_to_file=True
)

# 批量生成二分图
instances_list = list(scenario_instances.values())
updated_instances = generator.generate_bipartite_graphs_for_instances(
    instances=instances_list,
    include_power_system_semantics=True,
    save_graphs=True,
    graph_output_dir="output/scenario_graphs"
)

# 统计分析
analysis = generator.analyze_bipartite_graph_statistics(updated_instances)
print(f"有效图数量: {analysis['valid_bipartite_graphs']}")
print(f"平均约束节点: {analysis['constraint_nodes_stats']['mean']:.1f}")
print(f"平均变量节点: {analysis['variable_nodes_stats']['mean']:.1f}")
print(f"平均图密度: {analysis['density_stats']['mean']:.6f}")
```

### 示例3: 直接使用G2MILP生成器

```python
from src.models.g2milp_bipartite import create_g2milp_generator

# 创建G2MILP生成器
g2milp_generator = create_g2milp_generator(include_power_system_semantics=True)

# 从现有MILP实例生成二分图
bipartite_graph = g2milp_generator.generate_from_milp_instance(instance)

# 从CVXPY问题直接生成
bipartite_graph = g2milp_generator.generate_from_cvxpy_problem(
    cvxpy_problem=instance.cvxpy_problem,
    instance_id="direct_generation",
    system_data=instance.perturbed_system_data
)

# 访问详细信息
print(f"图统计信息: {bipartite_graph.graph_statistics}")
print(f"约束特征矩阵形状: {bipartite_graph.constraint_feature_matrix.shape}")
print(f"变量特征矩阵形状: {bipartite_graph.variable_feature_matrix.shape}")
print(f"边特征矩阵形状: {bipartite_graph.edge_feature_matrix.shape}")
```

---

## 📊 数据格式说明

### 二分图结构

G2MILP二分图 G = (V_c ∪ V_v, E)，其中：
- **V_c**: 约束节点集合 {c_1, c_2, ..., c_m}
- **V_v**: 变量节点集合 {v_1, v_2, ..., v_n}  
- **E**: 边集合 ⊆ V_c × V_v

### 变量节点9维特征向量

根据G2MILP论文标准，变量节点具有以下9维特征：

| 维度 | 特征名称 | 描述 | 取值范围 |
|------|----------|------|----------|
| 1 | 变量类型 | 0.0=连续, 1.0=二进制, 0.5=整数 | [0.0, 1.0] |
| 2 | 目标函数系数 | 目标函数中的系数 c_j | ℝ |
| 3 | 下界 | 变量下界 l_j | ℝ ∪ {-∞} |
| 4 | 上界 | 变量上界 u_j | ℝ ∪ {+∞} |
| 5 | 变量度数 | 变量出现的约束数量 | ℕ |
| 6 | 系数均值 | 该变量在所有约束中系数的均值 | ℝ |
| 7 | 系数标准差 | 该变量在所有约束中系数的标准差 | ℝ⁺ |
| 8 | 系数最大值 | 该变量系数的最大绝对值 | ℝ⁺ |
| 9 | 归一化索引 | 归一化的变量索引 j/n | [0, 1] |

### 约束节点特征 (16维)

| 特征类别 | 特征名称 | 描述 |
|----------|----------|------|
| 基础信息 | 约束类型 | 0=等式, 1=不等式(≤), 2=不等式(≥), 3=边界 |
| | 右端项值 | 约束右端项 b_i |
| | 约束方向 | 0=≤, 1=≥, 2== |
| 规模信息 | 非零元素数 | 约束中非零系数数量 |
| | 行密度 | 非零元素/总变量数 |
| 统计特征 | 系数和/均值/标准差/最大值/最小值/范围 | 约束系数的统计信息 |
| 拓扑特征 | 约束度数 | 连接的变量数量 |
| 归一化特征 | 归一化右端项/系数和 | 全局归一化的特征 |
| 电力系统语义 | 节点ID | 关联的电力节点ID (-1表示非节点约束) |
| | 时间段 | 时间段编号 (-1表示非时变约束) |

### 边特征 (8维)

| 特征名称 | 描述 |
|----------|------|
| 系数值 | 约束矩阵中的原始系数 A_ij |
| 系数绝对值 | \|A_ij\| |
| 对数绝对值 | log(\|A_ij\| + ε) |
| 行归一化系数 | 按约束归一化的系数 |
| 列归一化系数 | 按变量归一化的系数 |
| 全局归一化系数 | 全局归一化的系数 |
| 行内排名 | 在该约束中的系数排名 [0,1] |
| 列内排名 | 在该变量中的系数排名 [0,1] |

---

## 🔧 高级功能

### 电力系统语义增强

针对电力系统优化问题，框架提供专门的语义增强功能：

```python
# 启用电力系统语义增强
bipartite_graph = g2milp_generator.generate_from_milp_instance(
    instance, 
    include_power_system_semantics=True  # 启用语义增强
)

# 访问语义信息
for i, constraint_feature in enumerate(bipartite_graph.constraint_features):
    if constraint_feature.bus_id >= 0:
        print(f"约束 {i} 关联节点: {constraint_feature.bus_id}")
    if constraint_feature.time_period >= 0:
        print(f"约束 {i} 时间段: {constraint_feature.time_period}")
    print(f"约束类别: {constraint_feature.constraint_category}")

for j, variable_feature in enumerate(bipartite_graph.variable_features):
    print(f"变量 {j} 语义: {variable_feature.var_semantic}")
```

### 自定义特征提取

```python
from src.models.g2milp_bipartite import BipartiteGraphBuilder, MILPDataExtractor

# 创建自定义构建器
class CustomBipartiteGraphBuilder(BipartiteGraphBuilder):
    def _extract_custom_features(self, constraint_id, system_data):
        # 自定义特征提取逻辑
        pass

# 使用自定义构建器
extractor = MILPDataExtractor()
milp_data = extractor.extract_problem_data(instance.cvxpy_problem)

custom_builder = CustomBipartiteGraphBuilder(include_power_system_semantics=True)
custom_bipartite_graph = custom_builder.build_bipartite_graph(
    milp_data, instance.instance_id, instance.perturbed_system_data
)
```

### 性能优化配置

```python
# 大规模问题的内存优化
generator = BiasedMILPGenerator(
    base_system_data=system_data,
    output_dir="output/large_scale"
)

# 批量处理时的性能监控
import time
import psutil

start_time = time.time()
start_memory = psutil.Process().memory_info().rss

# 执行批量处理
instances = generator.generate_batch_instances(configs, save_to_file=False)
updated_instances = generator.generate_bipartite_graphs_for_instances(
    instances, save_graphs=False
)

end_time = time.time()
end_memory = psutil.Process().memory_info().rss

print(f"处理时间: {end_time - start_time:.2f} 秒")
print(f"内存使用: {(end_memory - start_memory) / 1024 / 1024:.2f} MB")
```

---

## 🧪 测试和验证

### 运行测试套件

```bash
# 运行完整测试套件
python test_g2milp_bipartite.py
# 选择: 1 (运行测试套件)

# 运行使用示例
python test_g2milp_bipartite.py  
# 选择: 2 (运行使用示例)

# 运行演示脚本
python demo_g2milp_usage.py
```

### 测试覆盖范围

- ✅ **MILP数据提取测试** - 验证CVXPY问题数据提取的正确性
- ✅ **二分图构建测试** - 验证图结构和特征矩阵的一致性  
- ✅ **9维特征向量测试** - 验证变量节点标准特征的正确性
- ✅ **约束特征测试** - 验证约束节点16维特征的完整性
- ✅ **边特征测试** - 验证边特征计算和排名的准确性
- ✅ **格式转换测试** - 测试PyTorch Geometric和DGL格式导出
- ✅ **批量处理测试** - 验证批量生成和统计分析功能
- ✅ **序列化测试** - 测试保存和加载的数据一致性
- ✅ **集成测试** - 验证与现有MILP生成器的集成
- ✅ **性能测试** - 监控内存使用和处理速度

---

## 📈 输出文件格式

### 二分图文件

```
output/
├── bipartite_graphs/
│   ├── instance_id_bipartite.pkl      # 完整二分图对象
│   └── instance_id_bipartite.json     # 二分图元信息
├── features/
│   ├── g2milp_features.npz           # NumPy特征矩阵
│   ├── g2milp_pyg_data.pt            # PyTorch Geometric数据
│   └── g2milp_dgl_graph.pkl          # DGL图数据
└── analysis/
    ├── feature_analysis.json          # 特征分析报告
    ├── batch_processing_results.json  # 批量处理结果
    └── demo_summary_report.json       # 演示总结报告
```

### 二分图元信息示例

```json
{
  "milp_instance_id": "example_instance",
  "generation_timestamp": "2025-06-29T10:30:00",
  "n_constraint_nodes": 8756,
  "n_variable_nodes": 15240,
  "n_edges": 45623,
  "bipartite_density": 0.000342,
  "graph_statistics": {
    "avg_constraint_degree": 5.21,
    "avg_variable_degree": 2.99,
    "variable_type_distribution": {
      "continuous": 14820,
      "binary": 420,
      "integer": 0
    }
  }
}
```

### 特征分析报告示例

```json
{
  "variable_features_analysis": {
    "变量类型": {"mean": 0.028, "std": 0.164, "min": 0.0, "max": 1.0},
    "目标函数系数": {"mean": 245.67, "std": 1456.23, "min": 0.0, "max": 9999.0},
    "变量度数": {"mean": 2.99, "std": 1.87, "min": 0.0, "max": 12.0}
  },
  "constraint_features_analysis": {
    "type_distribution": {"0": 4523, "1": 4233},
    "row_density_stats": {"mean": 0.0034, "std": 0.0015}
  }
}
```

---

## 🎪 技术特色

### 🔧 标准化特征提取

- **变量节点9维特征向量** - 严格按照G2MILP论文标准实现
- **丰富的约束节点特征** - 16维特征包含完整的约束信息
- **多层次边特征** - 8维边特征涵盖系数、排名、归一化信息
- **电力系统语义增强** - 专门针对电力系统优化问题的语义信息

### 🛡️ 鲁棒性设计

- **完整的错误处理** - 全面的异常捕获和错误提示
- **数据类型验证** - 严格的输入输出数据格式检查
- **边界条件处理** - 处理空图、单节点等极端情况
- **内存管理** - 大规模问题的内存优化策略

### 📊 多格式支持

- **PyTorch Geometric** - 标准的异构图数据格式
- **DGL** - 深度图库的异构图格式
- **NumPy** - 原始特征矩阵格式
- **JSON** - 人类可读的元信息格式

### 🔄 无缝集成

- **现有框架兼容** - 与BiasedMILPGenerator完美集成
- **模块化设计** - 松耦合、可扩展的架构
- **标准化接口** - 统一的API设计模式
- **向后兼容** - 不影响现有功能的扩展方式

---

## 📋 应用场景

### 🔬 图神经网络研究

```python
# 为GNN训练准备数据
pyg_data = instance.export_pytorch_geometric()

# 图卷积网络示例
import torch
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GCNConv

class G2MILP_GNN(torch.nn.Module):
    def __init__(self, constraint_dim=16, variable_dim=9, hidden_dim=64):
        super().__init__()
        
        self.convs = torch.nn.ModuleList([
            HeteroConv({
                ('constraint', 'connects', 'variable'): GCNConv(constraint_dim, hidden_dim),
                ('variable', 'connected_by', 'constraint'): GCNConv(variable_dim, hidden_dim),
            }),
            HeteroConv({
                ('constraint', 'connects', 'variable'): GCNConv(hidden_dim, hidden_dim),
                ('variable', 'connected_by', 'constraint'): GCNConv(hidden_dim, hidden_dim),
            })
        ])
        
    def forward(self, x_dict, edge_index_dict):
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: F.relu(x) for key, x in x_dict.items()}
        return x_dict

# 使用模型
model = G2MILP_GNN()
out = model(pyg_data.x_dict, pyg_data.edge_index_dict)
```

### 🏭 优化算法研究

```python
# 分析不同扰动对图结构的影响
scenarios = ["负荷高峰", "光伏不稳定", "设备故障"]
graph_metrics = {}

for scenario in scenarios:
    instance = scenario_instances[scenario]
    bg = instance.bipartite_graph
    
    graph_metrics[scenario] = {
        'density': bg.graph_statistics['bipartite_density'],
        'avg_constraint_degree': bg.graph_statistics['avg_constraint_degree'],
        'avg_variable_degree': bg.graph_statistics['avg_variable_degree'],
        'variable_type_ratio': bg.graph_statistics['variable_type_distribution']
    }

# 分析图结构与优化难度的关系
import matplotlib.pyplot as plt
densities = [metrics['density'] for metrics in graph_metrics.values()]
degrees = [metrics['avg_variable_degree'] for metrics in graph_metrics.values()]

plt.scatter(densities, degrees)
plt.xlabel('Graph Density')
plt.ylabel('Average Variable Degree')
plt.title('Graph Structure Analysis')
plt.show()
```

### 📚 算法性能基准测试

```python
# 创建标准化的测试集
test_configs = create_default_perturbation_configs()
test_instances = generator.generate_batch_instances(
    perturbation_configs=test_configs,
    instance_prefix="benchmark_test"
)

# 生成二分图测试集
test_instances = generator.generate_bipartite_graphs_for_instances(
    test_instances, save_graphs=True
)

# 评估不同GNN架构的性能
def evaluate_gnn_performance(model, test_data):
    """评估GNN模型在MILP问题上的性能"""
    model.eval()
    total_loss = 0
    
    for data in test_data:
        pyg_data = data.export_pytorch_geometric()
        output = model(pyg_data.x_dict, pyg_data.edge_index_dict)
        # 计算损失（根据具体任务定义）
        loss = compute_task_loss(output, pyg_data)
        total_loss += loss.item()
    
    return total_loss / len(test_data)

# 建立基准测试结果
benchmark_results = {
    'test_set_size': len(test_instances),
    'avg_graph_size': np.mean([inst.bipartite_graph.n_edges for inst in test_instances]),
    'model_performance': {}
}
```

---

## 🔮 未来扩展

### 🚀 功能扩展

- [ ] **增量图更新** - 支持MILP问题的增量修改和图更新
- [ ] **图压缩算法** - 大规模图的压缩存储和快速访问
- [ ] **多目标优化支持** - 扩展到多目标MILP问题的表示
- [ ] **动态图表示** - 支持时间序列MILP问题的动态图建模

### 📈 性能优化

- [ ] **并行化处理** - 多进程/多线程的批量图生成
- [ ] **GPU加速** - 利用GPU加速大规模特征计算
- [ ] **内存映射** - 超大规模图的内存映射存储
- [ ] **分布式处理** - 支持集群环境的分布式图生成

### 🔌 接口扩展

- [ ] **REST API** - 提供Web服务接口
- [ ] **数据库集成** - 与图数据库的集成支持
- [ ] **可视化工具** - 交互式二分图可视化界面
- [ ] **标准化数据集** - 构建标准的MILP-图数据集

### 🎯 算法研究支持

- [ ] **图增强技术** - 图数据增强和噪声注入
- [ ] **图嵌入预训练** - 预训练的图嵌入模型
- [ ] **迁移学习支持** - 跨领域MILP问题的迁移学习
- [ ] **自监督学习** - 基于图结构的自监督学习框架

---

## ⚠️ 注意事项

### 🔧 环境要求

- **Python版本**: 3.8或更高版本
- **内存要求**: 建议8GB+（大规模问题需要更多内存）
- **存储空间**: 二分图文件可能较大，注意磁盘空间
- **依赖库**: CVXPY、NumPy、SciPy、Pandas

### 📊 性能考虑

- **计算复杂度**: 特征计算复杂度为O(|E| + |V|log|V|)
- **内存使用**: 约为原MILP问题内存的2-3倍
- **图规模限制**: 单个图建议不超过100万个节点
- **批量处理**: 大批量时建议分批处理避免内存溢出

### 🛡️ 数据安全

- **敏感信息**: 生成的图包含完整的MILP信息，注意数据保护
- **文件权限**: 设置适当的文件访问权限
- **版本兼容**: 不同版本间的序列化格式可能不兼容

### 🔍 调试建议

```python
# 启用详细日志
import logging
logging.basicConfig(level=logging.DEBUG)

# 检查数据一致性
def validate_bipartite_graph(bg):
    """验证二分图数据的一致性"""
    assert bg.n_constraint_nodes > 0, "约束节点数必须大于0"
    assert bg.n_variable_nodes > 0, "变量节点数必须大于0"
    assert bg.variable_feature_matrix.shape[1] == 9, "变量特征必须为9维"
    assert len(bg.edges) == bg.n_edges, "边数量不一致"
    print("✅ 二分图数据验证通过")

# 性能监控
import psutil
import time

def monitor_performance(func):
    """性能监控装饰器"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss
        
        result = func(*args, **kwargs)
        
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss
        
        print(f"执行时间: {end_time - start_time:.2f}秒")
        print(f"内存增量: {(end_memory - start_memory)/1024/1024:.2f}MB")
        
        return result
    return wrapper
```

---

## 🎉 总结

G2MILP二分图数据表示框架为混合整数线性规划问题的图神经网络研究提供了完整的解决方案。通过标准化的特征提取、多格式支持和无缝集成设计，该框架能够高效地将复杂的优化问题转换为适合机器学习的图数据格式。

### ✅ 主要优势

1. **标准化实现** - 严格按照G2MILP论文标准实现，确保与学术界接轨
2. **完整集成** - 与现有MILP生成框架无缝集成，无需重构现有代码
3. **多格式支持** - 支持主流图神经网络框架，降低使用门槛
4. **电力系统优化** - 针对电力系统优化问题进行专门优化
5. **生产就绪** - 完整的测试覆盖、错误处理和性能优化

### 🎯 适用场景

- **学术研究**: 图神经网络在优化问题上的应用研究
- **工业应用**: 电力系统智能调度算法开发
- **算法竞赛**: 优化算法性能评估和对比
- **教学演示**: 优化理论与机器学习结合的教学案例

通过这个框架，研究者和工程师可以专注于图神经网络算法的创新，而无需担心底层数据表示的复杂性。这为优化问题的智能求解开辟了新的可能性。

---

🚀 **立即开始使用G2MILP框架，探索图神经网络在优化问题中的无限潜力！**

---

**版本**: v1.0.0  
**更新日期**: 2025年6月29日  
**开发状态**: ✅ 生产就绪  
**维护状态**: 🔄 持续维护
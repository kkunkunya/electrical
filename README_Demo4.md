# Demo 4: G2MILP 实例生成

## 项目概述

Demo 4实现了G2MILP（Graph to Mixed Integer Linear Programming）的核心生成逻辑，以Demo 3生成的"有偏差的"二分图为基础，生成新的、结构和难度相似但可能更优的MILP实例。

## 技术架构

### 1. 核心组件

```
src/models/g2milp/
├── __init__.py          # 模块导入和接口
├── encoder.py           # GNN编码器
├── decoder.py           # 解码器(4个预测器)
├── masking.py           # 约束遮盖过程
├── generator.py         # 主生成器
├── training.py          # 训练逻辑
└── inference.py         # 推理逻辑
```

### 2. 技术特性

- **深度学习框架**: PyTorch + PyTorch Geometric
- **模型架构**: Masked VAE (Variational Autoencoder)
- **图神经网络**: 支持GCN、GAT、GraphSAGE
- **异构图处理**: 约束节点-变量节点二分图
- **可控生成**: 通过η参数控制相似度vs创新性

## 实现详情

### 编码器 (Encoder)

基于图神经网络的变分编码器，将二分图编码为潜向量分布：

```python
q_φ(Z|G) = Π q_φ(z_u|G)
z_u ~ N(μ_φ(h_u^G), exp(Σ_φ(h_u^G)))
```

**关键特性**：
- 异构图卷积层
- 重参数化技巧
- KL散度正则化
- 支持多种GNN架构

### 解码器 (Decoder)

包含四个协作的预测器模块：

1. **Bias Predictor**: 预测约束右端项 b_v
2. **Degree Predictor**: 预测约束度数 d_v  
3. **Logits Predictor**: 预测变量连接概率 δ_{v,u}
4. **Weights Predictor**: 预测边权重 e_{v,u}

### 遮盖过程 (Masking)

实现论文中的约束遮盖机制：

1. 随机选择约束节点进行遮盖
2. 用特殊[mask]标记替换
3. 添加虚拟边连接到所有变量节点
4. 为遮盖元素分配特殊嵌入

### 生成流程

根据论文算法2实现迭代生成过程：

```python
for iteration in range(η * |V|):
    1. 随机选择约束节点遮盖
    2. 从先验分布采样潜变量Z
    3. 解码生成新约束
    4. 替换原约束
```

## 配置参数

### 模型配置

```json
{
  "model": {
    "constraint_feature_dim": 16,
    "variable_feature_dim": 9,
    "edge_feature_dim": 8
  },
  "encoder": {
    "gnn_type": "GCN",
    "hidden_dim": 128,
    "latent_dim": 64,
    "num_layers": 3,
    "dropout": 0.1
  },
  "decoder": {
    "hidden_dim": 128,
    "latent_dim": 64,
    "num_layers": 2,
    "predictor_hidden_dim": 64
  }
}
```

### 训练配置

```json
{
  "training": {
    "num_epochs": 500,
    "iterations_per_epoch": 50,
    "learning_rate": 1e-3,
    "kl_annealing": true,
    "early_stopping": true
  }
}
```

### 推理配置

```json
{
  "inference": {
    "eta": 0.1,
    "temperature": 1.0,
    "sample_from_prior": true,
    "num_test_instances": 5
  }
}
```

## 使用方法

### 1. 环境设置

```bash
# 激活虚拟环境
source venv_linux/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 2. 运行完整流程

```bash
# 运行Demo 4主程序
python demo4_g2milp_instance_generation.py
```

### 3. 运行测试

```bash
# 基础测试（无PyTorch依赖）
python test_demo4_basic.py

# 完整测试（需要PyTorch）
python test_demo4_g2milp.py
```

## 输出结果

### 目录结构

```
output/demo4_g2milp/
├── training/
│   ├── final_model.pth
│   ├── training_history.json
│   └── training_curves.png
├── inference/
│   ├── generated_instances/
│   ├── generation_results.pkl
│   └── multi_instance_analysis.json
├── plots/
│   ├── similarity_distribution.png
│   ├── edge_count_changes.png
│   └── generation_iterations.png
└── demo4_summary.json
```

### 关键输出

1. **训练模型**: `training/final_model.pth`
2. **生成实例**: `inference/generated_instances/`
3. **分析报告**: `generation_results.pkl`
4. **可视化图表**: `plots/`

## 性能指标

### 生成质量评估

- **相似度指标**: 与源实例的结构相似度
- **多样性指标**: 生成实例间的差异性
- **约束有效性**: 生成约束的数学有效性
- **优化潜力**: 相对于源实例的改进程度

### 控制参数

- **η (eta)**: 遮盖比例，控制相似度vs创新性
  - η=0.05: 高相似度，小幅改进
  - η=0.1: 平衡相似度和创新性
  - η=0.2: 较大结构变化

## 技术亮点

### 1. 异构图神经网络

专门为约束-变量二分图设计的GNN架构，能够有效处理MILP的异构结构。

### 2. Masked VAE范式

创新性地将自然语言处理中的掩码机制引入到MILP生成，实现可控的实例生成。

### 3. 多预测器协作

四个专门的预测器协作工作，分别负责约束的不同方面，提高生成质量。

### 4. 自我学习训练

基于单个有偏差实例进行自我学习训练，无需大量标注数据。

## 与Demo 3的集成

Demo 4完全基于Demo 3的输出：

1. **输入**: Demo 3生成的二分图表示
2. **处理**: 转换为PyTorch Geometric格式
3. **训练**: 使用该实例进行自我学习
4. **生成**: 产生结构相似的新实例

## 扩展可能性

### 1. 多实例训练

当有多个训练实例时，可以扩展为标准的监督学习。

### 2. 强化学习集成

结合强化学习优化生成策略，直接优化求解性能。

### 3. 在线学习

实现在线学习机制，根据求解反馈不断改进生成质量。

### 4. 迁移学习

将训练好的模型迁移到不同领域的MILP问题。

## 故障排除

### 常见问题

1. **PyTorch未安装**
   ```bash
   pip install torch torch-geometric
   ```

2. **Demo 3数据缺失**
   ```bash
   python demo3_g2milp_bipartite_conversion.py
   ```

3. **CUDA内存不足**
   - 减小batch_size或hidden_dim
   - 使用CPU模式：设置device="cpu"

4. **训练不收敛**
   - 调整学习率和KL权重
   - 增加训练轮数
   - 检查数据质量

### 日志分析

所有运行日志保存在`logs/`目录，包含详细的调试信息：

- 模型参数统计
- 训练损失曲线
- 生成过程追踪
- 错误堆栈信息

## 论文引用

基于以下研究工作：

```
@article{g2milp2023,
  title={Learning to Generate Mixed Integer Linear Programming Instances},
  author={...},
  journal={...},
  year={2023}
}
```

## 开发团队

Demo 4 G2MILP实现团队

## 版本历史

- **v1.0.0**: 初始实现，完整的G2MILP生成框架
- 支持基础的编码-解码架构
- 实现四个预测器协作机制
- 完整的训练和推理流程
- 与Demo 3的无缝集成
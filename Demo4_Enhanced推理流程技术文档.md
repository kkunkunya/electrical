# Demo 4 Enhanced 推理流程技术文档

## 文档概述

本文档详细描述 `demo4_g2milp_enhanced.py` 中实现的G2MILP深度学习模型推理流程，为后续开发工作提供技术指引。

---

## 1. 整体架构概览

### 1.1 推理流程图
```
训练完成的模型 → 数据加载 → 格式转换 → 推理生成 → 质量评估 → 结果保存
      ↓              ↓         ↓         ↓         ↓         ↓
   G2MILPGenerator → Demo3数据 → HeteroData → 新实例 → 评估报告 → JSON/PKL
```

### 1.2 核心组件
- **数据加载器**: `load_bipartite_data()`
- **格式转换器**: `load_and_convert_demo3_data()`
- **推理引擎**: `enhanced_inference()`
- **质量评估器**: `enhanced_evaluation()`
- **结果管理器**: `save_enhanced_results()`

---

## 2. 数据加载与转换流程

### 2.1 数据源
```python
# 输入数据路径
data_path = "output/demo3_g2milp/bipartite_graphs/demo3_bipartite_graph.pkl"

# 数据格式
源格式: BipartiteGraph (Demo 3输出)
目标格式: HeteroData (PyTorch Geometric)
```

### 2.2 转换核心步骤

#### 步骤1: 数据加载
```python
def load_bipartite_data(data_path: str) -> Optional[Dict[str, Any]]:
    with open(bipartite_file, 'rb') as f:
        bipartite_graph = pickle.load(f)
    
    # 验证数据完整性
    if not hasattr(bipartite_graph, 'constraint_nodes'):
        return None
```

#### 步骤2: 内联格式转换
```python
def load_and_convert_demo3_data(data_path: str, device: str = "cuda"):
    # 2.1 提取节点和边信息
    constraint_nodes = bipartite_graph.constraint_nodes  # 约束节点
    variable_nodes = bipartite_graph.variable_nodes      # 变量节点  
    edges = bipartite_graph.edges                        # 边连接
    
    # 2.2 特征矩阵构建
    constraint_features = 提取约束特征(constraint_nodes)  # → (N_c, 16)
    variable_features = 提取变量特征(variable_nodes)      # → (N_v, 9)
    edge_indices, edge_features = 提取边特征(edges)       # → (2, E), (E, 8)
```

#### 步骤3: 数值稳定性处理
```python
# 3.1 NaN/Inf值处理
constraint_features = np.nan_to_num(constraint_features, nan=0.0, posinf=1.0, neginf=-1.0)
variable_features = np.nan_to_num(variable_features, nan=0.0, posinf=1.0, neginf=-1.0)
edge_features = np.nan_to_num(edge_features, nan=0.0, posinf=1.0, neginf=-1.0)

# 3.2 特征归一化
constraint_std = np.std(constraint_features, axis=0) + 1e-8
constraint_features = constraint_features / constraint_std
constraint_features = np.clip(constraint_features, -5.0, 5.0)

# 3.3 应用到所有特征类型
# (同样处理variable_features和edge_features)
```

#### 步骤4: PyTorch张量创建
```python
# 4.1 节点特征张量
data['constraint'].x = torch.tensor(constraint_features, dtype=torch.float32, device=device)
data['variable'].x = torch.tensor(variable_features, dtype=torch.float32, device=device)

# 4.2 边连接张量
data['constraint', 'connects', 'variable'].edge_index = torch.tensor(edge_indices, dtype=torch.long, device=device)
data['constraint', 'connects', 'variable'].edge_attr = torch.tensor(edge_features, dtype=torch.float32, device=device)

# 4.3 反向边连接（G2MILP模型要求）
reverse_edge_indices = torch.stack([
    torch.tensor(edge_indices[1], dtype=torch.long, device=device),
    torch.tensor(edge_indices[0], dtype=torch.long, device=device)
], dim=0)
data['variable', 'connected_by', 'constraint'].edge_index = reverse_edge_indices
```

### 2.3 输出数据结构
```python
转换结果 = {
    'bipartite_data': HeteroData对象,
    'metadata': {
        'source': 'demo3_bipartite_graph',
        'num_constraints': int,
        'num_variables': int, 
        'num_edges': int,
        'device': str
    },
    'extraction_summary': {
        'conversion_method': 'demo3_to_demo4_inline',
        'bidirectional_edges': True,
        'timestamp': str
    }
}
```

---

## 3. 推理生成流程

### 3.1 推理配置
```python
inference_config = InferenceConfig(
    eta=0.1,                              # 遮盖比例：10%的约束被遮盖
    num_test_instances=5,                 # 生成实例数量
    temperature=1.0,                      # 采样温度
    sample_from_prior=True,               # 从先验分布采样
    constraint_selection_strategy="random", # 约束选择策略
    diversity_boost=True,                 # 启用多样性增强
    num_diverse_samples=3,                # 多样性样本数
    compute_similarity_metrics=True,      # 计算相似度指标
    generate_comparison_report=True,      # 生成对比报告
    experiment_name=f"enhanced_inference_{timestamp}"
)
```

### 3.2 推理引擎核心流程

#### 步骤1: 推理器初始化
```python
def enhanced_inference(generator, training_data, configs):
    # 创建推理引擎
    inference_engine = G2MILPInference(generator, configs['inference'])
```

#### 步骤2: 实例生成
```python
    # 执行推理生成
    inference_results = inference_engine.generate_instances(
        training_data['bipartite_data'],              # 输入数据
        num_samples=configs['inference'].num_test_instances  # 生成数量
    )
```

#### 步骤3: 结果分析
```python
    # 提取生成结果
    generated_samples = inference_results['generated_instances']
    generation_info = inference_results['generation_info']
    
    # 多样性统计分析
    for i, info in enumerate(generation_info):
        if 'diversity_stats' in info:
            stats = info['diversity_stats']
            # 分析偏置标准差、度数标准差、连接标准差、约束多样性等
```

### 3.3 推理结果结构
```python
inference_results = {
    'generated_instances': [
        # 生成的实例列表，每个实例是HeteroData对象
    ],
    'generation_info': [
        {
            'diversity_stats': {
                'bias_std': float,
                'degree_std': float, 
                'connection_std': float,
                'unique_constraints_ratio': float
            },
            'generation_time': float,
            'generation_parameters': dict
        }
    ],
    'config': InferenceConfig对象
}
```

---

## 4. 质量评估流程

### 4.1 评估器配置
```python
evaluation_config = EvaluationConfig(
    num_evaluation_samples=3,             # 评估样本数
    similarity_metrics=['cosine', 'l2'],  # 相似度指标
    diversity_metrics=['std', 'entropy'], # 多样性指标
    structural_metrics=True,              # 结构指标
    save_detailed_analysis=True           # 保存详细分析
)
```

### 4.2 评估核心流程

#### 步骤1: 评估器初始化
```python
def enhanced_evaluation(original_data, inference_results, configs):
    evaluator = G2MILPEvaluator(configs['evaluation'])
```

#### 步骤2: 多维度质量评估
```python
    evaluation_results = evaluator.evaluate_comprehensive(
        original_data=original_data['bipartite_data'],
        generated_instances=inference_results['generated_instances'],
        generation_info=inference_results['generation_info']
    )
```

#### 步骤3: 质量指标计算
```python
# 4.2.1 结构相似度
structural_similarity = 计算图结构相似度(原始图, 生成图)

# 4.2.2 特征分布相似度  
feature_similarity = 计算特征分布相似度(原始特征, 生成特征)

# 4.2.3 多样性指标
diversity_score = 计算生成样本间多样性()

# 4.2.4 有效性检验
validity_score = 验证生成实例的有效性()

# 4.2.5 综合质量得分
overall_quality = 加权平均(结构相似度, 特征相似度, 多样性, 有效性)
```

### 4.3 评估结果结构
```python
evaluation_results = {
    'overall_quality_score': float,       # 综合质量得分 [0,1]
    'structural_similarity': float,       # 结构相似度 [0,1]  
    'feature_similarity': float,          # 特征相似度 [0,1]
    'diversity_score': float,             # 多样性得分 [0,1]
    'validity_score': float,              # 有效性得分 [0,1]
    'detailed_metrics': {
        'graph_metrics': dict,            # 图级别指标
        'node_metrics': dict,             # 节点级别指标
        'edge_metrics': dict              # 边级别指标
    },
    'comparison_analysis': dict,          # 对比分析
    'evaluation_timestamp': str
}
```

---

## 5. 结果管理流程

### 5.1 结果保存策略
```python
def save_enhanced_results(training_results, inference_results, evaluation_results, output_dir):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = output_dir / f"enhanced_results_{timestamp}"
```

### 5.2 保存的文件类型

#### 5.2.1 训练结果 (`training_results.json`)
```json
{
    "training_summary": {
        "training_time_seconds": float,
        "best_validation_loss": float,
        "total_iterations": int
    },
    "training_history": {
        "train_loss": [float],
        "val_loss": [float],
        "quality_scores": [float]
    }
}
```

#### 5.2.2 推理结果 (`inference_results.json`)
```json
{
    "num_samples": int,
    "generation_info": [
        {
            "diversity_stats": dict,
            "generation_time": float
        }
    ],
    "inference_config": dict,
    "timestamp": str
}
```

#### 5.2.3 评估结果 (`evaluation_results.json`)
```json
{
    "overall_quality_score": float,
    "detailed_metrics": dict,
    "comparison_analysis": dict,
    "evaluation_timestamp": str
}
```

#### 5.2.4 生成数据 (`generated_instances.pkl`)
```python
# 二进制格式保存生成的图数据
generated_instances = [
    HeteroData对象1,
    HeteroData对象2,
    ...
]
```

---

## 6. 关键技术要点

### 6.1 数值稳定性保障
```python
# 6.1.1 异常值处理
features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)

# 6.1.2 归一化稳定性
std = np.std(features, axis=0) + 1e-8  # 防止除零

# 6.1.3 值域限制  
features = np.clip(features, -5.0, 5.0)  # 防止梯度爆炸
```

### 6.2 设备管理
```python
# 6.2.1 自动设备检测
device = "cuda" if torch.cuda.is_available() else "cpu"

# 6.2.2 张量设备一致性
all_tensors.to(device)

# 6.2.3 内存优化
torch.cuda.empty_cache()  # 释放GPU缓存
```

### 6.3 错误恢复机制
```python
try:
    # 推理操作
    result = inference_engine.generate_instances(...)
except Exception as e:
    logger.error(f"推理失败: {e}")
    # 清理资源
    # 记录错误状态
    raise  # 重新抛出异常供上层处理
```

---

## 7. 性能优化要点

### 7.1 计算效率
- **批量处理**: 尽可能使用批量操作减少循环
- **张量运算**: 优先使用PyTorch原生张量操作
- **GPU加速**: 充分利用CUDA加速计算

### 7.2 内存管理
- **延迟加载**: 按需加载大型数据
- **及时释放**: 不再使用的张量及时释放
- **梯度控制**: 推理时关闭梯度计算 (`torch.no_grad()`)

### 7.3 并发优化
- **异步处理**: 数据预处理与模型推理并行
- **多进程**: 可利用多进程加速数据转换
- **流水线**: 推理与后处理流水线化

---

## 8. 使用指南

### 8.1 基本用法
```python
# 执行完整的Enhanced推理流程
python demo4_g2milp_enhanced.py
```

### 8.2 自定义配置
```python
# 修改推理参数
inference_config.eta = 0.15              # 调整遮盖比例
inference_config.num_test_instances = 10  # 增加生成实例数
inference_config.temperature = 1.5        # 调整采样温度
```

### 8.3 调试模式
```python
# 启用详细日志
logging.getLogger().setLevel(logging.DEBUG)

# 保存中间结果
inference_config.save_intermediate_states = True
```

---

## 9. 扩展接口

### 9.1 自定义评估指标
```python
def custom_evaluation_metric(original_data, generated_data):
    # 实现自定义评估逻辑
    return metric_score

# 集成到评估流程
evaluator.add_custom_metric('custom_metric', custom_evaluation_metric)
```

### 9.2 推理后处理
```python
def custom_post_processing(generated_instances):
    # 实现自定义后处理逻辑
    return processed_instances

# 应用后处理
processed_results = custom_post_processing(inference_results['generated_instances'])
```

---

## 10. 故障排除

### 10.1 常见问题
1. **CUDA内存不足**: 减少batch_size或使用CPU
2. **数值不稳定**: 检查输入数据是否包含异常值
3. **推理超时**: 调整推理参数或检查模型复杂度

### 10.2 调试检查清单
- [ ] 输入数据格式正确
- [ ] 模型已正确加载
- [ ] 设备配置一致
- [ ] 配置参数合理
- [ ] 足够的计算资源

---

**文档版本**: v1.0  
**更新日期**: 2025年7月7日  
**适用范围**: Demo 4 Enhanced推理流程  
**技术栈**: PyTorch + PyTorch Geometric + CUDA
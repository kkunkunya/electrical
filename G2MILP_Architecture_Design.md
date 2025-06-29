# G2MILP二分图构建完整架构设计文档

## 1. 架构概览

G2MILP（Graph to Mixed Integer Linear Programming）系统是一个完整的CVXPY优化问题到二分图转换框架，专门设计用于将混合整数线性规划（MILP）问题表示为图神经网络友好的二分图结构。

### 1.1 核心设计理念

- **模块化设计**: 每个组件职责单一，便于维护和扩展
- **数据完整性**: 保持原始MILP问题的所有关键信息
- **性能优化**: 支持大规模问题的高效处理
- **易用性**: 提供简洁的API和丰富的工具函数
- **可扩展性**: 支持新的约束类型和特征提取方法

### 1.2 系统架构图

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   CVXPY问题      │    │  BiasedMILP      │    │   外部MILP      │
│     对象        │    │   Generator      │    │     数据        │
└─────────┬───────┘    └─────────┬────────┘    └─────────┬───────┘
          │                      │                       │
          └──────────────────────┼───────────────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │   CVXPYToMILPExtractor  │
                    │   (MILP标准形式提取)     │
                    └────────────┬────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │ BipartiteGraphBuilder   │
                    │   (二分图构建器)        │
                    └────────────┬────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │   G2MILPConverter       │
                    │   (主转换控制器)        │
                    └────────────┬────────────┘
                                 │
          ┌──────────────────────┼──────────────────────┐
          │                      │                      │
┌─────────▼───────┐   ┌─────────▼────────┐   ┌─────────▼────────┐
│ Serializer      │   │ Visualizer       │   │ Validator        │
│ (序列化工具)     │   │ (可视化工具)      │   │ (验证工具)        │
└─────────────────┘   └──────────────────┘   └──────────────────┘
```

## 2. 核心数据结构设计

### 2.1 变量节点 (VariableNode)

变量节点表示MILP问题中的决策变量，包含9维特征向量：

```python
@dataclass
class VariableNode:
    # 基本标识
    node_id: str                        # 节点唯一标识
    cvxpy_var_name: str                # CVXPY变量名
    original_shape: Tuple[int, ...]     # 原始变量形状
    flat_index: int                     # 扁平化索引
    
    # 9维特征向量
    var_type: VariableType             # 变量类型 (维度1)
    lower_bound: float                 # 下界 (维度2)
    upper_bound: float                 # 上界 (维度3)
    obj_coeff: float                   # 目标函数系数 (维度4)
    has_lower_bound: bool              # 是否有下界 (维度5)
    has_upper_bound: bool              # 是否有上界 (维度6)
    degree: int = 0                    # 节点度数 (维度7)
    constraint_types: Set[ConstraintType] = field(default_factory=set)  # 关联约束类型 (维度8)
    coeff_statistics: Dict[str, float] = field(default_factory=dict)    # 系数统计信息 (维度9)
```

#### 特征向量详细定义

1. **变量类型** (维度1): 数值编码
   - 连续变量: 0.0
   - 二进制变量: 1.0
   - 整数变量: 2.0
   - 半连续变量: 3.0
   - 半整数变量: 4.0

2. **下界** (维度2): 变量的下界值，-∞用-1e20表示

3. **上界** (维度3): 变量的上界值，+∞用1e20表示

4. **目标函数系数** (维度4): 该变量在目标函数中的系数

5. **是否有下界** (维度5): 布尔值，1.0表示有，0.0表示无

6. **是否有上界** (维度6): 布尔值，1.0表示有，0.0表示无

7. **节点度数** (维度7): 该变量参与的约束数量

8. **约束类型编码** (维度8): 位图编码，表示该变量参与的约束类型
   - 线性等式: 1
   - 线性不等式: 2
   - 二次约束: 4
   - SOC约束: 8
   - SDP约束: 16
   - 指数约束: 32
   - 对数约束: 64

9. **系数统计** (维度9): 该变量在所有约束中系数的平均绝对值

### 2.2 约束节点 (ConstraintNode)

约束节点表示MILP问题中的约束条件：

```python
@dataclass
class ConstraintNode:
    # 基本标识
    node_id: str                       # 节点唯一标识
    constraint_name: str               # 约束名称
    constraint_type: ConstraintType    # 约束类型
    
    # 约束属性
    lhs_coefficients: Dict[str, float] = field(default_factory=dict)  # 左侧系数 {变量ID: 系数}
    rhs_value: float = 0.0             # 右侧常数值
    sense: str = "=="                  # 约束方向: "==", "<=", ">="
    
    # 约束特征
    nnz_count: int = 0                 # 非零系数数量
    coefficient_range: Tuple[float, float] = (0.0, 0.0)  # 系数范围
    degree: int = 0                    # 约束度数（关联变量数）
```

### 2.3 边 (BipartiteEdge)

边表示变量和约束之间的关系：

```python
@dataclass 
class BipartiteEdge:
    edge_id: str                       # 边唯一标识
    constraint_node_id: str            # 约束节点ID
    variable_node_id: str              # 变量节点ID
    coefficient: float                 # 约束系数（边权重）
    
    # 边特征
    abs_coefficient: float = field(init=False)  # 系数绝对值
    is_nonzero: bool = field(init=False)        # 是否非零
    normalized_coeff: Optional[float] = None     # 归一化系数
```

### 2.4 完整二分图 (BipartiteGraph)

```python
@dataclass
class BipartiteGraph:
    # 基本标识
    graph_id: str                      # 图唯一标识
    source_problem_id: str             # 源MILP问题ID
    
    # 图数据
    variable_nodes: Dict[str, VariableNode] = field(default_factory=dict)     # 变量节点
    constraint_nodes: Dict[str, ConstraintNode] = field(default_factory=dict) # 约束节点
    edges: Dict[str, BipartiteEdge] = field(default_factory=dict)             # 边
    
    # 邻接信息（用于快速查找）
    variable_to_constraints: Dict[str, Set[str]] = field(default_factory=dict)  # 变量->约束
    constraint_to_variables: Dict[str, Set[str]] = field(default_factory=dict)  # 约束->变量
    
    # 统计信息
    statistics: GraphStatistics = field(default_factory=GraphStatistics)
```

## 3. 核心转换类架构

### 3.1 CVXPYToMILPExtractor

负责从CVXPY问题对象中提取MILP标准形式。

#### 主要功能

- **问题解析**: 解析CVXPY问题的变量、约束和目标函数
- **标准化**: 将问题转换为标准的Ax≤b, Aeq x=beq, l≤x≤u形式
- **类型识别**: 识别变量类型（连续、二进制、整数）
- **约束分类**: 区分等式约束、不等式约束、SOC约束等

#### 核心方法

```python
class CVXPYToMILPExtractor:
    def extract(self, use_sparse: bool = True, tolerance: float = 1e-12) -> MILPStandardForm:
        """执行提取过程"""
        
    def _extract_constraints(self, data: Dict, use_sparse: bool) -> Tuple[...]:
        """提取约束信息"""
        
    def _extract_variables(self, data: Dict, chain) -> Tuple[...]:
        """提取变量信息"""
        
    def _extract_objective(self, data: Dict) -> Tuple[...]:
        """提取目标函数信息"""
```

### 3.2 BipartiteGraphBuilder

负责从MILP标准形式构建二分图。

#### 主要功能

- **节点创建**: 为每个变量和约束创建对应的图节点
- **边构建**: 基于约束矩阵创建变量-约束边
- **特征计算**: 计算节点和边的特征向量
- **图优化**: 执行系数归一化、度数更新等优化操作

#### 核心方法

```python
class BipartiteGraphBuilder:
    def build_graph(self, standard_form: MILPStandardForm, graph_id: str = None) -> BipartiteGraph:
        """构建二分图"""
        
    def _build_variable_nodes(self, graph: BipartiteGraph, standard_form: MILPStandardForm):
        """构建变量节点"""
        
    def _build_constraint_nodes(self, graph: BipartiteGraph, standard_form: MILPStandardForm):
        """构建约束节点"""
        
    def _build_edges(self, graph: BipartiteGraph, standard_form: MILPStandardForm):
        """构建边连接"""
```

### 3.3 G2MILPConverter

主转换控制器，协调整个转换流程。

#### 主要功能

- **流程控制**: 协调提取器和构建器的工作
- **批量处理**: 支持多个MILP实例的批量转换
- **错误处理**: 提供完整的错误处理和恢复机制
- **性能监控**: 记录转换过程的性能指标

#### 核心方法

```python
class G2MILPConverter:
    def convert_problem(self, problem: cp.Problem, problem_name: str = None, graph_id: str = None) -> ConversionResult:
        """转换单个CVXPY问题"""
        
    def convert_batch(self, problems: List[cp.Problem], ...) -> List[ConversionResult]:
        """批量转换CVXPY问题"""
        
    def convert_from_milp_instances(self, milp_instances: List[Any], ...) -> List[ConversionResult]:
        """从MILP实例批量转换"""
```

## 4. 序列化和持久化接口

### 4.1 支持的格式

1. **Pickle格式**: 完整的Python对象序列化，保持所有信息
2. **JSON格式**: 轻量级文本格式，便于跨平台交换
3. **HDF5格式**: 适用于大规模图数据的高效存储
4. **NetworkX格式**: 兼容图分析工具
5. **矩阵格式**: CSV格式的矩阵和向量，便于外部工具使用

### 4.2 序列化器设计

```python
class BipartiteGraphSerializer:
    def save_pickle(self, graph: BipartiteGraph, filepath: str) -> bool:
        """保存为Pickle格式"""
        
    def save_json(self, graph: BipartiteGraph, filepath: str, include_features: bool = True) -> bool:
        """保存为JSON格式"""
        
    def save_hdf5(self, graph: BipartiteGraph, filepath: str) -> bool:
        """保存为HDF5格式"""
        
    def to_networkx(self, graph: BipartiteGraph) -> nx.Graph:
        """转换为NetworkX图"""
        
    def export_matrix_format(self, graph: BipartiteGraph, output_dir: str) -> bool:
        """导出矩阵格式"""
```

## 5. 可视化接口

### 5.1 可视化功能

1. **图布局可视化**: 使用NetworkX布局算法显示图结构
2. **特征热图**: 显示节点特征向量的热图
3. **度数分布**: 分析和可视化节点度数分布
4. **系数分析**: 系数分布和统计分析
5. **交互式图表**: 基于Plotly的交互式可视化

### 5.2 可视化器设计

```python
class BipartiteGraphVisualizer:
    def plot_graph_layout(self, graph: BipartiteGraph, layout: str = 'bipartite') -> plt.Figure:
        """绘制图布局"""
        
    def plot_feature_heatmap(self, graph: BipartiteGraph, feature_type: str = 'variable') -> plt.Figure:
        """绘制特征向量热图"""
        
    def plot_degree_distribution(self, graph: BipartiteGraph) -> plt.Figure:
        """绘制度数分布图"""
        
    def plot_coefficient_analysis(self, graph: BipartiteGraph) -> plt.Figure:
        """绘制系数分布分析图"""
        
    def create_interactive_plot(self, graph: BipartiteGraph) -> plotly.Figure:
        """创建交互式可视化"""
```

## 6. 性能优化和内存管理

### 6.1 性能优化策略

1. **稀疏矩阵支持**: 对于稀疏问题使用scipy.sparse矩阵
2. **批处理**: 支持大批量数据的分批处理
3. **内存映射**: 对大型数据使用内存映射文件
4. **并行处理**: 支持多线程/多进程并行转换
5. **增量构建**: 支持图的增量构建和更新

### 6.2 内存管理

```python
class MemoryManager:
    def __init__(self, memory_limit_gb: float = 8.0):
        self.memory_limit = memory_limit_gb * 1024 * 1024 * 1024  # 转换为字节
        
    def check_memory_usage(self) -> float:
        """检查当前内存使用量"""
        
    def optimize_graph_storage(self, graph: BipartiteGraph) -> BipartiteGraph:
        """优化图的内存存储"""
        
    def clear_cache(self):
        """清理缓存数据"""
```

## 7. 错误处理和验证机制

### 7.1 验证规则

1. **图结构完整性**: 检查节点ID一致性、边的有效性
2. **数据有效性**: 验证特征向量的数学有效性
3. **约束一致性**: 验证约束系数与边权重的一致性
4. **类型兼容性**: 检查变量类型与边界的兼容性

### 7.2 验证器设计

```python
class BipartiteGraphValidator:
    def validate_graph(self, graph: BipartiteGraph) -> ValidationReport:
        """验证二分图"""
        
    def _check_graph_structure(self, graph: BipartiteGraph) -> List[ValidationIssue]:
        """检查图结构完整性"""
        
    def _check_feature_vectors(self, graph: BipartiteGraph) -> List[ValidationIssue]:
        """检查特征向量有效性"""
        
    def _check_data_consistency(self, graph: BipartiteGraph) -> List[ValidationIssue]:
        """检查数据一致性"""
```

## 8. 项目目录结构

```
src/
├── models/
│   ├── bipartite_graph/
│   │   ├── __init__.py
│   │   ├── data_structures.py      # 二分图数据结构
│   │   ├── extractor.py           # CVXPY到MILP提取器
│   │   ├── builder.py             # 二分图构建器
│   │   ├── converter.py           # G2MILP主转换器
│   │   ├── serializer.py          # 序列化工具
│   │   ├── visualizer.py          # 可视化工具
│   │   └── validator.py           # 验证工具
│   └── g2milp_converter.py         # 主入口模块
└── utils/
    └── graph_utils.py              # 图操作工具函数
```

## 9. 使用示例

### 9.1 基本使用

```python
from src.models.g2milp_converter import G2MILPSystem
import cvxpy as cp

# 创建CVXPY问题
x = cp.Variable(3)
constraints = [x >= 0, x[0] + x[1] + x[2] <= 10]
objective = cp.Minimize(cp.sum(x))
problem = cp.Problem(objective, constraints)

# 初始化G2MILP系统
system = G2MILPSystem()

# 转换为二分图
result = system.convert_cvxpy_problem(
    problem=problem,
    problem_name="示例问题",
    save_graph=True,
    generate_visualizations=True
)

if result['success']:
    graph = result['conversion_result'].bipartite_graph
    print(graph.summary())
```

### 9.2 批量转换

```python
from src.models.biased_milp_generator import BiasedMILPGenerator

# 生成MILP实例
generator = BiasedMILPGenerator(system_data)
instances = generator.generate_batch_instances(configs)

# 批量转换
system = G2MILPSystem()
batch_result = system.convert_from_milp_instances(instances)
```

## 10. 扩展和定制

### 10.1 添加新的约束类型

1. 在`ConstraintType`枚举中添加新类型
2. 在`CVXPYToMILPExtractor`中添加解析逻辑
3. 在`BipartiteGraphBuilder`中添加处理逻辑
4. 更新验证规则

### 10.2 添加新的特征

1. 在节点数据结构中添加新字段
2. 在`get_feature_vector`方法中添加特征计算
3. 更新序列化和可视化代码
4. 添加相应的验证规则

## 11. 性能基准

### 11.1 转换性能

- 小型问题 (10变量, 5约束): < 0.1秒
- 中型问题 (100变量, 50约束): < 1秒
- 大型问题 (1000变量, 500约束): < 10秒

### 11.2 内存使用

- 基本开销: ~1KB/节点
- 稀疏图: 线性扩展
- 稠密图: 可能需要优化

## 12. 未来改进方向

1. **GPU加速**: 支持GPU加速的大规模图构建
2. **分布式处理**: 支持分布式图构建和分析
3. **在线学习**: 支持在线更新和增量学习
4. **更多约束类型**: 支持非线性约束的近似表示
5. **自适应特征**: 根据问题类型自动选择特征

## 13. 总结

G2MILP系统提供了一个完整、高效、易用的CVXPY问题到二分图转换解决方案。通过模块化的设计和丰富的功能，它能够满足从研究到生产的各种需求，为图神经网络在优化问题求解中的应用提供了强有力的工具支持。
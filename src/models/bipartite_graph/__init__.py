"""
G2MILP二分图构建模块
用于将CVXPY优化问题转换为二分图表示

主要组件:
- data_structures: 二分图数据结构定义
- extractor: CVXPY到MILP标准形式提取器
- builder: 二分图构建器
- converter: G2MILP主转换器
- serializer: 序列化和持久化工具
- visualizer: 图可视化工具
- validator: 验证和错误检查工具
"""

from .data_structures import (
    VariableNode,
    ConstraintNode, 
    BipartiteEdge,
    BipartiteGraph,
    VariableType,
    ConstraintType,
    GraphStatistics
)

from .extractor import CVXPYToMILPExtractor, MILPStandardForm
from .builder import BipartiteGraphBuilder
from .converter import G2MILPConverter, ConversionConfig
from .serializer import BipartiteGraphSerializer
from .visualizer import BipartiteGraphVisualizer
from .validator import BipartiteGraphValidator

__all__ = [
    # 数据结构
    'VariableNode',
    'ConstraintNode',
    'BipartiteEdge', 
    'BipartiteGraph',
    'VariableType',
    'ConstraintType',
    'GraphStatistics',
    'MILPStandardForm',
    
    # 核心组件
    'CVXPYToMILPExtractor',
    'BipartiteGraphBuilder',
    'G2MILPConverter',
    'ConversionConfig',
    'BipartiteGraphSerializer',
    'BipartiteGraphVisualizer',
    'BipartiteGraphValidator'
]

__version__ = "1.0.0"
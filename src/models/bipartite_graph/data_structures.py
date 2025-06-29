"""
G2MILP二分图数据结构定义
包含变量节点、约束节点、边和完整二分图的数据结构

数据结构特点:
1. 变量节点: 9维特征向量表示
2. 约束节点: 约束类型和系数信息
3. 边: 约束-变量关系的权重和属性
4. 二分图: 完整的图结构和统计信息
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional, Union, Any
from enum import Enum
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class VariableType(Enum):
    """变量类型枚举"""
    CONTINUOUS = "continuous"    # 连续变量
    BINARY = "binary"           # 二进制变量 
    INTEGER = "integer"         # 整数变量
    SEMI_CONTINUOUS = "semi_continuous"  # 半连续变量
    SEMI_INTEGER = "semi_integer"        # 半整数变量


class ConstraintType(Enum):
    """约束类型枚举"""
    LINEAR_EQ = "linear_equality"        # 线性等式约束
    LINEAR_INEQ = "linear_inequality"    # 线性不等式约束
    QUADRATIC = "quadratic"              # 二次约束
    SOC = "second_order_cone"           # 二阶锥约束
    SDP = "semidefinite"                # 半定约束
    EXPONENTIAL = "exponential"          # 指数约束
    LOG = "logarithmic"                 # 对数约束


@dataclass
class VariableNode:
    """
    变量节点数据结构
    包含9维特征向量和相关元信息
    """
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
    
    # 额外元信息
    creation_time: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_feature_vector(self) -> np.ndarray:
        """
        获取9维特征向量
        
        Returns:
            9维numpy数组特征向量
        """
        # 将变量类型编码为数值
        type_encoding = {
            VariableType.CONTINUOUS: 0.0,
            VariableType.BINARY: 1.0,
            VariableType.INTEGER: 2.0,
            VariableType.SEMI_CONTINUOUS: 3.0,
            VariableType.SEMI_INTEGER: 4.0
        }
        
        # 约束类型的数值表示（位图编码）
        constraint_type_bits = 0
        for ct in self.constraint_types:
            if ct == ConstraintType.LINEAR_EQ:
                constraint_type_bits |= 1
            elif ct == ConstraintType.LINEAR_INEQ:
                constraint_type_bits |= 2
            elif ct == ConstraintType.QUADRATIC:
                constraint_type_bits |= 4
            elif ct == ConstraintType.SOC:
                constraint_type_bits |= 8
            elif ct == ConstraintType.SDP:
                constraint_type_bits |= 16
            elif ct == ConstraintType.EXPONENTIAL:
                constraint_type_bits |= 32
            elif ct == ConstraintType.LOG:
                constraint_type_bits |= 64
        
        # 系数统计信息（使用平均绝对值）
        coeff_stat = self.coeff_statistics.get('mean_abs_coeff', 0.0)
        
        feature_vector = np.array([
            type_encoding.get(self.var_type, 0.0),  # 维度1: 变量类型
            self.lower_bound,                        # 维度2: 下界
            self.upper_bound,                        # 维度3: 上界
            self.obj_coeff,                         # 维度4: 目标函数系数
            float(self.has_lower_bound),            # 维度5: 是否有下界
            float(self.has_upper_bound),            # 维度6: 是否有上界
            float(self.degree),                     # 维度7: 节点度数
            float(constraint_type_bits),            # 维度8: 约束类型编码
            coeff_stat                              # 维度9: 系数统计
        ], dtype=np.float64)
        
        return feature_vector
    
    def update_statistics(self, coefficients: List[float]):
        """
        更新变量节点的系数统计信息
        
        Args:
            coefficients: 该变量在各约束中的系数列表
        """
        if not coefficients:
            return
            
        coeffs = np.array(coefficients)
        self.coeff_statistics.update({
            'mean_coeff': float(np.mean(coeffs)),
            'std_coeff': float(np.std(coeffs)),
            'min_coeff': float(np.min(coeffs)),
            'max_coeff': float(np.max(coeffs)),
            'mean_abs_coeff': float(np.mean(np.abs(coeffs))),
            'nnz_count': int(np.count_nonzero(coeffs))
        })
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'node_id': self.node_id,
            'cvxpy_var_name': self.cvxpy_var_name,
            'original_shape': self.original_shape,
            'flat_index': self.flat_index,
            'var_type': self.var_type.value,
            'lower_bound': self.lower_bound,
            'upper_bound': self.upper_bound,
            'obj_coeff': self.obj_coeff,
            'has_lower_bound': self.has_lower_bound,
            'has_upper_bound': self.has_upper_bound,
            'degree': self.degree,
            'constraint_types': [ct.value for ct in self.constraint_types],
            'coeff_statistics': self.coeff_statistics,
            'feature_vector': self.get_feature_vector().tolist(),
            'creation_time': self.creation_time.isoformat(),
            'metadata': self.metadata
        }


@dataclass
class ConstraintNode:
    """
    约束节点数据结构
    表示MILP中的约束
    """
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
    
    # 额外信息
    is_binding: Optional[bool] = None   # 是否为紧约束（求解后确定）
    slack_value: Optional[float] = None # 松弛变量值
    dual_value: Optional[float] = None  # 对偶变量值
    
    # 元信息
    creation_time: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_constraint_features(self) -> np.ndarray:
        """
        获取约束节点特征向量
        
        Returns:
            约束特征向量
        """
        # 约束类型编码
        type_encoding = {
            ConstraintType.LINEAR_EQ: 0.0,
            ConstraintType.LINEAR_INEQ: 1.0,
            ConstraintType.QUADRATIC: 2.0,
            ConstraintType.SOC: 3.0,
            ConstraintType.SDP: 4.0,
            ConstraintType.EXPONENTIAL: 5.0,
            ConstraintType.LOG: 6.0
        }
        
        # 约束方向编码
        sense_encoding = {"==": 0.0, "<=": 1.0, ">=": 2.0}
        
        # 系数统计
        if self.lhs_coefficients:
            coeffs = np.array(list(self.lhs_coefficients.values()))
            mean_abs_coeff = np.mean(np.abs(coeffs))
            max_abs_coeff = np.max(np.abs(coeffs))
            coeff_std = np.std(coeffs)
        else:
            mean_abs_coeff = max_abs_coeff = coeff_std = 0.0
        
        feature_vector = np.array([
            type_encoding.get(self.constraint_type, 0.0),  # 约束类型
            sense_encoding.get(self.sense, 0.0),           # 约束方向
            float(self.degree),                            # 约束度数
            float(self.nnz_count),                         # 非零系数数量
            abs(self.rhs_value),                           # 右侧值绝对值
            mean_abs_coeff,                                # 平均绝对系数
            max_abs_coeff,                                 # 最大绝对系数
            coeff_std,                                     # 系数标准差
            float(self.is_binding) if self.is_binding is not None else 0.0  # 是否紧约束
        ], dtype=np.float64)
        
        return feature_vector
    
    def update_coefficient_statistics(self):
        """更新系数统计信息"""
        if not self.lhs_coefficients:
            self.coefficient_range = (0.0, 0.0)
            self.nnz_count = 0
            return
            
        coeffs = list(self.lhs_coefficients.values())
        self.nnz_count = len([c for c in coeffs if abs(c) > 1e-12])
        self.coefficient_range = (min(coeffs), max(coeffs))
        self.degree = len(self.lhs_coefficients)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'node_id': self.node_id,
            'constraint_name': self.constraint_name,
            'constraint_type': self.constraint_type.value,
            'lhs_coefficients': self.lhs_coefficients,
            'rhs_value': self.rhs_value,
            'sense': self.sense,
            'nnz_count': self.nnz_count,
            'coefficient_range': self.coefficient_range,
            'degree': self.degree,
            'is_binding': self.is_binding,
            'slack_value': self.slack_value,
            'dual_value': self.dual_value,
            'feature_vector': self.get_constraint_features().tolist(),
            'creation_time': self.creation_time.isoformat(),
            'metadata': self.metadata
        }


@dataclass 
class BipartiteEdge:
    """
    二分图边数据结构
    表示约束节点和变量节点之间的连接
    """
    edge_id: str                       # 边唯一标识
    constraint_node_id: str            # 约束节点ID
    variable_node_id: str              # 变量节点ID
    coefficient: float                 # 约束系数（边权重）
    
    # 边特征
    abs_coefficient: float = field(init=False)  # 系数绝对值
    is_nonzero: bool = field(init=False)        # 是否非零
    normalized_coeff: Optional[float] = None     # 归一化系数
    
    # 元信息
    creation_time: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """后初始化处理"""
        self.abs_coefficient = abs(self.coefficient)
        self.is_nonzero = abs(self.coefficient) > 1e-12
    
    def get_edge_features(self) -> np.ndarray:
        """
        获取边特征向量
        
        Returns:
            边特征向量
        """
        feature_vector = np.array([
            self.coefficient,                    # 原始系数
            self.abs_coefficient,               # 系数绝对值
            float(self.is_nonzero),             # 是否非零
            self.normalized_coeff or 0.0,       # 归一化系数
            np.sign(self.coefficient)           # 系数符号
        ], dtype=np.float64)
        
        return feature_vector
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'edge_id': self.edge_id,
            'constraint_node_id': self.constraint_node_id,
            'variable_node_id': self.variable_node_id,
            'coefficient': self.coefficient,
            'abs_coefficient': self.abs_coefficient,
            'is_nonzero': self.is_nonzero,
            'normalized_coeff': self.normalized_coeff,
            'feature_vector': self.get_edge_features().tolist(),
            'creation_time': self.creation_time.isoformat(),
            'metadata': self.metadata
        }


@dataclass
class GraphStatistics:
    """二分图统计信息"""
    # 基本统计
    n_variable_nodes: int = 0
    n_constraint_nodes: int = 0
    n_edges: int = 0
    n_nonzero_edges: int = 0
    
    # 变量统计
    n_continuous_vars: int = 0
    n_binary_vars: int = 0
    n_integer_vars: int = 0
    
    # 约束统计
    n_equality_constraints: int = 0
    n_inequality_constraints: int = 0
    n_soc_constraints: int = 0
    n_quadratic_constraints: int = 0
    
    # 图结构统计
    density: float = 0.0                # 图密度
    avg_variable_degree: float = 0.0    # 平均变量度数
    avg_constraint_degree: float = 0.0  # 平均约束度数
    max_variable_degree: int = 0        # 最大变量度数
    max_constraint_degree: int = 0      # 最大约束度数
    
    # 系数统计
    coefficient_stats: Dict[str, float] = field(default_factory=dict)
    
    # 时间信息
    creation_time: datetime = field(default_factory=datetime.now)
    build_duration: Optional[float] = None  # 构建耗时（秒）
    
    def update_density(self):
        """更新图密度"""
        if self.n_variable_nodes > 0 and self.n_constraint_nodes > 0:
            max_edges = self.n_variable_nodes * self.n_constraint_nodes
            self.density = self.n_edges / max_edges
        else:
            self.density = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'n_variable_nodes': self.n_variable_nodes,
            'n_constraint_nodes': self.n_constraint_nodes,
            'n_edges': self.n_edges,
            'n_nonzero_edges': self.n_nonzero_edges,
            'n_continuous_vars': self.n_continuous_vars,
            'n_binary_vars': self.n_binary_vars,
            'n_integer_vars': self.n_integer_vars,
            'n_equality_constraints': self.n_equality_constraints,
            'n_inequality_constraints': self.n_inequality_constraints,
            'n_soc_constraints': self.n_soc_constraints,
            'n_quadratic_constraints': self.n_quadratic_constraints,
            'density': self.density,
            'avg_variable_degree': self.avg_variable_degree,
            'avg_constraint_degree': self.avg_constraint_degree,
            'max_variable_degree': self.max_variable_degree,
            'max_constraint_degree': self.max_constraint_degree,
            'coefficient_stats': self.coefficient_stats,
            'creation_time': self.creation_time.isoformat(),
            'build_duration': self.build_duration
        }


@dataclass
class BipartiteGraph:
    """
    完整的MILP二分图数据结构
    包含所有节点、边和统计信息
    """
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
    
    # 元信息
    creation_time: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_variable_node(self, node: VariableNode):
        """添加变量节点"""
        self.variable_nodes[node.node_id] = node
        if node.node_id not in self.variable_to_constraints:
            self.variable_to_constraints[node.node_id] = set()
    
    def add_constraint_node(self, node: ConstraintNode):
        """添加约束节点"""
        self.constraint_nodes[node.node_id] = node
        if node.node_id not in self.constraint_to_variables:
            self.constraint_to_variables[node.node_id] = set()
    
    def add_edge(self, edge: BipartiteEdge):
        """添加边"""
        self.edges[edge.edge_id] = edge
        
        # 更新邻接信息
        if edge.variable_node_id not in self.variable_to_constraints:
            self.variable_to_constraints[edge.variable_node_id] = set()
        if edge.constraint_node_id not in self.constraint_to_variables:
            self.constraint_to_variables[edge.constraint_node_id] = set()
            
        self.variable_to_constraints[edge.variable_node_id].add(edge.constraint_node_id)
        self.constraint_to_variables[edge.constraint_node_id].add(edge.variable_node_id)
    
    def get_variable_neighbors(self, variable_node_id: str) -> Set[str]:
        """获取变量节点的邻居约束节点"""
        return self.variable_to_constraints.get(variable_node_id, set())
    
    def get_constraint_neighbors(self, constraint_node_id: str) -> Set[str]:
        """获取约束节点的邻居变量节点"""
        return self.constraint_to_variables.get(constraint_node_id, set())
    
    def update_node_degrees(self):
        """更新所有节点的度数"""
        # 更新变量节点度数
        for var_id, var_node in self.variable_nodes.items():
            var_node.degree = len(self.variable_to_constraints.get(var_id, set()))
        
        # 更新约束节点度数
        for cons_id, cons_node in self.constraint_nodes.items():
            cons_node.degree = len(self.constraint_to_variables.get(cons_id, set()))
    
    def compute_statistics(self):
        """计算图统计信息"""
        stats = GraphStatistics()
        
        # 基本计数
        stats.n_variable_nodes = len(self.variable_nodes)
        stats.n_constraint_nodes = len(self.constraint_nodes)
        stats.n_edges = len(self.edges)
        stats.n_nonzero_edges = sum(1 for edge in self.edges.values() if edge.is_nonzero)
        
        # 变量类型统计
        for var_node in self.variable_nodes.values():
            if var_node.var_type == VariableType.CONTINUOUS:
                stats.n_continuous_vars += 1
            elif var_node.var_type == VariableType.BINARY:
                stats.n_binary_vars += 1
            elif var_node.var_type == VariableType.INTEGER:
                stats.n_integer_vars += 1
        
        # 约束类型统计
        for cons_node in self.constraint_nodes.values():
            if cons_node.constraint_type == ConstraintType.LINEAR_EQ:
                stats.n_equality_constraints += 1
            elif cons_node.constraint_type == ConstraintType.LINEAR_INEQ:
                stats.n_inequality_constraints += 1
            elif cons_node.constraint_type == ConstraintType.SOC:
                stats.n_soc_constraints += 1
            elif cons_node.constraint_type == ConstraintType.QUADRATIC:
                stats.n_quadratic_constraints += 1
        
        # 度数统计
        if self.variable_nodes:
            var_degrees = [node.degree for node in self.variable_nodes.values()]
            stats.avg_variable_degree = np.mean(var_degrees)
            stats.max_variable_degree = max(var_degrees)
        
        if self.constraint_nodes:
            cons_degrees = [node.degree for node in self.constraint_nodes.values()]
            stats.avg_constraint_degree = np.mean(cons_degrees)
            stats.max_constraint_degree = max(cons_degrees)
        
        # 系数统计
        if self.edges:
            coeffs = [edge.coefficient for edge in self.edges.values()]
            stats.coefficient_stats = {
                'mean': float(np.mean(coeffs)),
                'std': float(np.std(coeffs)),
                'min': float(np.min(coeffs)),
                'max': float(np.max(coeffs)),
                'mean_abs': float(np.mean(np.abs(coeffs))),
                'nnz_ratio': float(stats.n_nonzero_edges / stats.n_edges)
            }
        
        # 更新密度
        stats.update_density()
        
        self.statistics = stats
        return stats
    
    def get_adjacency_matrix(self, sparse: bool = True) -> Union[np.ndarray, 'sp.csr_matrix']:
        """
        获取邻接矩阵
        
        Args:
            sparse: 是否返回稀疏矩阵
            
        Returns:
            邻接矩阵 (约束节点 x 变量节点)
        """
        try:
            import scipy.sparse as sp
        except ImportError:
            logger.warning("scipy不可用，将返回稠密矩阵")
            sparse = False
        
        n_constraints = len(self.constraint_nodes)
        n_variables = len(self.variable_nodes)
        
        # 创建节点ID到索引的映射
        constraint_id_to_idx = {cid: i for i, cid in enumerate(self.constraint_nodes.keys())}
        variable_id_to_idx = {vid: i for i, vid in enumerate(self.variable_nodes.keys())}
        
        if sparse:
            # 创建稀疏矩阵
            row_indices = []
            col_indices = []
            data = []
            
            for edge in self.edges.values():
                row_idx = constraint_id_to_idx[edge.constraint_node_id]
                col_idx = variable_id_to_idx[edge.variable_node_id]
                row_indices.append(row_idx)
                col_indices.append(col_idx)
                data.append(edge.coefficient)
            
            adjacency_matrix = sp.csr_matrix(
                (data, (row_indices, col_indices)),
                shape=(n_constraints, n_variables)
            )
        else:
            # 创建稠密矩阵
            adjacency_matrix = np.zeros((n_constraints, n_variables))
            
            for edge in self.edges.values():
                row_idx = constraint_id_to_idx[edge.constraint_node_id]
                col_idx = variable_id_to_idx[edge.variable_node_id]
                adjacency_matrix[row_idx, col_idx] = edge.coefficient
        
        return adjacency_matrix
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'graph_id': self.graph_id,
            'source_problem_id': self.source_problem_id,
            'variable_nodes': {vid: node.to_dict() for vid, node in self.variable_nodes.items()},
            'constraint_nodes': {cid: node.to_dict() for cid, node in self.constraint_nodes.items()},
            'edges': {eid: edge.to_dict() for eid, edge in self.edges.items()},
            'variable_to_constraints': {vid: list(cids) for vid, cids in self.variable_to_constraints.items()},
            'constraint_to_variables': {cid: list(vids) for cid, vids in self.constraint_to_variables.items()},
            'statistics': self.statistics.to_dict(),
            'creation_time': self.creation_time.isoformat(),
            'metadata': self.metadata
        }
    
    def summary(self) -> str:
        """返回图的摘要信息"""
        return f"""
G2MILP二分图摘要 - {self.graph_id}
===============================================
源问题: {self.source_problem_id}
变量节点: {len(self.variable_nodes)} (连续: {self.statistics.n_continuous_vars}, 二进制: {self.statistics.n_binary_vars})
约束节点: {len(self.constraint_nodes)} (等式: {self.statistics.n_equality_constraints}, 不等式: {self.statistics.n_inequality_constraints})
边数量: {len(self.edges)} (非零: {self.statistics.n_nonzero_edges})
图密度: {self.statistics.density:.4f}
平均度数: 变量={self.statistics.avg_variable_degree:.2f}, 约束={self.statistics.avg_constraint_degree:.2f}
创建时间: {self.creation_time.strftime('%Y-%m-%d %H:%M:%S')}
===============================================
        """
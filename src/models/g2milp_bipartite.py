"""
G2MILP二分图数据表示框架
G2MILP Bipartite Graph Data Representation Framework

本模块实现了完整的G2MILP框架中的二分图数据表示方法，包括：
1. MILP实例到二分图的转换
2. 约束节点和变量节点的特征提取
3. 变量节点的9维特征向量实现
4. 边特征计算和连接条件
5. 与现有MILP生成器的集成接口
6. 支持多种图神经网络框架的数据格式转换

技术特性：
- 基于CVXPY优化问题的标准MILP形式参数提取
- 完整的约束节点特征定义（电力系统语义增强）
- 标准的9维变量节点特征向量
- 丰富的边特征和拓扑信息
- 支持PyTorch Geometric和DGL格式转换
- 批量处理和性能优化
"""

import cvxpy as cp
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime
import json
import pickle
from scipy.sparse import csr_matrix, issparse
import warnings

# 设置日志
logger = logging.getLogger(__name__)


@dataclass
class ConstraintNodeFeatures:
    """
    约束节点特征定义
    
    根据G2MILP框架设计，约束节点包含丰富的约束信息和电力系统语义
    """
    # 基础约束信息
    constraint_type: int        # 约束类型：0=等式，1=不等式(≤)，2=不等式(≥)，3=边界约束
    rhs_value: float           # 右端项值 (b_i 或 d_i)
    constraint_sense: int      # 约束方向：0=≤, 1=≥, 2==
    
    # 约束规模信息
    n_nonzeros: int           # 非零系数数量
    row_density: float        # 行密度 (非零元素/总变量数)
    
    # 统计特征
    coeff_sum: float          # 系数和
    coeff_mean: float         # 系数均值
    coeff_std: float          # 系数标准差
    coeff_max: float          # 最大系数绝对值
    coeff_min: float          # 最小非零系数绝对值
    coeff_range: float        # 系数范围（最大值-最小值）
    
    # 拓扑特征
    constraint_degree: int    # 约束度数（连接的变量数）
    
    # 归一化特征
    rhs_normalized: float     # 归一化的右端项值
    coeff_sum_normalized: float  # 归一化的系数和
    
    # 电力系统专用特征
    bus_id: int              # 关联的节点ID (-1表示非节点约束)
    time_period: int         # 时间段 (-1表示非时变约束)
    constraint_category: str  # 约束类别："power_balance", "voltage", "flow", "capacity", "energy", "other"
    
    def to_feature_vector(self) -> np.ndarray:
        """转换为特征向量"""
        return np.array([
            self.constraint_type,
            self.rhs_value,
            self.constraint_sense,
            self.n_nonzeros,
            self.row_density,
            self.coeff_sum,
            self.coeff_mean,
            self.coeff_std,
            self.coeff_max,
            self.coeff_min,
            self.coeff_range,
            self.constraint_degree,
            self.rhs_normalized,
            self.coeff_sum_normalized,
            self.bus_id,
            self.time_period
        ], dtype=np.float32)


@dataclass
class VariableNodeFeatures:
    """
    变量节点的9维特征向量
    
    根据G2MILP论文的标准设计，实现变量节点的9维特征表示
    """
    # G2MILP标准9维特征向量
    var_type: float           # 第1维：变量类型 (0.0=连续, 1.0=二进制, 0.5=整数)
    obj_coeff: float          # 第2维：目标函数系数 c_j
    lower_bound: float        # 第3维：变量下界 l_j
    upper_bound: float        # 第4维：变量上界 u_j
    variable_degree: float    # 第5维：变量度数（出现在多少个约束中）
    coeff_mean: float         # 第6维：该变量在所有约束中系数的均值
    coeff_std: float          # 第7维：该变量在所有约束中系数的标准差
    coeff_max: float          # 第8维：该变量系数的最大绝对值
    var_index_norm: float     # 第9维：归一化的变量索引 (j/n)
    
    # 扩展特征（不计入9维标准特征）
    coeff_min: float = 0.0           # 最小系数绝对值
    coeff_sum: float = 0.0           # 系数总和
    bound_range: float = 0.0         # 上下界范围
    var_name: str = ""               # 变量名称
    var_semantic: str = "unknown"    # 变量语义类别
    
    def to_standard_9d_vector(self) -> np.ndarray:
        """返回标准的9维特征向量"""
        return np.array([
            self.var_type,
            self.obj_coeff,
            self.lower_bound,
            self.upper_bound,
            self.variable_degree,
            self.coeff_mean,
            self.coeff_std,
            self.coeff_max,
            self.var_index_norm
        ], dtype=np.float32)
    
    def to_extended_vector(self) -> np.ndarray:
        """返回扩展特征向量"""
        standard = self.to_standard_9d_vector()
        extended = np.array([
            self.coeff_min,
            self.coeff_sum,
            self.bound_range
        ], dtype=np.float32)
        return np.concatenate([standard, extended])


@dataclass  
class EdgeFeatures:
    """
    边特征定义
    
    表示约束节点和变量节点之间的连接关系及其特征
    """
    # 基础系数信息
    coefficient: float        # 约束矩阵中的系数 A_ij
    abs_coefficient: float    # 系数绝对值
    log_abs_coeff: float      # 系数绝对值的对数（处理大数值范围）
    
    # 归一化特征
    coeff_normalized_row: float     # 按行归一化的系数
    coeff_normalized_col: float     # 按列归一化的系数
    coeff_normalized_global: float  # 全局归一化的系数
    
    # 相对重要性
    coeff_rank_in_row: float       # 在该约束中的系数排名 (0-1)
    coeff_rank_in_col: float       # 在该变量中的系数排名 (0-1)
    
    def to_feature_vector(self) -> np.ndarray:
        """转换为边特征向量"""
        return np.array([
            self.coefficient,
            self.abs_coefficient,
            self.log_abs_coeff,
            self.coeff_normalized_row,
            self.coeff_normalized_col,
            self.coeff_normalized_global,
            self.coeff_rank_in_row,
            self.coeff_rank_in_col
        ], dtype=np.float32)


@dataclass
class BipartiteGraphRepresentation:
    """
    G2MILP二分图表示
    
    完整的二分图数据结构，包含节点、边、特征和元信息
    """
    # 图结构基本信息
    n_constraint_nodes: int
    n_variable_nodes: int  
    n_edges: int
    edges: List[Tuple[int, int]]  # (约束节点ID, 变量节点ID+偏移)
    
    # 节点特征
    constraint_features: List[ConstraintNodeFeatures]
    variable_features: List[VariableNodeFeatures]
    edge_features: List[EdgeFeatures]
    
    # 特征矩阵（便于批处理）
    constraint_feature_matrix: np.ndarray    # (n_constraints, n_constraint_features)
    variable_feature_matrix: np.ndarray      # (n_variables, 9) 标准9维特征
    edge_feature_matrix: np.ndarray          # (n_edges, n_edge_features)
    
    # 原始MILP数据
    constraint_matrix: csr_matrix    # A矩阵（稀疏格式）
    objective_coeffs: np.ndarray     # c向量
    rhs_values: np.ndarray          # b向量
    variable_bounds: np.ndarray      # (n_variables, 2) [lower, upper]
    variable_types: List[str]        # ['continuous', 'binary', 'integer']
    constraint_senses: List[str]     # ['<=', '>=', '=']
    
    # 统计信息
    graph_statistics: Dict[str, Any]
    
    # 元信息
    milp_instance_id: str
    generation_timestamp: str
    perturbation_applied: bool
    source_problem_name: str = ""
    
    def get_node_degrees(self) -> Tuple[np.ndarray, np.ndarray]:
        """获取节点度数"""
        constraint_degrees = np.zeros(self.n_constraint_nodes)
        variable_degrees = np.zeros(self.n_variable_nodes)
        
        for c_id, v_id in self.edges:
            constraint_degrees[c_id] += 1
            variable_degrees[v_id - self.n_constraint_nodes] += 1
            
        return constraint_degrees, variable_degrees
    
    def get_adjacency_matrix(self) -> csr_matrix:
        """获取邻接矩阵"""
        rows, cols = zip(*self.edges) if self.edges else ([], [])
        data = [1] * len(self.edges)
        total_nodes = self.n_constraint_nodes + self.n_variable_nodes
        
        return csr_matrix((data, (rows, cols)), shape=(total_nodes, total_nodes))
    
    def to_pytorch_geometric(self):
        """转换为PyTorch Geometric异构图格式"""
        try:
            import torch
            from torch_geometric.data import HeteroData
        except ImportError:
            raise ImportError("需要安装 torch_geometric: pip install torch-geometric")
        
        data = HeteroData()
        
        # 数据归一化（关键修复）
        constraint_features_normalized = self._normalize_features(
            self.constraint_feature_matrix, "constraint"
        )
        variable_features_normalized = self._normalize_features(
            self.variable_feature_matrix, "variable"
        )
        
        # 约束节点（使用归一化特征）
        data['constraint'].x = torch.tensor(constraint_features_normalized, dtype=torch.float)
        data['constraint'].num_nodes = self.n_constraint_nodes
        
        # 变量节点（使用归一化的标准9维特征）
        data['variable'].x = torch.tensor(variable_features_normalized, dtype=torch.float)  
        data['variable'].num_nodes = self.n_variable_nodes
        
        # 边（约束-变量连接）
        if self.edges:
            # 检查边的格式和有效性
            logger.info(f"原始边数量: {len(self.edges)}, 边特征矩阵形状: {self.edge_feature_matrix.shape}")
            
            # 修复边索引：假设边格式为 (constraint_id, variable_id)
            # 约束ID应该在 [0, n_constraint_nodes)，变量ID应该在 [0, n_variable_nodes)
            valid_edges = []
            for i, edge in enumerate(self.edges):
                if len(edge) >= 2:
                    c, v = edge[0], edge[1]
                    # 确保索引在有效范围内
                    if 0 <= c < self.n_constraint_nodes and 0 <= v < self.n_variable_nodes:
                        valid_edges.append((c, v))
                    elif c < self.n_constraint_nodes:
                        # 如果变量索引需要调整（从全局索引转换为局部索引）
                        v_adjusted = v - self.n_constraint_nodes if v >= self.n_constraint_nodes else v
                        if 0 <= v_adjusted < self.n_variable_nodes:
                            valid_edges.append((c, v_adjusted))
            
            logger.info(f"有效边数量: {len(valid_edges)}")
            
            if valid_edges:
                edge_index = torch.tensor(valid_edges, dtype=torch.long).t().contiguous()
                
                data['constraint', 'connects', 'variable'].edge_index = edge_index
                # 只使用与有效边对应的边特征，并进行归一化
                edge_attr = self.edge_feature_matrix[:len(valid_edges)] if len(self.edge_feature_matrix) >= len(valid_edges) else self.edge_feature_matrix
                edge_attr_normalized = self._normalize_features(edge_attr, "edge")
                data['constraint', 'connects', 'variable'].edge_attr = torch.tensor(
                    edge_attr_normalized, dtype=torch.float
                )
                
                # 反向边
                reverse_edge_index = torch.tensor(
                    [(v, c) for c, v in valid_edges], 
                    dtype=torch.long
                ).t().contiguous()
                data['variable', 'connected_by', 'constraint'].edge_index = reverse_edge_index
                data['variable', 'connected_by', 'constraint'].edge_attr = torch.tensor(
                    edge_attr_normalized, dtype=torch.float
                )
            else:
                # 创建空边
                logger.warning("没有有效的边，创建空连接")
                empty_edge_index = torch.empty((2, 0), dtype=torch.long)
                empty_edge_attr = torch.empty((0, self.edge_feature_matrix.shape[1] if self.edge_feature_matrix.size > 0 else 8), dtype=torch.float)
                
                data['constraint', 'connects', 'variable'].edge_index = empty_edge_index
                data['constraint', 'connects', 'variable'].edge_attr = empty_edge_attr
                data['variable', 'connected_by', 'constraint'].edge_index = empty_edge_index
                data['variable', 'connected_by', 'constraint'].edge_attr = empty_edge_attr
        
        # 元信息
        data.metadata = {
            'milp_instance_id': self.milp_instance_id,
            'n_constraint_nodes': self.n_constraint_nodes,
            'n_variable_nodes': self.n_variable_nodes,
            'n_edges': self.n_edges,
            'graph_statistics': self.graph_statistics
        }
        
        return data
    
    def to_dgl_graph(self):
        """转换为DGL异构图格式"""
        try:
            import dgl
            import torch
        except ImportError:
            raise ImportError("需要安装 dgl: pip install dgl")
        
        if not self.edges:
            # 创建空图
            graph_data = {
                ('constraint', 'connects', 'variable'): ([], []),
                ('variable', 'connected_by', 'constraint'): ([], [])
            }
        else:
            # 调整边索引
            constraints, variables = zip(*self.edges)
            variables = [v - self.n_constraint_nodes for v in variables]
            
            graph_data = {
                ('constraint', 'connects', 'variable'): (constraints, variables),
                ('variable', 'connected_by', 'constraint'): (variables, constraints)
            }
        
        g = dgl.heterograph(graph_data)
        
        # 设置节点数量
        g.num_nodes_dict = {
            'constraint': self.n_constraint_nodes,
            'variable': self.n_variable_nodes
        }
        
        # 添加节点特征
        if self.n_constraint_nodes > 0:
            g.nodes['constraint'].data['features'] = torch.tensor(
                self.constraint_feature_matrix, dtype=torch.float
            )
        if self.n_variable_nodes > 0:
            g.nodes['variable'].data['features'] = torch.tensor(
                self.variable_feature_matrix, dtype=torch.float
            )
        
        # 添加边特征
        if self.edges:
            g.edges['connects'].data['features'] = torch.tensor(
                self.edge_feature_matrix, dtype=torch.float
            )
            g.edges['connected_by'].data['features'] = torch.tensor(
                self.edge_feature_matrix, dtype=torch.float
            )
        
        return g
    
    def save_to_file(self, filepath: str) -> bool:
        """保存到文件"""
        try:
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            # 使用pickle保存完整对象
            with open(filepath, 'wb') as f:
                pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # 保存元信息为JSON
            metadata_file = filepath.with_suffix('.json')
            metadata = {
                'milp_instance_id': self.milp_instance_id,
                'generation_timestamp': self.generation_timestamp,
                'n_constraint_nodes': self.n_constraint_nodes,
                'n_variable_nodes': self.n_variable_nodes,
                'n_edges': self.n_edges,
                'perturbation_applied': self.perturbation_applied,
                'source_problem_name': self.source_problem_name,
                'graph_statistics': self.graph_statistics
            }
            
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"二分图表示已保存: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"保存二分图表示失败: {e}")
            return False
    
    @classmethod
    def load_from_file(cls, filepath: str) -> Optional['BipartiteGraphRepresentation']:
        """从文件加载"""
        try:
            with open(filepath, 'rb') as f:
                instance = pickle.load(f)
            logger.info(f"二分图表示已加载: {filepath}")
            return instance
        except Exception as e:
            logger.error(f"加载二分图表示失败: {e}")
            return None
    
    def _normalize_features(self, features: np.ndarray, feature_type: str) -> np.ndarray:
        """
        特征归一化方法（Z-score标准化）
        
        Args:
            features: 特征矩阵 (n_samples, n_features)
            feature_type: 特征类型（用于日志）
            
        Returns:
            归一化后的特征矩阵
        """
        if features.size == 0:
            return features
        
        # 处理无限值和NaN
        features = np.nan_to_num(features, nan=0.0, posinf=1000.0, neginf=-1000.0)
        
        try:
            # Z-score标准化：(x - μ) / σ
            # 按列（特征维度）进行标准化
            means = np.mean(features, axis=0, keepdims=True)
            stds = np.std(features, axis=0, keepdims=True)
            
            # 避免除零：如果标准差为0，设为1（该特征列是常数）
            stds = np.where(stds == 0, 1.0, stds)
            
            normalized_features = (features - means) / stds
            
            # 再次处理可能的无限值
            normalized_features = np.nan_to_num(
                normalized_features, nan=0.0, posinf=3.0, neginf=-3.0
            )
            
            # 检查归一化效果
            if features.shape[0] > 1:  # 只有多个样本时才检查
                final_means = np.mean(normalized_features, axis=0)
                final_stds = np.std(normalized_features, axis=0)
                logger.debug(f"{feature_type}特征归一化 - 均值范围: [{np.min(final_means):.4f}, {np.max(final_means):.4f}], "
                           f"标准差范围: [{np.min(final_stds):.4f}, {np.max(final_stds):.4f}]")
            
            return normalized_features.astype(np.float32)
            
        except Exception as e:
            logger.warning(f"{feature_type}特征归一化失败: {e}，返回原始特征")
            return features.astype(np.float32)


class MILPDataExtractor:
    """
    MILP数据提取器
    
    从CVXPY问题中提取标准MILP形式的参数：c, A, b, l, u
    """
    
    @staticmethod
    def extract_problem_data(cvxpy_problem: cp.Problem) -> Dict[str, Any]:
        """
        提取CVXPY问题的标准形式数据
        
        Args:
            cvxpy_problem: CVXPY问题对象
            
        Returns:
            包含MILP标准形式参数的字典
        """
        try:
            logger.info("开始提取MILP问题数据...")
            
            # 获取问题数据（使用SCS求解器格式）
            data, chain, inverse_data = cvxpy_problem.get_problem_data(solver='SCS')
            
            # 提取基本参数
            c = data.get('c', np.array([]))  # 目标函数系数
            A = data.get('A', csr_matrix((0, 0)))  # 约束矩阵
            b = data.get('b', np.array([]))  # 右端项
            
            # 确保A是稀疏矩阵
            if not issparse(A):
                A = csr_matrix(A)
            
            # 提取变量信息
            variables = cvxpy_problem.variables()
            var_info = MILPDataExtractor._extract_variable_info(variables)
            
            # 提取约束信息
            constraints = cvxpy_problem.constraints
            constraint_info = MILPDataExtractor._extract_constraint_info(constraints, A.shape[0])
            
            result = {
                'c': c,
                'A': A,
                'b': b,
                'variable_info': var_info,
                'constraint_info': constraint_info,
                'cvxpy_data': data,
                'problem_size': {
                    'n_variables': A.shape[1] if A.size > 0 else len(c),
                    'n_constraints': A.shape[0] if A.size > 0 else len(b),
                    'n_nonzeros': A.nnz if A.size > 0 else 0
                }
            }
            
            logger.info(f"数据提取完成 - 变量: {result['problem_size']['n_variables']}, "
                       f"约束: {result['problem_size']['n_constraints']}")
            
            return result
            
        except Exception as e:
            logger.error(f"MILP数据提取失败: {e}")
            raise
    
    @staticmethod
    def _extract_variable_info(variables: List[cp.Variable]) -> Dict[str, Any]:
        """提取变量信息"""
        var_types = []
        var_bounds = []
        var_names = []
        var_shapes = []
        
        for var in variables:
            # 变量类型
            if var.attributes.get('boolean', False):
                var_type = 'binary'
            elif var.attributes.get('integer', False):
                var_type = 'integer'
            else:
                var_type = 'continuous'
            
            # 变量界限（CVXPY内部处理）
            lower_bound = -np.inf
            upper_bound = np.inf
            
            # 对于二进制变量
            if var_type == 'binary':
                lower_bound = 0.0
                upper_bound = 1.0
            
            # 扩展到变量的所有元素
            size = var.size
            var_types.extend([var_type] * size)
            var_bounds.extend([(lower_bound, upper_bound)] * size)
            var_names.extend([f"{var.name()}_{i}" if size > 1 else var.name() for i in range(size)])
            var_shapes.append((var.shape, size))
        
        return {
            'types': var_types,
            'bounds': var_bounds,
            'names': var_names,
            'shapes': var_shapes,
            'total_size': len(var_types)
        }
    
    @staticmethod
    def _extract_constraint_info(constraints: List, n_constraints: int) -> Dict[str, Any]:
        """提取约束信息"""
        constraint_types = ['unknown'] * n_constraints
        constraint_senses = ['<='] * n_constraints
        
        # 尽量从CVXPY约束中推断信息
        try:
            for i, constraint in enumerate(constraints[:n_constraints]):
                if hasattr(constraint, '__class__'):
                    class_name = constraint.__class__.__name__
                    if 'Equality' in class_name or 'Zero' in class_name:
                        constraint_senses[i] = '='
                        constraint_types[i] = 'equality'
                    elif 'Inequality' in class_name or 'NonPos' in class_name:
                        constraint_senses[i] = '<='
                        constraint_types[i] = 'inequality'
                    elif 'NonNeg' in class_name:
                        constraint_senses[i] = '>='
                        constraint_types[i] = 'inequality'
        except Exception as e:
            logger.warning(f"约束信息提取部分失败: {e}")
        
        return {
            'types': constraint_types,
            'senses': constraint_senses
        }


class BipartiteGraphBuilder:
    """
    二分图构建器
    
    将MILP标准形式数据转换为G2MILP二分图表示
    """
    
    def __init__(self, include_power_system_semantics: bool = True):
        """
        初始化二分图构建器
        
        Args:
            include_power_system_semantics: 是否包含电力系统语义特征
        """
        self.include_power_system_semantics = include_power_system_semantics
        
    def build_bipartite_graph(self, 
                             milp_data: Dict[str, Any],
                             instance_id: str = "",
                             system_data: Optional[Any] = None) -> BipartiteGraphRepresentation:
        """
        构建二分图表示
        
        Args:
            milp_data: 从MILPDataExtractor提取的数据
            instance_id: 实例ID
            system_data: 电力系统数据（可选）
            
        Returns:
            完整的二分图表示
        """
        logger.info("开始构建二分图表示...")
        
        # 提取基本数据
        c = milp_data['c']
        A = milp_data['A']
        b = milp_data['b']
        var_info = milp_data['variable_info']
        constraint_info = milp_data['constraint_info']
        
        # 获取真实的问题维度
        if A.size > 0:
            n_constraints, matrix_n_variables = A.shape
        else:
            n_constraints = len(b)
            matrix_n_variables = len(c)
        
        # 使用var_info中的变量数量作为权威数据源
        n_variables = var_info.get('total_size', matrix_n_variables)
        
        logger.info(f"问题维度 - 约束: {n_constraints}, 变量: {n_variables} (矩阵列数: {matrix_n_variables})")
        
        # 1. 构建约束节点特征
        logger.info("构建约束节点特征...")
        constraint_features = self._build_constraint_features(
            A, b, constraint_info, system_data
        )
        
        # 2. 构建变量节点特征  
        logger.info("构建变量节点特征...")
        variable_features = self._build_variable_features(
            A, c, var_info, system_data
        )
        
        # 3. 构建边和边特征
        logger.info("构建边特征...")
        edges, edge_features = self._build_edge_features(A)
        
        # 4. 生成特征矩阵
        constraint_feature_matrix = np.array([
            cf.to_feature_vector() for cf in constraint_features
        ]) if constraint_features else np.empty((0, 16))
        
        variable_feature_matrix = np.array([
            vf.to_standard_9d_vector() for vf in variable_features
        ]) if variable_features else np.empty((0, 9))
        
        edge_feature_matrix = np.array([
            ef.to_feature_vector() for ef in edge_features
        ]) if edge_features else np.empty((0, 8))
        
        # 5. 计算图统计信息
        graph_stats = self._compute_graph_statistics(
            A, constraint_features, variable_features, edges
        )
        
        # 6. 创建二分图表示
        bipartite_graph = BipartiteGraphRepresentation(
            n_constraint_nodes=n_constraints,
            n_variable_nodes=n_variables,
            n_edges=len(edges),
            edges=edges,
            constraint_features=constraint_features,
            variable_features=variable_features,
            edge_features=edge_features,
            constraint_feature_matrix=constraint_feature_matrix,
            variable_feature_matrix=variable_feature_matrix,
            edge_feature_matrix=edge_feature_matrix,
            constraint_matrix=A,
            objective_coeffs=c,
            rhs_values=b,
            variable_bounds=np.array(var_info['bounds']),
            variable_types=var_info['types'],
            constraint_senses=constraint_info['senses'],
            graph_statistics=graph_stats,
            milp_instance_id=instance_id,
            generation_timestamp=datetime.now().isoformat(),
            perturbation_applied=False  # 根据实际情况设置
        )
        
        logger.info(f"二分图构建完成 - 约束节点: {n_constraints}, "
                   f"变量节点: {n_variables}, 边: {len(edges)}")
        
        return bipartite_graph
    
    def _build_constraint_features(self, 
                                  A: csr_matrix, 
                                  b: np.ndarray,
                                  constraint_info: Dict,
                                  system_data: Optional[Any]) -> List[ConstraintNodeFeatures]:
        """构建约束节点特征"""
        n_constraints = A.shape[0] if A.size > 0 else len(b)
        constraint_features = []
        
        # 全局统计用于归一化
        global_rhs_max = np.max(np.abs(b)) if len(b) > 0 else 1.0
        global_coeff_max = np.max(np.abs(A.data)) if A.size > 0 else 1.0
        
        for i in range(n_constraints):
            if A.size > 0:
                # 获取第i行
                row_start = A.indptr[i]
                row_end = A.indptr[i + 1]
                row_data = A.data[row_start:row_end]
                row_indices = A.indices[row_start:row_end]
            else:
                row_data = np.array([])
                row_indices = np.array([])
            
            # 基础信息
            constraint_sense = constraint_info['senses'][i]
            constraint_type = self._encode_constraint_type(constraint_sense)
            rhs_value = b[i] if i < len(b) else 0.0
            
            # 统计特征
            n_nonzeros = len(row_data)
            row_density = n_nonzeros / A.shape[1] if A.size > 0 else 0.0
            
            if n_nonzeros > 0:
                coeff_sum = np.sum(row_data)
                coeff_mean = np.mean(row_data)
                coeff_std = np.std(row_data)
                coeff_max = np.max(np.abs(row_data))
                coeff_min = np.min(np.abs(row_data))
                coeff_range = coeff_max - coeff_min
            else:
                coeff_sum = coeff_mean = coeff_std = 0.0
                coeff_max = coeff_min = coeff_range = 0.0
            
            # 归一化特征
            rhs_normalized = rhs_value / global_rhs_max if global_rhs_max > 0 else 0.0
            coeff_sum_normalized = coeff_sum / global_coeff_max if global_coeff_max > 0 else 0.0
            
            # 电力系统语义（如果可用）
            bus_id, time_period, category = self._extract_power_system_semantics(
                i, system_data
            ) if self.include_power_system_semantics else (-1, -1, "other")
            
            constraint_feature = ConstraintNodeFeatures(
                constraint_type=constraint_type,
                rhs_value=rhs_value,
                constraint_sense=self._encode_constraint_sense(constraint_sense),
                n_nonzeros=n_nonzeros,
                row_density=row_density,
                coeff_sum=coeff_sum,
                coeff_mean=coeff_mean,
                coeff_std=coeff_std,
                coeff_max=coeff_max,
                coeff_min=coeff_min,
                coeff_range=coeff_range,
                constraint_degree=n_nonzeros,
                rhs_normalized=rhs_normalized,
                coeff_sum_normalized=coeff_sum_normalized,
                bus_id=bus_id,
                time_period=time_period,
                constraint_category=category
            )
            
            constraint_features.append(constraint_feature)
        
        return constraint_features
    
    def _build_variable_features(self,
                               A: csr_matrix,
                               c: np.ndarray,
                               var_info: Dict,
                               system_data: Optional[Any]) -> List[VariableNodeFeatures]:
        """构建变量节点的9维特征向量"""
        # 获取真实的变量数量，优先使用var_info中的信息
        if var_info and 'total_size' in var_info:
            n_variables = var_info['total_size']
        elif A.size > 0:
            n_variables = A.shape[1]
        elif len(c) > 0:
            n_variables = len(c)
        else:
            n_variables = 0
            
        variable_features = []
        
        # 全局统计用于归一化
        global_obj_max = np.max(np.abs(c)) if len(c) > 0 else 1.0
        
        logger.info(f"构建变量特征 - 变量数: {n_variables}, 目标系数长度: {len(c)}, var_info长度: {len(var_info.get('types', []))}")
        
        for j in range(n_variables):
            # 变量基础信息 - 安全访问数组
            var_type = var_info['types'][j] if j < len(var_info.get('types', [])) else 'continuous'
            var_bounds = var_info['bounds'][j] if j < len(var_info.get('bounds', [])) else (-np.inf, np.inf)
            var_name = var_info['names'][j] if j < len(var_info.get('names', [])) else f"x_{j}"
            
            # 目标函数系数
            obj_coeff = c[j] if j < len(c) else 0.0
            
            # 变量界限
            lower_bound = var_bounds[0] if not np.isinf(var_bounds[0]) else 0.0
            upper_bound = var_bounds[1] if not np.isinf(var_bounds[1]) else 1.0
            
            if A.size > 0 and j < A.shape[1]:
                # 获取第j列 - 安全访问
                col_data = A[:, j].toarray().flatten()
                nonzero_indices = np.nonzero(col_data)[0]
                col_nonzero_data = col_data[nonzero_indices]
            else:
                col_nonzero_data = np.array([])
                nonzero_indices = np.array([])
            
            # 变量度数
            variable_degree = len(nonzero_indices)
            
            # 系数统计
            if len(col_nonzero_data) > 0:
                coeff_mean = np.mean(col_nonzero_data)
                coeff_std = np.std(col_nonzero_data)
                coeff_max = np.max(np.abs(col_nonzero_data))
                coeff_min = np.min(np.abs(col_nonzero_data))
                coeff_sum = np.sum(col_nonzero_data)
            else:
                coeff_mean = coeff_std = coeff_max = 0.0
                coeff_min = coeff_sum = 0.0
            
            # 归一化索引
            var_index_norm = j / n_variables if n_variables > 0 else 0.0
            
            # 其他特征
            bound_range = upper_bound - lower_bound if not (np.isinf(upper_bound) or np.isinf(lower_bound)) else 0.0
            var_semantic = self._extract_variable_semantic(j, var_name, system_data) if self.include_power_system_semantics else "unknown"
            
            variable_feature = VariableNodeFeatures(
                var_type=self._encode_variable_type(var_type),
                obj_coeff=obj_coeff,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                variable_degree=float(variable_degree),
                coeff_mean=coeff_mean,
                coeff_std=coeff_std,
                coeff_max=coeff_max,
                var_index_norm=var_index_norm,
                coeff_min=coeff_min,
                coeff_sum=coeff_sum,
                bound_range=bound_range,
                var_name=var_name,
                var_semantic=var_semantic
            )
            
            variable_features.append(variable_feature)
        
        return variable_features
    
    def _build_edge_features(self, A: csr_matrix) -> Tuple[List[Tuple[int, int]], List[EdgeFeatures]]:
        """构建边和边特征"""
        edges = []
        edge_features = []
        
        if A.size == 0:
            return edges, edge_features
        
        # 全局统计
        global_coeff_max = np.max(np.abs(A.data))
        global_coeff_sum = np.sum(np.abs(A.data))
        
        # 计算行和列的归一化因子
        row_norms = np.array([np.linalg.norm(A[i, :].toarray()) for i in range(A.shape[0])])
        col_norms = np.array([np.linalg.norm(A[:, j].toarray()) for j in range(A.shape[1])])
        
        # 遍历所有非零元素
        rows, cols = A.nonzero()
        for k in range(len(rows)):
            i, j = rows[k], cols[k]
            coeff = A[i, j]
            
            # 创建边 (约束节点i, 变量节点j+偏移)
            edge = (i, j + A.shape[0])  # 变量节点ID = j + n_constraints
            edges.append(edge)
            
            # 计算边特征
            abs_coeff = abs(coeff)
            log_abs_coeff = np.log(abs_coeff + 1e-10)  # 避免log(0)
            
            # 归一化特征
            coeff_normalized_row = coeff / row_norms[i] if row_norms[i] > 0 else 0.0
            coeff_normalized_col = coeff / col_norms[j] if col_norms[j] > 0 else 0.0
            coeff_normalized_global = coeff / global_coeff_max if global_coeff_max > 0 else 0.0
            
            # 排名特征
            row_data = A[i, :].toarray().flatten()
            col_data = A[:, j].toarray().flatten()
            
            coeff_rank_in_row = self._compute_coefficient_rank(abs_coeff, np.abs(row_data))
            coeff_rank_in_col = self._compute_coefficient_rank(abs_coeff, np.abs(col_data))
            
            edge_feature = EdgeFeatures(
                coefficient=coeff,
                abs_coefficient=abs_coeff,
                log_abs_coeff=log_abs_coeff,
                coeff_normalized_row=coeff_normalized_row,
                coeff_normalized_col=coeff_normalized_col,
                coeff_normalized_global=coeff_normalized_global,
                coeff_rank_in_row=coeff_rank_in_row,
                coeff_rank_in_col=coeff_rank_in_col
            )
            
            edge_features.append(edge_feature)
        
        return edges, edge_features
    
    def _compute_graph_statistics(self,
                                A: csr_matrix,
                                constraint_features: List[ConstraintNodeFeatures],
                                variable_features: List[VariableNodeFeatures],
                                edges: List[Tuple[int, int]]) -> Dict[str, Any]:
        """计算图统计信息"""
        n_constraints = len(constraint_features)
        n_variables = len(variable_features)
        n_edges = len(edges)
        
        # 基本统计
        stats = {
            'n_constraint_nodes': n_constraints,
            'n_variable_nodes': n_variables,
            'n_edges': n_edges,
            'graph_density': (2 * n_edges) / (n_constraints + n_variables) / (n_constraints + n_variables - 1) if (n_constraints + n_variables) > 1 else 0.0,
            'bipartite_density': n_edges / (n_constraints * n_variables) if (n_constraints * n_variables) > 0 else 0.0
        }
        
        # 度数统计
        if edges:
            constraint_degrees = [0] * n_constraints
            variable_degrees = [0] * n_variables
            
            for c_id, v_id in edges:
                constraint_degrees[c_id] += 1
                variable_degrees[v_id - n_constraints] += 1
            
            stats.update({
                'avg_constraint_degree': np.mean(constraint_degrees),
                'max_constraint_degree': np.max(constraint_degrees),
                'min_constraint_degree': np.min(constraint_degrees),
                'avg_variable_degree': np.mean(variable_degrees),
                'max_variable_degree': np.max(variable_degrees),
                'min_variable_degree': np.min(variable_degrees)
            })
        
        # 矩阵统计
        if A.size > 0:
            stats.update({
                'matrix_density': A.nnz / (A.shape[0] * A.shape[1]),
                'avg_nonzeros_per_row': A.nnz / A.shape[0],
                'avg_nonzeros_per_col': A.nnz / A.shape[1],
                'max_coefficient': float(np.max(np.abs(A.data))),
                'min_nonzero_coefficient': float(np.min(np.abs(A.data))),
                'coefficient_range': float(np.max(A.data) - np.min(A.data))
            })
        
        # 变量类型统计
        var_type_counts = {'continuous': 0, 'binary': 0, 'integer': 0}
        for vf in variable_features:
            if vf.var_type == 0.0:
                var_type_counts['continuous'] += 1
            elif vf.var_type == 1.0:
                var_type_counts['binary'] += 1
            else:
                var_type_counts['integer'] += 1
        
        stats['variable_type_distribution'] = var_type_counts
        
        return stats
    
    # 辅助方法
    def _encode_variable_type(self, var_type: str) -> float:
        """编码变量类型"""
        mapping = {
            'continuous': 0.0,
            'binary': 1.0,
            'integer': 0.5
        }
        return mapping.get(var_type, 0.0)
    
    def _encode_constraint_type(self, constraint_sense: str) -> int:
        """编码约束类型"""
        mapping = {
            '=': 0,   # 等式
            '<=': 1,  # 不等式(≤)
            '>=': 2,  # 不等式(≥)
            'bound': 3  # 边界约束
        }
        return mapping.get(constraint_sense, 1)
    
    def _encode_constraint_sense(self, constraint_sense: str) -> int:
        """编码约束方向"""
        mapping = {
            '<=': 0,
            '>=': 1,
            '=': 2
        }
        return mapping.get(constraint_sense, 0)
    
    def _compute_coefficient_rank(self, coeff_abs: float, all_coeffs_abs: np.ndarray) -> float:
        """计算系数在数组中的排名（归一化到0-1）"""
        if len(all_coeffs_abs) <= 1:
            return 0.5
        
        rank = np.sum(all_coeffs_abs <= coeff_abs) - 1
        return rank / (len(all_coeffs_abs) - 1)
    
    def _extract_power_system_semantics(self, constraint_id: int, system_data: Optional[Any]) -> Tuple[int, int, str]:
        """提取电力系统语义信息"""
        # 这里可以根据实际的system_data结构来实现
        # 目前返回默认值
        return -1, -1, "other"
    
    def _extract_variable_semantic(self, var_id: int, var_name: str, system_data: Optional[Any]) -> str:
        """提取变量语义信息"""
        # 根据变量名称推断语义
        var_name_lower = var_name.lower()
        
        if any(keyword in var_name_lower for keyword in ['power', 'p_', 'pg', 'pl']):
            return "power"
        elif any(keyword in var_name_lower for keyword in ['voltage', 'v_', 'vol']):
            return "voltage"
        elif any(keyword in var_name_lower for keyword in ['current', 'i_', 'current']):
            return "current"
        elif any(keyword in var_name_lower for keyword in ['energy', 'e_', 'soc']):
            return "energy"
        elif any(keyword in var_name_lower for keyword in ['binary', 'z_', 'delta', 'switch']):
            return "binary_decision"
        else:
            return "unknown"


# 使用示例和工厂函数
class G2MILPBipartiteGenerator:
    """
    G2MILP二分图生成器主类
    
    集成所有组件，提供统一的接口
    """
    
    def __init__(self, include_power_system_semantics: bool = True):
        """
        初始化生成器
        
        Args:
            include_power_system_semantics: 是否包含电力系统语义增强
        """
        self.data_extractor = MILPDataExtractor()
        self.graph_builder = BipartiteGraphBuilder(include_power_system_semantics)
        
    def generate_from_cvxpy_problem(self,
                                  cvxpy_problem: cp.Problem,
                                  instance_id: str = "",
                                  system_data: Optional[Any] = None) -> BipartiteGraphRepresentation:
        """
        从CVXPY问题生成二分图表示
        
        Args:
            cvxpy_problem: CVXPY问题对象
            instance_id: 实例ID
            system_data: 电力系统数据（可选）
            
        Returns:
            二分图表示对象
        """
        # 1. 提取MILP数据
        milp_data = self.data_extractor.extract_problem_data(cvxpy_problem)
        
        # 2. 构建二分图
        bipartite_graph = self.graph_builder.build_bipartite_graph(
            milp_data, instance_id, system_data
        )
        
        return bipartite_graph
    
    def generate_from_milp_instance(self,
                                  milp_instance,  # MILPInstance类型
                                  include_perturbation_info: bool = True) -> BipartiteGraphRepresentation:
        """
        从现有的MILP实例生成二分图表示
        
        Args:
            milp_instance: MILP实例对象（来自BiasedMILPGenerator）
            include_perturbation_info: 是否包含扰动信息
            
        Returns:
            二分图表示对象
        """
        # 生成二分图
        bipartite_graph = self.generate_from_cvxpy_problem(
            milp_instance.cvxpy_problem,
            milp_instance.instance_id,
            milp_instance.perturbed_system_data
        )
        
        # 更新扰动信息
        if include_perturbation_info:
            bipartite_graph.perturbation_applied = milp_instance.perturbation_config is not None
            bipartite_graph.source_problem_name = milp_instance.problem_name
        
        return bipartite_graph


def create_g2milp_generator(include_power_system_semantics: bool = True) -> G2MILPBipartiteGenerator:
    """
    创建G2MILP二分图生成器的工厂函数
    
    Args:
        include_power_system_semantics: 是否包含电力系统语义
        
    Returns:
        G2MILP二分图生成器实例
    """
    return G2MILPBipartiteGenerator(include_power_system_semantics)


if __name__ == "__main__":
    # 示例用法
    print("G2MILP二分图表示框架")
    print("=" * 50)
    
    # 这里只是演示代码结构，实际使用时需要加载真实的CVXPY问题
    print("主要功能:")
    print("1. MILP实例到二分图的转换")
    print("2. 约束节点和变量节点的特征提取")
    print("3. 变量节点的标准9维特征向量")
    print("4. 丰富的边特征计算")
    print("5. 支持PyTorch Geometric和DGL格式")
    print("6. 与现有MILP生成器的无缝集成")
    print()
    print("请参考文档和测试用例了解详细使用方法")
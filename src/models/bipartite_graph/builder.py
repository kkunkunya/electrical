"""
二分图构建器
从MILP标准形式构建二分图表示

主要功能:
1. 从约束矩阵构建变量节点和约束节点
2. 创建约束-变量边
3. 计算节点特征向量
4. 优化图结构存储
5. 支持增量构建和批处理
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Set, Any
from datetime import datetime
import scipy.sparse as sp

from .data_structures import (
    VariableNode, ConstraintNode, BipartiteEdge, BipartiteGraph,
    VariableType, ConstraintType, GraphStatistics
)
from .extractor import MILPStandardForm

logger = logging.getLogger(__name__)


class BipartiteGraphBuilder:
    """
    二分图构建器
    负责从MILP标准形式构建完整的二分图表示
    """
    
    def __init__(self, 
                 normalize_coefficients: bool = True,
                 sparse_threshold: float = 0.1,
                 batch_size: int = 1000):
        """
        初始化构建器
        
        Args:
            normalize_coefficients: 是否归一化系数
            sparse_threshold: 稀疏矩阵密度阈值
            batch_size: 批处理大小
        """
        self.normalize_coefficients = normalize_coefficients
        self.sparse_threshold = sparse_threshold
        self.batch_size = batch_size
        
        # 构建过程记录
        self.build_log: List[str] = []
        self.build_statistics: Dict[str, Any] = {}
        
        logger.info(f"二分图构建器初始化完成")
        logger.info(f"  归一化系数: {normalize_coefficients}")
        logger.info(f"  稀疏阈值: {sparse_threshold}")
        logger.info(f"  批处理大小: {batch_size}")
    
    def build_graph(self, 
                   standard_form: MILPStandardForm,
                   graph_id: str = None) -> BipartiteGraph:
        """
        构建二分图
        
        Args:
            standard_form: MILP标准形式对象
            graph_id: 图标识符
            
        Returns:
            完整的二分图对象
        """
        logger.info("=" * 60)
        logger.info("开始构建MILP二分图")
        logger.info("=" * 60)
        
        start_time = datetime.now()
        
        if graph_id is None:
            graph_id = f"bipartite_graph_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]}"
        
        try:
            # 1. 创建二分图对象
            bipartite_graph = BipartiteGraph(
                graph_id=graph_id,
                source_problem_id=standard_form.problem_name
            )
            
            # 2. 构建变量节点
            logger.info("构建变量节点...")
            self._build_variable_nodes(bipartite_graph, standard_form)
            
            # 3. 构建约束节点
            logger.info("构建约束节点...")
            self._build_constraint_nodes(bipartite_graph, standard_form)
            
            # 4. 构建边
            logger.info("构建边连接...")
            self._build_edges(bipartite_graph, standard_form)
            
            # 5. 更新节点度数
            logger.info("更新节点度数...")
            bipartite_graph.update_node_degrees()
            
            # 6. 更新变量节点统计信息
            logger.info("更新变量节点统计...")
            self._update_variable_statistics(bipartite_graph)
            
            # 7. 归一化系数（如果需要）
            if self.normalize_coefficients:
                logger.info("归一化边系数...")
                self._normalize_edge_coefficients(bipartite_graph)
            
            # 8. 计算图统计信息
            logger.info("计算图统计信息...")
            bipartite_graph.compute_statistics()
            
            # 9. 验证图结构
            logger.info("验证图结构...")
            self._validate_graph(bipartite_graph, standard_form)
            
            build_duration = (datetime.now() - start_time).total_seconds()
            bipartite_graph.statistics.build_duration = build_duration
            
            logger.info("=" * 60)
            logger.info("✅ 二分图构建完成!")
            logger.info("=" * 60)
            logger.info(f"⏱️  构建耗时: {build_duration:.3f} 秒")
            logger.info(f"📊 变量节点: {len(bipartite_graph.variable_nodes)}")
            logger.info(f"📊 约束节点: {len(bipartite_graph.constraint_nodes)}")
            logger.info(f"📊 边数量: {len(bipartite_graph.edges)}")
            logger.info(f"📊 图密度: {bipartite_graph.statistics.density:.4f}")
            logger.info("=" * 60)
            
            return bipartite_graph
            
        except Exception as e:
            logger.error(f"二分图构建失败: {e}")
            self.build_log.append(f"构建失败: {e}")
            raise
    
    def _build_variable_nodes(self, graph: BipartiteGraph, standard_form: MILPStandardForm):
        """构建变量节点"""
        n_variables = standard_form.n_variables
        
        for i in range(n_variables):
            var_info = standard_form.variable_info[i]
            
            # 创建变量节点
            var_node = VariableNode(
                node_id=var_info['variable_id'],
                cvxpy_var_name=var_info['cvxpy_var_name'],
                original_shape=var_info['original_shape'],
                flat_index=var_info['flat_index'],
                var_type=standard_form.variable_types[i],
                lower_bound=standard_form.lower_bounds[i],
                upper_bound=standard_form.upper_bounds[i],
                obj_coeff=standard_form.objective_coefficients[i],
                has_lower_bound=np.isfinite(standard_form.lower_bounds[i]),
                has_upper_bound=np.isfinite(standard_form.upper_bounds[i]),
                metadata={
                    'cvxpy_var_id': var_info.get('cvxpy_var_id'),
                    'original_index': var_info.get('original_index'),
                    'offset_in_problem': var_info.get('offset_in_problem')
                }
            )
            
            graph.add_variable_node(var_node)
            
            if (i + 1) % self.batch_size == 0:
                logger.debug(f"已处理变量节点: {i + 1}/{n_variables}")
        
        logger.info(f"变量节点构建完成: {len(graph.variable_nodes)} 个")
    
    def _build_constraint_nodes(self, graph: BipartiteGraph, standard_form: MILPStandardForm):
        """构建约束节点"""
        n_constraints = standard_form.n_constraints
        
        for i in range(n_constraints):
            constraint_info = standard_form.constraint_info[i]
            
            # 确定约束方向
            sense = standard_form.constraint_senses[i] if i < len(standard_form.constraint_senses) else "=="
            
            # 创建约束节点
            constraint_node = ConstraintNode(
                node_id=constraint_info['constraint_id'],
                constraint_name=constraint_info['constraint_name'],
                constraint_type=standard_form.constraint_types[i],
                rhs_value=standard_form.rhs_vector[i],
                sense=sense,
                metadata={
                    'original_index': constraint_info.get('original_index'),
                    'source': constraint_info.get('source'),
                    'soc_group': constraint_info.get('soc_group')
                }
            )
            
            graph.add_constraint_node(constraint_node)
            
            if (i + 1) % self.batch_size == 0:
                logger.debug(f"已处理约束节点: {i + 1}/{n_constraints}")
        
        logger.info(f"约束节点构建完成: {len(graph.constraint_nodes)} 个")
    
    def _build_edges(self, graph: BipartiteGraph, standard_form: MILPStandardForm):
        """构建边连接"""
        constraint_matrix = standard_form.constraint_matrix
        
        # 处理稀疏矩阵和稠密矩阵
        if sp.issparse(constraint_matrix):
            self._build_edges_sparse(graph, constraint_matrix, standard_form)
        else:
            self._build_edges_dense(graph, constraint_matrix, standard_form)
    
    def _build_edges_sparse(self, graph: BipartiteGraph, matrix: sp.csr_matrix, standard_form: MILPStandardForm):
        """从稀疏矩阵构建边"""
        edge_count = 0
        
        # 获取非零元素
        coo_matrix = matrix.tocoo()
        
        for matrix_row, matrix_col, coeff in zip(coo_matrix.row, coo_matrix.col, coo_matrix.data):
            if abs(coeff) > 1e-12:  # 过滤极小值
                # 获取对应的节点ID
                constraint_id = standard_form.constraint_info[matrix_row]['constraint_id']
                variable_id = standard_form.variable_info[matrix_col]['variable_id']
                
                # 创建边
                edge_id = f"edge_{constraint_id}_{variable_id}"
                edge = BipartiteEdge(
                    edge_id=edge_id,
                    constraint_node_id=constraint_id,
                    variable_node_id=variable_id,
                    coefficient=float(coeff)
                )
                
                graph.add_edge(edge)
                
                # 更新约束节点的系数信息
                constraint_node = graph.constraint_nodes[constraint_id]
                constraint_node.lhs_coefficients[variable_id] = float(coeff)
                
                edge_count += 1
                
                if edge_count % self.batch_size == 0:
                    logger.debug(f"已处理边: {edge_count}")
        
        logger.info(f"稀疏边构建完成: {edge_count} 条边")
    
    def _build_edges_dense(self, graph: BipartiteGraph, matrix: np.ndarray, standard_form: MILPStandardForm):
        """从稠密矩阵构建边"""
        edge_count = 0
        n_constraints, n_variables = matrix.shape
        
        for i in range(n_constraints):
            for j in range(n_variables):
                coeff = matrix[i, j]
                
                if abs(coeff) > 1e-12:  # 过滤极小值
                    # 获取对应的节点ID
                    constraint_id = standard_form.constraint_info[i]['constraint_id']
                    variable_id = standard_form.variable_info[j]['variable_id']
                    
                    # 创建边
                    edge_id = f"edge_{constraint_id}_{variable_id}"
                    edge = BipartiteEdge(
                        edge_id=edge_id,
                        constraint_node_id=constraint_id,
                        variable_node_id=variable_id,
                        coefficient=float(coeff)
                    )
                    
                    graph.add_edge(edge)
                    
                    # 更新约束节点的系数信息
                    constraint_node = graph.constraint_nodes[constraint_id]
                    constraint_node.lhs_coefficients[variable_id] = float(coeff)
                    
                    edge_count += 1
        
        logger.info(f"稠密边构建完成: {edge_count} 条边")
    
    def _update_variable_statistics(self, graph: BipartiteGraph):
        """更新变量节点的统计信息"""
        for var_id, var_node in graph.variable_nodes.items():
            # 收集该变量在各约束中的系数
            coefficients = []
            constraint_types = set()
            
            for edge in graph.edges.values():
                if edge.variable_node_id == var_id:
                    coefficients.append(edge.coefficient)
                    
                    # 获取约束类型
                    constraint_node = graph.constraint_nodes[edge.constraint_node_id]
                    constraint_types.add(constraint_node.constraint_type)
            
            # 更新变量节点信息
            var_node.constraint_types = constraint_types
            var_node.update_statistics(coefficients)
        
        logger.info(f"变量统计信息更新完成")
    
    def _normalize_edge_coefficients(self, graph: BipartiteGraph):
        """归一化边系数"""
        # 收集所有系数
        all_coeffs = [edge.coefficient for edge in graph.edges.values()]
        
        if not all_coeffs:
            return
        
        # 计算归一化参数（使用最大绝对值）
        max_abs_coeff = max(abs(c) for c in all_coeffs)
        
        if max_abs_coeff > 0:
            # 归一化所有边的系数
            for edge in graph.edges.values():
                edge.normalized_coeff = edge.coefficient / max_abs_coeff
            
            logger.info(f"系数归一化完成，归一化因子: {max_abs_coeff:.2e}")
        else:
            logger.warning("所有系数均为零，跳过归一化")
    
    def _validate_graph(self, graph: BipartiteGraph, standard_form: MILPStandardForm):
        """验证图结构的有效性"""
        # 检查节点数量
        if len(graph.variable_nodes) != standard_form.n_variables:
            raise ValueError(f"变量节点数量不匹配: {len(graph.variable_nodes)} vs {standard_form.n_variables}")
        
        if len(graph.constraint_nodes) != standard_form.n_constraints:
            raise ValueError(f"约束节点数量不匹配: {len(graph.constraint_nodes)} vs {standard_form.n_constraints}")
        
        # 检查边的有效性
        invalid_edges = 0
        for edge in graph.edges.values():
            if edge.constraint_node_id not in graph.constraint_nodes:
                invalid_edges += 1
            if edge.variable_node_id not in graph.variable_nodes:
                invalid_edges += 1
        
        if invalid_edges > 0:
            raise ValueError(f"发现 {invalid_edges} 条无效边")
        
        # 检查邻接信息一致性
        for var_id in graph.variable_nodes:
            neighbors_from_edges = set()
            for edge in graph.edges.values():
                if edge.variable_node_id == var_id:
                    neighbors_from_edges.add(edge.constraint_node_id)
            
            neighbors_from_adj = graph.variable_to_constraints.get(var_id, set())
            
            if neighbors_from_edges != neighbors_from_adj:
                raise ValueError(f"变量 {var_id} 的邻接信息不一致")
        
        # 更新约束节点的系数统计
        for constraint_node in graph.constraint_nodes.values():
            constraint_node.update_coefficient_statistics()
        
        logger.info("✅ 图结构验证通过")
    
    def build_batch_graphs(self, 
                          standard_forms: List[MILPStandardForm],
                          graph_id_prefix: str = "batch_graph") -> List[BipartiteGraph]:
        """
        批量构建二分图
        
        Args:
            standard_forms: MILP标准形式列表
            graph_id_prefix: 图ID前缀
            
        Returns:
            二分图列表
        """
        logger.info(f"开始批量构建 {len(standard_forms)} 个二分图...")
        
        graphs = []
        
        for i, standard_form in enumerate(standard_forms):
            try:
                graph_id = f"{graph_id_prefix}_{i:03d}"
                logger.info(f"构建第 {i+1}/{len(standard_forms)} 个图: {graph_id}")
                
                graph = self.build_graph(standard_form, graph_id)
                graphs.append(graph)
                
            except Exception as e:
                logger.error(f"构建第 {i+1} 个图失败: {e}")
                continue
        
        logger.info(f"批量构建完成，成功构建 {len(graphs)} 个图")
        return graphs
    
    def get_build_report(self) -> Dict[str, Any]:
        """获取构建过程报告"""
        return {
            "builder_config": {
                "normalize_coefficients": self.normalize_coefficients,
                "sparse_threshold": self.sparse_threshold,
                "batch_size": self.batch_size
            },
            "build_log": self.build_log,
            "build_statistics": self.build_statistics
        }


def build_bipartite_graph(standard_form: MILPStandardForm,
                         graph_id: str = None,
                         normalize_coefficients: bool = True) -> BipartiteGraph:
    """
    便捷函数：构建二分图
    
    Args:
        standard_form: MILP标准形式对象
        graph_id: 图标识符
        normalize_coefficients: 是否归一化系数
        
    Returns:
        二分图对象
    """
    builder = BipartiteGraphBuilder(normalize_coefficients=normalize_coefficients)
    return builder.build_graph(standard_form, graph_id)


if __name__ == "__main__":
    """测试二分图构建器"""
    import sys
    from pathlib import Path
    
    # 添加项目路径
    project_root = Path(__file__).parent.parent.parent.parent
    sys.path.insert(0, str(project_root))
    
    from .extractor import extract_from_cvxpy_problem
    import cvxpy as cp
    
    # 创建测试问题
    logger.info("创建测试CVXPY问题...")
    
    x = cp.Variable(3, name='x')
    y = cp.Variable(2, boolean=True, name='y')
    
    constraints = [
        x[0] + x[1] + x[2] <= 10,
        2*x[0] - x[1] == 5,
        x >= 0,
        y[0] + y[1] <= 1
    ]
    
    objective = cp.Minimize(3*x[0] + 2*x[1] + x[2] + 5*y[0] + 3*y[1])
    problem = cp.Problem(objective, constraints)
    
    try:
        # 提取标准形式
        standard_form = extract_from_cvxpy_problem(problem, "测试问题")
        
        # 构建二分图
        builder = BipartiteGraphBuilder()
        graph = builder.build_graph(standard_form)
        
        print("✅ 二分图构建测试成功!")
        print(graph.summary())
        
        # 测试特征向量
        for var_id, var_node in list(graph.variable_nodes.items())[:3]:
            features = var_node.get_feature_vector()
            print(f"变量 {var_id} 特征向量: {features}")
        
        for cons_id, cons_node in list(graph.constraint_nodes.items())[:3]:
            features = cons_node.get_constraint_features()
            print(f"约束 {cons_id} 特征向量: {features}")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
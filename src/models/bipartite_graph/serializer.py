"""
二分图序列化工具
提供二分图的保存、加载和格式转换功能

支持格式:
1. Pickle - 完整的Python对象序列化
2. JSON - 轻量级数据交换格式  
3. NetworkX - 图分析兼容格式
4. NumPy - 矩阵格式
5. 自定义HDF5 - 大规模图数据存储
"""

import json
import pickle
import numpy as np
import logging
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from datetime import datetime
import gzip

from .data_structures import BipartiteGraph, VariableNode, ConstraintNode, BipartiteEdge

logger = logging.getLogger(__name__)

try:
    import h5py
    HDF5_AVAILABLE = True
except ImportError:
    HDF5_AVAILABLE = False
    logger.warning("h5py不可用，HDF5格式不支持")

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    logger.warning("NetworkX不可用，图格式转换受限")


class BipartiteGraphSerializer:
    """
    二分图序列化工具
    支持多种格式的保存和加载
    """
    
    def __init__(self, compression: bool = True):
        """
        初始化序列化器
        
        Args:
            compression: 是否启用压缩
        """
        self.compression = compression
        
        logger.info(f"二分图序列化器初始化完成，压缩: {compression}")
    
    def save_pickle(self, 
                   graph: BipartiteGraph, 
                   filepath: Union[str, Path],
                   protocol: int = pickle.HIGHEST_PROTOCOL) -> bool:
        """
        保存为Pickle格式
        
        Args:
            graph: 二分图对象
            filepath: 保存路径
            protocol: Pickle协议版本
            
        Returns:
            是否保存成功
        """
        try:
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            if self.compression:
                # 使用gzip压缩
                if not str(filepath).endswith('.gz'):
                    filepath = filepath.with_suffix(filepath.suffix + '.gz')
                
                with gzip.open(filepath, 'wb') as f:
                    pickle.dump(graph, f, protocol=protocol)
            else:
                with open(filepath, 'wb') as f:
                    pickle.dump(graph, f, protocol=protocol)
            
            logger.info(f"二分图已保存为Pickle格式: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Pickle保存失败: {e}")
            return False
    
    def load_pickle(self, filepath: Union[str, Path]) -> Optional[BipartiteGraph]:
        """
        从Pickle格式加载
        
        Args:
            filepath: 文件路径
            
        Returns:
            二分图对象或None
        """
        try:
            filepath = Path(filepath)
            
            if str(filepath).endswith('.gz'):
                with gzip.open(filepath, 'rb') as f:
                    graph = pickle.load(f)
            else:
                with open(filepath, 'rb') as f:
                    graph = pickle.load(f)
            
            logger.info(f"从Pickle格式加载二分图: {filepath}")
            return graph
            
        except Exception as e:
            logger.error(f"Pickle加载失败: {e}")
            return None
    
    def save_json(self, 
                 graph: BipartiteGraph, 
                 filepath: Union[str, Path],
                 include_features: bool = True) -> bool:
        """
        保存为JSON格式
        
        Args:
            graph: 二分图对象
            filepath: 保存路径
            include_features: 是否包含特征向量
            
        Returns:
            是否保存成功
        """
        try:
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            # 转换为JSON兼容格式
            graph_data = self._graph_to_json_dict(graph, include_features)
            
            if self.compression and not str(filepath).endswith('.gz'):
                filepath = filepath.with_suffix(filepath.suffix + '.gz')
                with gzip.open(filepath, 'wt', encoding='utf-8') as f:
                    json.dump(graph_data, f, indent=2, ensure_ascii=False, default=str)
            else:
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(graph_data, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"二分图已保存为JSON格式: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"JSON保存失败: {e}")
            return False
    
    def load_json(self, filepath: Union[str, Path]) -> Optional[BipartiteGraph]:
        """
        从JSON格式加载
        
        Args:
            filepath: 文件路径
            
        Returns:
            二分图对象或None
        """
        try:
            filepath = Path(filepath)
            
            if str(filepath).endswith('.gz'):
                with gzip.open(filepath, 'rt', encoding='utf-8') as f:
                    graph_data = json.load(f)
            else:
                with open(filepath, 'r', encoding='utf-8') as f:
                    graph_data = json.load(f)
            
            # 从JSON数据重建图对象
            graph = self._json_dict_to_graph(graph_data)
            
            logger.info(f"从JSON格式加载二分图: {filepath}")
            return graph
            
        except Exception as e:
            logger.error(f"JSON加载失败: {e}")
            return None
    
    def save_hdf5(self, 
                 graph: BipartiteGraph, 
                 filepath: Union[str, Path]) -> bool:
        """
        保存为HDF5格式（适用于大规模图）
        
        Args:
            graph: 二分图对象
            filepath: 保存路径
            
        Returns:
            是否保存成功
        """
        if not HDF5_AVAILABLE:
            logger.error("HDF5格式需要h5py库")
            return False
        
        try:
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            with h5py.File(filepath, 'w') as f:
                # 保存图基本信息
                f.attrs['graph_id'] = graph.graph_id
                f.attrs['source_problem_id'] = graph.source_problem_id
                f.attrs['creation_time'] = graph.creation_time.isoformat()
                f.attrs['n_variable_nodes'] = len(graph.variable_nodes)
                f.attrs['n_constraint_nodes'] = len(graph.constraint_nodes)
                f.attrs['n_edges'] = len(graph.edges)
                
                # 保存变量节点
                var_group = f.create_group('variable_nodes')
                var_ids = list(graph.variable_nodes.keys())
                var_features = np.array([node.get_feature_vector() 
                                       for node in graph.variable_nodes.values()])
                
                var_group.create_dataset('node_ids', data=[id.encode('utf-8') for id in var_ids])
                var_group.create_dataset('features', data=var_features)
                
                # 保存变量节点元信息
                for i, (var_id, var_node) in enumerate(graph.variable_nodes.items()):
                    node_group = var_group.create_group(f'node_{i}')
                    node_group.attrs['node_id'] = var_id
                    node_group.attrs['cvxpy_var_name'] = var_node.cvxpy_var_name
                    node_group.attrs['var_type'] = var_node.var_type.value
                    node_group.attrs['lower_bound'] = var_node.lower_bound
                    node_group.attrs['upper_bound'] = var_node.upper_bound
                    node_group.attrs['obj_coeff'] = var_node.obj_coeff
                
                # 保存约束节点
                cons_group = f.create_group('constraint_nodes')
                cons_ids = list(graph.constraint_nodes.keys())
                cons_features = np.array([node.get_constraint_features() 
                                        for node in graph.constraint_nodes.values()])
                
                cons_group.create_dataset('node_ids', data=[id.encode('utf-8') for id in cons_ids])
                cons_group.create_dataset('features', data=cons_features)
                
                # 保存约束节点元信息
                for i, (cons_id, cons_node) in enumerate(graph.constraint_nodes.items()):
                    node_group = cons_group.create_group(f'node_{i}')
                    node_group.attrs['node_id'] = cons_id
                    node_group.attrs['constraint_name'] = cons_node.constraint_name
                    node_group.attrs['constraint_type'] = cons_node.constraint_type.value
                    node_group.attrs['rhs_value'] = cons_node.rhs_value
                    node_group.attrs['sense'] = cons_node.sense
                
                # 保存边信息
                edge_group = f.create_group('edges')
                edge_data = []
                for edge in graph.edges.values():
                    var_idx = var_ids.index(edge.variable_node_id)
                    cons_idx = cons_ids.index(edge.constraint_node_id)
                    edge_data.append([var_idx, cons_idx, edge.coefficient])
                
                edge_array = np.array(edge_data)
                edge_group.create_dataset('edge_list', data=edge_array)
                
                # 保存邻接矩阵（稀疏格式）
                adj_matrix = graph.get_adjacency_matrix(sparse=True)
                if hasattr(adj_matrix, 'tocoo'):
                    coo = adj_matrix.tocoo()
                    adj_group = f.create_group('adjacency_matrix')
                    adj_group.create_dataset('row', data=coo.row)
                    adj_group.create_dataset('col', data=coo.col)
                    adj_group.create_dataset('data', data=coo.data)
                    adj_group.attrs['shape'] = coo.shape
                
                # 保存统计信息
                stats_group = f.create_group('statistics')
                stats_dict = graph.statistics.to_dict()
                for key, value in stats_dict.items():
                    if isinstance(value, dict):
                        sub_group = stats_group.create_group(key)
                        for sub_key, sub_value in value.items():
                            sub_group.attrs[sub_key] = sub_value
                    else:
                        stats_group.attrs[key] = value
            
            logger.info(f"二分图已保存为HDF5格式: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"HDF5保存失败: {e}")
            return False
    
    def load_hdf5(self, filepath: Union[str, Path]) -> Optional[BipartiteGraph]:
        """
        从HDF5格式加载
        
        Args:
            filepath: 文件路径
            
        Returns:
            二分图对象或None
        """
        if not HDF5_AVAILABLE:
            logger.error("HDF5格式需要h5py库")
            return None
        
        try:
            with h5py.File(filepath, 'r') as f:
                # 重建基本图信息
                graph = BipartiteGraph(
                    graph_id=f.attrs['graph_id'],
                    source_problem_id=f.attrs['source_problem_id']
                )
                
                # 重建变量节点
                var_group = f['variable_nodes']
                var_ids = [id.decode('utf-8') for id in var_group['node_ids']]
                
                for i, var_id in enumerate(var_ids):
                    node_group = var_group[f'node_{i}']
                    
                    # 从属性重建变量节点（简化版本）
                    var_node = VariableNode(
                        node_id=var_id,
                        cvxpy_var_name=node_group.attrs['cvxpy_var_name'],
                        original_shape=(),  # 简化
                        flat_index=i,
                        var_type=VariableType(node_group.attrs['var_type']),
                        lower_bound=node_group.attrs['lower_bound'],
                        upper_bound=node_group.attrs['upper_bound'],
                        obj_coeff=node_group.attrs['obj_coeff']
                    )
                    
                    graph.add_variable_node(var_node)
                
                # 重建约束节点
                cons_group = f['constraint_nodes']
                cons_ids = [id.decode('utf-8') for id in cons_group['node_ids']]
                
                for i, cons_id in enumerate(cons_ids):
                    node_group = cons_group[f'node_{i}']
                    
                    cons_node = ConstraintNode(
                        node_id=cons_id,
                        constraint_name=node_group.attrs['constraint_name'],
                        constraint_type=ConstraintType(node_group.attrs['constraint_type']),
                        rhs_value=node_group.attrs['rhs_value'],
                        sense=node_group.attrs['sense']
                    )
                    
                    graph.add_constraint_node(cons_node)
                
                # 重建边
                edge_group = f['edges']
                edge_list = edge_group['edge_list'][:]
                
                for var_idx, cons_idx, coeff in edge_list:
                    var_id = var_ids[int(var_idx)]
                    cons_id = cons_ids[int(cons_idx)]
                    edge_id = f"edge_{cons_id}_{var_id}"
                    
                    edge = BipartiteEdge(
                        edge_id=edge_id,
                        constraint_node_id=cons_id,
                        variable_node_id=var_id,
                        coefficient=float(coeff)
                    )
                    
                    graph.add_edge(edge)
                
                # 更新度数和统计
                graph.update_node_degrees()
                graph.compute_statistics()
            
            logger.info(f"从HDF5格式加载二分图: {filepath}")
            return graph
            
        except Exception as e:
            logger.error(f"HDF5加载失败: {e}")
            return None
    
    def to_networkx(self, graph: BipartiteGraph) -> Optional['nx.Graph']:
        """
        转换为NetworkX图对象
        
        Args:
            graph: 二分图对象
            
        Returns:
            NetworkX图对象或None
        """
        if not NETWORKX_AVAILABLE:
            logger.error("NetworkX转换需要networkx库")
            return None
        
        try:
            # 创建NetworkX二分图
            G = nx.Graph()
            
            # 添加变量节点
            for var_id, var_node in graph.variable_nodes.items():
                G.add_node(var_id, 
                          bipartite=0,  # 变量节点集合
                          node_type='variable',
                          var_type=var_node.var_type.value,
                          lower_bound=var_node.lower_bound,
                          upper_bound=var_node.upper_bound,
                          obj_coeff=var_node.obj_coeff,
                          features=var_node.get_feature_vector().tolist())
            
            # 添加约束节点
            for cons_id, cons_node in graph.constraint_nodes.items():
                G.add_node(cons_id,
                          bipartite=1,  # 约束节点集合
                          node_type='constraint',
                          constraint_type=cons_node.constraint_type.value,
                          rhs_value=cons_node.rhs_value,
                          sense=cons_node.sense,
                          features=cons_node.get_constraint_features().tolist())
            
            # 添加边
            for edge in graph.edges.values():
                G.add_edge(edge.variable_node_id, 
                          edge.constraint_node_id,
                          weight=edge.coefficient,
                          abs_weight=edge.abs_coefficient)
            
            logger.info(f"转换为NetworkX图: {G.number_of_nodes()} 节点, {G.number_of_edges()} 边")
            return G
            
        except Exception as e:
            logger.error(f"NetworkX转换失败: {e}")
            return None
    
    def from_networkx(self, G: 'nx.Graph', 
                     graph_id: str = None,
                     source_problem_id: str = None) -> Optional[BipartiteGraph]:
        """
        从NetworkX图对象创建二分图
        
        Args:
            G: NetworkX图对象
            graph_id: 图ID
            source_problem_id: 源问题ID
            
        Returns:
            二分图对象或None
        """
        if not NETWORKX_AVAILABLE:
            logger.error("NetworkX转换需要networkx库")
            return None
        
        try:
            if graph_id is None:
                graph_id = f"from_networkx_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            graph = BipartiteGraph(
                graph_id=graph_id,
                source_problem_id=source_problem_id or "networkx_import"
            )
            
            # 重建节点
            for node_id, node_data in G.nodes(data=True):
                if node_data.get('node_type') == 'variable':
                    var_node = VariableNode(
                        node_id=node_id,
                        cvxpy_var_name=node_id,
                        original_shape=(),
                        flat_index=0,
                        var_type=VariableType(node_data.get('var_type', 'continuous')),
                        lower_bound=node_data.get('lower_bound', -np.inf),
                        upper_bound=node_data.get('upper_bound', np.inf),
                        obj_coeff=node_data.get('obj_coeff', 0.0)
                    )
                    graph.add_variable_node(var_node)
                
                elif node_data.get('node_type') == 'constraint':
                    cons_node = ConstraintNode(
                        node_id=node_id,
                        constraint_name=node_id,
                        constraint_type=ConstraintType(node_data.get('constraint_type', 'linear_inequality')),
                        rhs_value=node_data.get('rhs_value', 0.0),
                        sense=node_data.get('sense', '<=')
                    )
                    graph.add_constraint_node(cons_node)
            
            # 重建边
            for u, v, edge_data in G.edges(data=True):
                # 确定哪个是变量节点，哪个是约束节点
                u_data = G.nodes[u]
                v_data = G.nodes[v]
                
                if u_data.get('node_type') == 'variable':
                    var_id, cons_id = u, v
                else:
                    var_id, cons_id = v, u
                
                edge = BipartiteEdge(
                    edge_id=f"edge_{cons_id}_{var_id}",
                    constraint_node_id=cons_id,
                    variable_node_id=var_id,
                    coefficient=edge_data.get('weight', 1.0)
                )
                graph.add_edge(edge)
            
            # 更新度数和统计
            graph.update_node_degrees()
            graph.compute_statistics()
            
            logger.info(f"从NetworkX创建二分图: {len(graph.variable_nodes)} 变量, {len(graph.constraint_nodes)} 约束")
            return graph
            
        except Exception as e:
            logger.error(f"从NetworkX创建失败: {e}")
            return None
    
    def export_matrix_format(self, 
                           graph: BipartiteGraph,
                           output_dir: Union[str, Path]) -> bool:
        """
        导出矩阵格式（适用于外部分析工具）
        
        Args:
            graph: 二分图对象
            output_dir: 输出目录
            
        Returns:
            是否导出成功
        """
        try:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # 导出邻接矩阵
            adj_matrix = graph.get_adjacency_matrix(sparse=False)
            np.savetxt(output_dir / 'adjacency_matrix.csv', adj_matrix, delimiter=',')
            
            # 导出变量特征矩阵
            var_features = np.array([node.get_feature_vector() 
                                   for node in graph.variable_nodes.values()])
            np.savetxt(output_dir / 'variable_features.csv', var_features, delimiter=',')
            
            # 导出约束特征矩阵
            cons_features = np.array([node.get_constraint_features() 
                                    for node in graph.constraint_nodes.values()])
            np.savetxt(output_dir / 'constraint_features.csv', cons_features, delimiter=',')
            
            # 导出节点ID映射
            var_ids = list(graph.variable_nodes.keys())
            cons_ids = list(graph.constraint_nodes.keys())
            
            with open(output_dir / 'variable_ids.txt', 'w') as f:
                f.write('\n'.join(var_ids))
            
            with open(output_dir / 'constraint_ids.txt', 'w') as f:
                f.write('\n'.join(cons_ids))
            
            # 导出边列表
            edges_data = []
            for edge in graph.edges.values():
                var_idx = var_ids.index(edge.variable_node_id)
                cons_idx = cons_ids.index(edge.constraint_node_id)
                edges_data.append([var_idx, cons_idx, edge.coefficient])
            
            np.savetxt(output_dir / 'edge_list.csv', edges_data, delimiter=',',
                      header='variable_index,constraint_index,coefficient')
            
            # 导出元信息
            meta_info = {
                'graph_id': graph.graph_id,
                'n_variables': len(graph.variable_nodes),
                'n_constraints': len(graph.constraint_nodes),
                'n_edges': len(graph.edges),
                'export_time': datetime.now().isoformat()
            }
            
            with open(output_dir / 'metadata.json', 'w') as f:
                json.dump(meta_info, f, indent=2)
            
            logger.info(f"矩阵格式已导出到: {output_dir}")
            return True
            
        except Exception as e:
            logger.error(f"矩阵格式导出失败: {e}")
            return False
    
    def _graph_to_json_dict(self, graph: BipartiteGraph, include_features: bool) -> Dict[str, Any]:
        """将图对象转换为JSON兼容字典"""
        graph_dict = graph.to_dict()
        
        if not include_features:
            # 移除特征向量以减小文件大小
            for var_data in graph_dict['variable_nodes'].values():
                var_data.pop('feature_vector', None)
            
            for cons_data in graph_dict['constraint_nodes'].values():
                cons_data.pop('feature_vector', None)
            
            for edge_data in graph_dict['edges'].values():
                edge_data.pop('feature_vector', None)
        
        return graph_dict
    
    def _json_dict_to_graph(self, graph_data: Dict[str, Any]) -> BipartiteGraph:
        """从JSON字典重建图对象"""
        from .data_structures import VariableType, ConstraintType
        
        # 创建基本图对象
        graph = BipartiteGraph(
            graph_id=graph_data['graph_id'],
            source_problem_id=graph_data['source_problem_id']
        )
        
        # 重建变量节点
        for var_id, var_data in graph_data['variable_nodes'].items():
            var_node = VariableNode(
                node_id=var_data['node_id'],
                cvxpy_var_name=var_data['cvxpy_var_name'],
                original_shape=tuple(var_data['original_shape']),
                flat_index=var_data['flat_index'],
                var_type=VariableType(var_data['var_type']),
                lower_bound=var_data['lower_bound'],
                upper_bound=var_data['upper_bound'],
                obj_coeff=var_data['obj_coeff'],
                has_lower_bound=var_data['has_lower_bound'],
                has_upper_bound=var_data['has_upper_bound'],
                degree=var_data['degree'],
                constraint_types={ConstraintType(ct) for ct in var_data['constraint_types']},
                coeff_statistics=var_data['coeff_statistics'],
                metadata=var_data['metadata']
            )
            
            graph.add_variable_node(var_node)
        
        # 重建约束节点
        for cons_id, cons_data in graph_data['constraint_nodes'].items():
            cons_node = ConstraintNode(
                node_id=cons_data['node_id'],
                constraint_name=cons_data['constraint_name'],
                constraint_type=ConstraintType(cons_data['constraint_type']),
                lhs_coefficients=cons_data['lhs_coefficients'],
                rhs_value=cons_data['rhs_value'],
                sense=cons_data['sense'],
                nnz_count=cons_data['nnz_count'],
                coefficient_range=tuple(cons_data['coefficient_range']),
                degree=cons_data['degree'],
                is_binding=cons_data['is_binding'],
                slack_value=cons_data['slack_value'],
                dual_value=cons_data['dual_value'],
                metadata=cons_data['metadata']
            )
            
            graph.add_constraint_node(cons_node)
        
        # 重建边
        for edge_id, edge_data in graph_data['edges'].items():
            edge = BipartiteEdge(
                edge_id=edge_data['edge_id'],
                constraint_node_id=edge_data['constraint_node_id'],
                variable_node_id=edge_data['variable_node_id'],
                coefficient=edge_data['coefficient'],
                normalized_coeff=edge_data['normalized_coeff'],
                metadata=edge_data['metadata']
            )
            
            graph.add_edge(edge)
        
        # 重建邻接信息
        graph.variable_to_constraints = {
            var_id: set(cons_ids) 
            for var_id, cons_ids in graph_data['variable_to_constraints'].items()
        }
        graph.constraint_to_variables = {
            cons_id: set(var_ids)
            for cons_id, var_ids in graph_data['constraint_to_variables'].items()
        }
        
        # 重建统计信息
        graph.compute_statistics()
        
        return graph


# 便捷函数
def save_bipartite_graph(graph: BipartiteGraph, 
                        filepath: Union[str, Path],
                        format: str = 'pickle',
                        **kwargs) -> bool:
    """
    保存二分图（便捷函数）
    
    Args:
        graph: 二分图对象
        filepath: 保存路径
        format: 保存格式 ('pickle', 'json', 'hdf5')
        **kwargs: 其他参数
        
    Returns:
        是否保存成功
    """
    serializer = BipartiteGraphSerializer()
    
    if format.lower() == 'pickle':
        return serializer.save_pickle(graph, filepath, **kwargs)
    elif format.lower() == 'json':
        return serializer.save_json(graph, filepath, **kwargs)
    elif format.lower() == 'hdf5':
        return serializer.save_hdf5(graph, filepath, **kwargs)
    else:
        logger.error(f"不支持的格式: {format}")
        return False


def load_bipartite_graph(filepath: Union[str, Path],
                        format: str = 'auto') -> Optional[BipartiteGraph]:
    """
    加载二分图（便捷函数）
    
    Args:
        filepath: 文件路径
        format: 文件格式 ('auto', 'pickle', 'json', 'hdf5')
        
    Returns:
        二分图对象或None
    """
    filepath = Path(filepath)
    serializer = BipartiteGraphSerializer()
    
    if format.lower() == 'auto':
        # 根据文件扩展名自动判断格式
        suffix = filepath.suffix.lower()
        if suffix in ['.pkl', '.pickle']:
            format = 'pickle'
        elif suffix in ['.json']:
            format = 'json'
        elif suffix in ['.h5', '.hdf5']:
            format = 'hdf5'
        else:
            logger.error(f"无法自动识别格式: {filepath}")
            return None
    
    if format.lower() == 'pickle':
        return serializer.load_pickle(filepath)
    elif format.lower() == 'json':
        return serializer.load_json(filepath)
    elif format.lower() == 'hdf5':
        return serializer.load_hdf5(filepath)
    else:
        logger.error(f"不支持的格式: {format}")
        return None


if __name__ == "__main__":
    """测试序列化工具"""
    import sys
    from pathlib import Path
    
    # 这里需要一个示例图进行测试
    # 由于依赖其他模块，这里只提供框架
    logger.info("二分图序列化工具测试")
    print("✅ 序列化工具模块加载成功!")
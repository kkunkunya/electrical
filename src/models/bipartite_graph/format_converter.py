"""
Demo 3 到 Demo 4 数据格式转换器
Demo 3 to Demo 4 Data Format Converter

负责将Demo 3生成的BipartiteGraph格式转换为Demo 4需要的PyTorch Geometric异构图格式
"""

import torch
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
import logging
from datetime import datetime

try:
    from torch_geometric.data import HeteroData
    from src.models.bipartite_graph.data_structures import BipartiteGraph
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保安装了torch-geometric和项目模块")


class Demo3ToDemo4Converter:
    """Demo 3 BipartiteGraph 到 Demo 4 格式转换器"""
    
    def __init__(self, device: Optional[str] = None):
        """
        初始化转换器
        
        Args:
            device: 目标设备 ('cuda', 'cpu' 或 None自动检测)
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        self.logger.info(f"格式转换器初始化完成，目标设备: {self.device}")
    
    def convert_bipartite_graph(self, bipartite_graph: BipartiteGraph) -> Dict[str, Any]:
        """
        将Demo 3的BipartiteGraph转换为Demo 4的PyTorch Geometric格式
        
        Args:
            bipartite_graph: Demo 3生成的BipartiteGraph对象
            
        Returns:
            Dict包含转换后的数据和元数据
        """
        self.logger.info("开始转换Demo 3 BipartiteGraph格式...")
        
        # 1. 提取基本信息
        constraint_nodes = bipartite_graph.constraint_nodes
        variable_nodes = bipartite_graph.variable_nodes
        edges = bipartite_graph.edges
        
        self.logger.info(f"原始数据统计:")
        self.logger.info(f"  - 约束节点: {len(constraint_nodes)}")
        self.logger.info(f"  - 变量节点: {len(variable_nodes)}")
        self.logger.info(f"  - 边连接: {len(edges)}")
        
        # 2. 转换节点特征
        constraint_features = self._convert_constraint_features(constraint_nodes)
        variable_features = self._convert_variable_features(variable_nodes)
        
        # 3. 转换边连接和特征
        edge_indices, edge_features = self._convert_edges(edges, len(constraint_nodes), len(variable_nodes))
        
        # 4. 创建PyTorch Geometric异构图
        data = HeteroData()
        
        # 约束节点
        data['constraint'].x = torch.tensor(
            constraint_features, 
            dtype=torch.float32, 
            device=self.device,
            requires_grad=True
        )
        
        # 变量节点
        data['variable'].x = torch.tensor(
            variable_features, 
            dtype=torch.float32, 
            device=self.device,
            requires_grad=True
        )
        
        # 边连接和特征
        edge_index_tensor = torch.tensor(
            edge_indices, 
            dtype=torch.long, 
            device=self.device
        )
        
        data['constraint', 'connects', 'variable'].edge_index = edge_index_tensor
        
        data['constraint', 'connects', 'variable'].edge_attr = torch.tensor(
            edge_features, 
            dtype=torch.float32, 
            device=self.device,
            requires_grad=True
        )
        
        # 添加反向边连接（G2MILP模型需要）
        reverse_edge_indices = torch.stack([edge_index_tensor[1], edge_index_tensor[0]], dim=0)
        data['variable', 'connected_by', 'constraint'].edge_index = reverse_edge_indices
        data['variable', 'connected_by', 'constraint'].edge_attr = torch.tensor(
            edge_features, 
            dtype=torch.float32, 
            device=self.device,
            requires_grad=True
        )
        
        # 5. 验证转换结果
        self._validate_conversion(data, len(constraint_nodes), len(variable_nodes), len(edges))
        
        # 6. 创建返回数据
        result = {
            'bipartite_data': data,
            'metadata': {
                'source': 'demo3_bipartite_graph',
                'conversion_timestamp': datetime.now().isoformat(),
                'num_constraints': len(constraint_nodes),
                'num_variables': len(variable_nodes),
                'num_edges': len(edges),
                'device': str(self.device),
                'constraint_feature_dim': constraint_features.shape[1],
                'variable_feature_dim': variable_features.shape[1],
                'edge_feature_dim': edge_features.shape[1]
            },
            'extraction_summary': {
                'conversion_method': 'demo3_to_demo4_converter',
                'requires_grad': True,
                'bidirectional_edges': True,
                'timestamp': datetime.now().isoformat()
            }
        }
        
        self.logger.info("格式转换完成！")
        return result
    
    def _convert_constraint_features(self, constraint_nodes: List) -> np.ndarray:
        """转换约束节点特征为16维向量"""
        num_constraints = len(constraint_nodes)
        features = np.zeros((num_constraints, 16), dtype=np.float32)
        
        for i, node in enumerate(constraint_nodes):
            # 提取约束节点的特征（根据BipartiteGraph的结构）
            if hasattr(node, 'features') and node.features is not None:
                # 如果节点有预计算的特征
                node_features = np.array(node.features, dtype=np.float32)
                if len(node_features) >= 16:
                    features[i] = node_features[:16]
                else:
                    features[i, :len(node_features)] = node_features
            else:
                # 构建基本特征
                feature_idx = 0
                
                # 约束类型 (0-2)
                if hasattr(node, 'constraint_type'):
                    if node.constraint_type == 'equality':
                        features[i, feature_idx] = 1.0
                    elif node.constraint_type == 'inequality':
                        features[i, feature_idx + 1] = 1.0
                feature_idx += 3
                
                # 右端项值 (3)
                if hasattr(node, 'rhs') and node.rhs is not None:
                    features[i, feature_idx] = float(node.rhs)
                feature_idx += 1
                
                # 约束度数 (4)
                if hasattr(node, 'degree'):
                    features[i, feature_idx] = float(node.degree)
                elif hasattr(node, 'coefficient_count'):
                    features[i, feature_idx] = float(node.coefficient_count)
                feature_idx += 1
                
                # 系数统计 (5-10)
                if hasattr(node, 'coefficient_stats'):
                    stats = node.coefficient_stats
                    if isinstance(stats, dict):
                        features[i, feature_idx:feature_idx+6] = [
                            stats.get('mean', 0.0),
                            stats.get('std', 0.0),
                            stats.get('min', 0.0),
                            stats.get('max', 0.0),
                            stats.get('nnz', 0.0),
                            stats.get('sparsity', 0.0)
                        ]
                feature_idx += 6
                
                # 电力系统特定特征 (11-15)
                if hasattr(node, 'power_system_features'):
                    ps_features = node.power_system_features
                    if isinstance(ps_features, (list, np.ndarray)):
                        remaining = min(5, len(ps_features))
                        features[i, feature_idx:feature_idx+remaining] = ps_features[:remaining]
                
        self.logger.info(f"约束节点特征转换完成: {features.shape}")
        return features
    
    def _convert_variable_features(self, variable_nodes: List) -> np.ndarray:
        """转换变量节点特征为9维向量"""
        num_variables = len(variable_nodes)
        features = np.zeros((num_variables, 9), dtype=np.float32)
        
        for i, node in enumerate(variable_nodes):
            if hasattr(node, 'features') and node.features is not None:
                # 如果节点有预计算的特征
                node_features = np.array(node.features, dtype=np.float32)
                if len(node_features) >= 9:
                    features[i] = node_features[:9]
                else:
                    features[i, :len(node_features)] = node_features
            else:
                # 构建基本特征
                feature_idx = 0
                
                # 变量类型 (0-2)
                if hasattr(node, 'variable_type'):
                    if node.variable_type == 'continuous':
                        features[i, feature_idx] = 1.0
                    elif node.variable_type == 'integer':
                        features[i, feature_idx + 1] = 1.0
                    elif node.variable_type == 'binary':
                        features[i, feature_idx + 2] = 1.0
                feature_idx += 3
                
                # 目标函数系数 (3)
                if hasattr(node, 'objective_coeff'):
                    features[i, feature_idx] = float(node.objective_coeff)
                feature_idx += 1
                
                # 变量边界 (4-5)
                if hasattr(node, 'lower_bound'):
                    features[i, feature_idx] = float(node.lower_bound) if node.lower_bound is not None else -1e6
                if hasattr(node, 'upper_bound'):
                    features[i, feature_idx + 1] = float(node.upper_bound) if node.upper_bound is not None else 1e6
                feature_idx += 2
                
                # 变量度数 (6)
                if hasattr(node, 'degree'):
                    features[i, feature_idx] = float(node.degree)
                feature_idx += 1
                
                # 系数统计 (7-8)
                if hasattr(node, 'coefficient_stats'):
                    stats = node.coefficient_stats
                    if isinstance(stats, dict):
                        features[i, feature_idx:feature_idx+2] = [
                            stats.get('mean', 0.0),
                            stats.get('std', 0.0)
                        ]
                
        self.logger.info(f"变量节点特征转换完成: {features.shape}")
        return features
    
    def _convert_edges(self, edges: List, num_constraints: int, num_variables: int) -> Tuple[np.ndarray, np.ndarray]:
        """转换边连接和特征"""
        num_edges = len(edges)
        edge_indices = np.zeros((2, num_edges), dtype=np.int64)
        edge_features = np.zeros((num_edges, 8), dtype=np.float32)
        
        for i, edge in enumerate(edges):
            # 边连接 (约束索引, 变量索引)
            if hasattr(edge, 'constraint_idx') and hasattr(edge, 'variable_idx'):
                edge_indices[0, i] = edge.constraint_idx
                edge_indices[1, i] = edge.variable_idx
            elif hasattr(edge, 'source') and hasattr(edge, 'target'):
                edge_indices[0, i] = edge.source
                edge_indices[1, i] = edge.target
            elif isinstance(edge, (tuple, list)) and len(edge) >= 2:
                edge_indices[0, i] = edge[0]
                edge_indices[1, i] = edge[1]
            
            # 边特征
            if hasattr(edge, 'features') and edge.features is not None:
                edge_feat = np.array(edge.features, dtype=np.float32)
                if len(edge_feat) >= 8:
                    edge_features[i] = edge_feat[:8]
                else:
                    edge_features[i, :len(edge_feat)] = edge_feat
            else:
                # 构建基本边特征
                feature_idx = 0
                
                # 系数值 (0)
                if hasattr(edge, 'coefficient'):
                    edge_features[i, feature_idx] = float(edge.coefficient)
                elif hasattr(edge, 'weight'):
                    edge_features[i, feature_idx] = float(edge.weight)
                feature_idx += 1
                
                # 归一化系数 (1)
                if hasattr(edge, 'normalized_coeff'):
                    edge_features[i, feature_idx] = float(edge.normalized_coeff)
                feature_idx += 1
                
                # 排名特征 (2-4)
                if hasattr(edge, 'rank_features'):
                    rank_feat = edge.rank_features
                    if isinstance(rank_feat, (list, np.ndarray)) and len(rank_feat) >= 3:
                        edge_features[i, feature_idx:feature_idx+3] = rank_feat[:3]
                feature_idx += 3
                
                # 统计特征 (5-7)
                if hasattr(edge, 'stats'):
                    stats = edge.stats
                    if isinstance(stats, dict):
                        edge_features[i, feature_idx:feature_idx+3] = [
                            stats.get('abs_value', 0.0),
                            stats.get('log_abs', 0.0),
                            stats.get('sign', 0.0)
                        ]
        
        self.logger.info(f"边特征转换完成: {edge_indices.shape}, {edge_features.shape}")
        return edge_indices, edge_features
    
    def _validate_conversion(self, data: HeteroData, num_constraints: int, num_variables: int, num_edges: int):
        """验证转换结果的正确性"""
        self.logger.info("验证转换结果...")
        
        # 检查节点数量
        assert data['constraint'].x.size(0) == num_constraints, f"约束节点数量不匹配: {data['constraint'].x.size(0)} vs {num_constraints}"
        assert data['variable'].x.size(0) == num_variables, f"变量节点数量不匹配: {data['variable'].x.size(0)} vs {num_variables}"
        
        # 检查特征维度
        assert data['constraint'].x.size(1) == 16, f"约束节点特征维度错误: {data['constraint'].x.size(1)}"
        assert data['variable'].x.size(1) == 9, f"变量节点特征维度错误: {data['variable'].x.size(1)}"
        
        # 检查边数量
        forward_edges = data['constraint', 'connects', 'variable'].edge_index.size(1)
        backward_edges = data['variable', 'connected_by', 'constraint'].edge_index.size(1)
        assert forward_edges == num_edges, f"前向边数量不匹配: {forward_edges} vs {num_edges}"
        assert backward_edges == num_edges, f"反向边数量不匹配: {backward_edges} vs {num_edges}"
        
        # 检查边特征维度
        assert data['constraint', 'connects', 'variable'].edge_attr.size(1) == 8, "边特征维度错误"
        
        # 检查设备一致性
        constraint_device = data['constraint'].x.device
        variable_device = data['variable'].x.device
        self.logger.info(f"设备检查: 约束节点={constraint_device}, 变量节点={variable_device}, 目标={self.device}")
        
        if constraint_device != self.device:
            self.logger.warning(f"约束节点设备不匹配: {constraint_device} vs {self.device}")
        if variable_device != self.device:
            self.logger.warning(f"变量节点设备不匹配: {variable_device} vs {self.device}")
        
        # 检查梯度追踪
        assert data['constraint'].x.requires_grad, "约束节点未启用梯度追踪"
        assert data['variable'].x.requires_grad, "变量节点未启用梯度追踪"
        
        self.logger.info("✅ 转换结果验证通过")
        self.logger.info(f"转换统计:")
        self.logger.info(f"  - 约束节点: {data['constraint'].x.size()} (device: {data['constraint'].x.device})")
        self.logger.info(f"  - 变量节点: {data['variable'].x.size()} (device: {data['variable'].x.device})")
        self.logger.info(f"  - 前向边: {forward_edges}, 反向边: {backward_edges}")
        self.logger.info(f"  - 边特征: {data['constraint', 'connects', 'variable'].edge_attr.size()}")
        self.logger.info(f"  - 梯度追踪: ✅")


def test_converter():
    """测试转换器功能"""
    import pickle
    
    print("测试Demo 3到Demo 4格式转换器...")
    
    # 创建转换器
    converter = Demo3ToDemo4Converter()
    
    # 加载Demo 3数据
    demo3_file = "output/demo3_g2milp/bipartite_graphs/demo3_bipartite_graph.pkl"
    try:
        with open(demo3_file, 'rb') as f:
            bipartite_graph = pickle.load(f)
        
        print(f"加载Demo 3数据成功: {type(bipartite_graph)}")
        
        # 执行转换
        result = converter.convert_bipartite_graph(bipartite_graph)
        
        print("转换成功！")
        print(f"结果格式: {type(result['bipartite_data'])}")
        print(f"元数据: {result['metadata']}")
        
        return True
        
    except Exception as e:
        print(f"测试失败: {e}")
        return False


if __name__ == "__main__":
    test_converter()
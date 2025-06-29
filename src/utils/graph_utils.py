"""
图操作工具函数
提供二分图的常用操作和分析功能

功能模块:
1. 图拓扑分析
2. 特征提取和预处理
3. 图采样和子图提取
4. 图比较和相似性分析
5. 图变换和增强
"""

import numpy as np
import logging
from typing import Dict, List, Set, Tuple, Optional, Any, Union
from collections import defaultdict, deque
import random

logger = logging.getLogger(__name__)

try:
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.warning("SciPy不可用，部分图分析功能受限")

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    logger.warning("NetworkX不可用，图算法功能受限")


class BipartiteGraphAnalyzer:
    """二分图分析器"""
    
    @staticmethod
    def compute_graph_properties(graph) -> Dict[str, Any]:
        """
        计算图的拓扑性质
        
        Args:
            graph: BipartiteGraph对象
            
        Returns:
            图性质字典
        """
        properties = {}
        
        try:
            # 基本统计
            n_var = len(graph.variable_nodes)
            n_cons = len(graph.constraint_nodes)
            n_edges = len(graph.edges)
            
            properties['basic_stats'] = {
                'n_variable_nodes': n_var,
                'n_constraint_nodes': n_cons,
                'n_edges': n_edges,
                'total_nodes': n_var + n_cons,
                'density': graph.statistics.density if hasattr(graph, 'statistics') else n_edges / (n_var * n_cons) if n_var * n_cons > 0 else 0
            }
            
            # 度数分析
            var_degrees = [node.degree for node in graph.variable_nodes.values()]
            cons_degrees = [node.degree for node in graph.constraint_nodes.values()]
            
            properties['degree_analysis'] = {
                'variable_degree_stats': {
                    'mean': np.mean(var_degrees) if var_degrees else 0,
                    'std': np.std(var_degrees) if var_degrees else 0,
                    'min': np.min(var_degrees) if var_degrees else 0,
                    'max': np.max(var_degrees) if var_degrees else 0,
                    'distribution': np.histogram(var_degrees, bins=10)[0].tolist() if var_degrees else []
                },
                'constraint_degree_stats': {
                    'mean': np.mean(cons_degrees) if cons_degrees else 0,
                    'std': np.std(cons_degrees) if cons_degrees else 0,
                    'min': np.min(cons_degrees) if cons_degrees else 0,
                    'max': np.max(cons_degrees) if cons_degrees else 0,
                    'distribution': np.histogram(cons_degrees, bins=10)[0].tolist() if cons_degrees else []
                }
            }
            
            # 连通性分析
            if SCIPY_AVAILABLE:
                adj_matrix = graph.get_adjacency_matrix(sparse=True)
                if adj_matrix.nnz > 0:
                    # 创建无向图的邻接矩阵进行连通性分析
                    n_total = n_var + n_cons
                    full_adj = sp.lil_matrix((n_total, n_total))
                    
                    # 填充上三角部分（变量到约束）
                    full_adj[:n_var, n_var:] = adj_matrix
                    # 填充下三角部分（约束到变量）
                    full_adj[n_var:, :n_var] = adj_matrix.T
                    
                    n_components, labels = connected_components(full_adj.tocsr())
                    
                    properties['connectivity'] = {
                        'n_connected_components': int(n_components),
                        'is_connected': n_components == 1,
                        'component_sizes': np.bincount(labels).tolist()
                    }
            
            # 系数分析
            coefficients = [edge.coefficient for edge in graph.edges.values()]
            if coefficients:
                properties['coefficient_analysis'] = {
                    'mean': float(np.mean(coefficients)),
                    'std': float(np.std(coefficients)),
                    'min': float(np.min(coefficients)),
                    'max': float(np.max(coefficients)),
                    'nnz_ratio': float(np.count_nonzero(coefficients) / len(coefficients)),
                    'positive_ratio': float(np.sum(np.array(coefficients) > 0) / len(coefficients)),
                    'negative_ratio': float(np.sum(np.array(coefficients) < 0) / len(coefficients))
                }
            
            logger.debug(f"图性质计算完成: {len(properties)} 个类别")
            
        except Exception as e:
            logger.error(f"图性质计算失败: {e}")
            properties['error'] = str(e)
        
        return properties
    
    @staticmethod
    def find_high_degree_nodes(graph, percentile: float = 95) -> Dict[str, List[str]]:
        """
        找出高度数节点
        
        Args:
            graph: BipartiteGraph对象
            percentile: 百分位数阈值
            
        Returns:
            高度数节点字典
        """
        try:
            var_degrees = [(node.node_id, node.degree) for node in graph.variable_nodes.values()]
            cons_degrees = [(node.node_id, node.degree) for node in graph.constraint_nodes.values()]
            
            # 计算阈值
            var_threshold = np.percentile([d for _, d in var_degrees], percentile) if var_degrees else 0
            cons_threshold = np.percentile([d for _, d in cons_degrees], percentile) if cons_degrees else 0
            
            high_degree_nodes = {
                'variable_nodes': [node_id for node_id, degree in var_degrees if degree >= var_threshold],
                'constraint_nodes': [node_id for node_id, degree in cons_degrees if degree >= cons_threshold],
                'thresholds': {
                    'variable_threshold': float(var_threshold),
                    'constraint_threshold': float(cons_threshold)
                }
            }
            
            return high_degree_nodes
            
        except Exception as e:
            logger.error(f"高度数节点查找失败: {e}")
            return {'error': str(e)}
    
    @staticmethod
    def compute_centrality_measures(graph) -> Dict[str, Dict[str, float]]:
        """
        计算节点中心性指标
        
        Args:
            graph: BipartiteGraph对象
            
        Returns:
            中心性指标字典
        """
        if not NETWORKX_AVAILABLE:
            logger.warning("NetworkX不可用，无法计算中心性指标")
            return {}
        
        try:
            from ..models.bipartite_graph.serializer import BipartiteGraphSerializer
            
            # 转换为NetworkX图
            serializer = BipartiteGraphSerializer()
            G = serializer.to_networkx(graph)
            
            if G is None:
                return {'error': '无法转换为NetworkX图'}
            
            centrality_measures = {}
            
            # 度中心性（已知）
            degree_centrality = nx.degree_centrality(G)
            centrality_measures['degree_centrality'] = degree_centrality
            
            # 接近中心性
            try:
                closeness_centrality = nx.closeness_centrality(G)
                centrality_measures['closeness_centrality'] = closeness_centrality
            except:
                logger.warning("接近中心性计算失败（可能由于图不连通）")
            
            # 介数中心性（对大图可能很慢）
            if len(G.nodes()) <= 1000:
                try:
                    betweenness_centrality = nx.betweenness_centrality(G)
                    centrality_measures['betweenness_centrality'] = betweenness_centrality
                except:
                    logger.warning("介数中心性计算失败")
            
            # 特征向量中心性
            try:
                eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000)
                centrality_measures['eigenvector_centrality'] = eigenvector_centrality
            except:
                logger.warning("特征向量中心性计算失败")
            
            return centrality_measures
            
        except Exception as e:
            logger.error(f"中心性指标计算失败: {e}")
            return {'error': str(e)}


class BipartiteGraphSampler:
    """二分图采样器"""
    
    @staticmethod
    def random_node_sample(graph, 
                          sample_ratio: float = 0.5,
                          preserve_connectivity: bool = True,
                          random_seed: int = None) -> 'BipartiteGraph':
        """
        随机节点采样
        
        Args:
            graph: 原始二分图
            sample_ratio: 采样比例
            preserve_connectivity: 是否保持连通性
            random_seed: 随机种子
            
        Returns:
            采样后的子图
        """
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
        
        try:
            from ..models.bipartite_graph.data_structures import BipartiteGraph
            
            # 计算采样数量
            n_var_sample = max(1, int(len(graph.variable_nodes) * sample_ratio))
            n_cons_sample = max(1, int(len(graph.constraint_nodes) * sample_ratio))
            
            # 随机选择节点
            var_nodes = list(graph.variable_nodes.keys())
            cons_nodes = list(graph.constraint_nodes.keys())
            
            sampled_var_nodes = random.sample(var_nodes, n_var_sample)
            sampled_cons_nodes = random.sample(cons_nodes, n_cons_sample)
            
            # 创建子图
            subgraph = BipartiteGraphSampler._create_subgraph(
                graph, sampled_var_nodes, sampled_cons_nodes,
                f"{graph.graph_id}_sampled_{sample_ratio}"
            )
            
            return subgraph
            
        except Exception as e:
            logger.error(f"随机节点采样失败: {e}")
            return None
    
    @staticmethod
    def degree_based_sample(graph,
                           sample_ratio: float = 0.5,
                           prefer_high_degree: bool = True,
                           random_seed: int = None) -> 'BipartiteGraph':
        """
        基于度数的采样
        
        Args:
            graph: 原始二分图
            sample_ratio: 采样比例
            prefer_high_degree: 是否优先选择高度数节点
            random_seed: 随机种子
            
        Returns:
            采样后的子图
        """
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
        
        try:
            # 按度数排序节点
            var_nodes_by_degree = sorted(graph.variable_nodes.items(), 
                                       key=lambda x: x[1].degree, 
                                       reverse=prefer_high_degree)
            cons_nodes_by_degree = sorted(graph.constraint_nodes.items(),
                                        key=lambda x: x[1].degree,
                                        reverse=prefer_high_degree)
            
            # 计算采样数量
            n_var_sample = max(1, int(len(graph.variable_nodes) * sample_ratio))
            n_cons_sample = max(1, int(len(graph.constraint_nodes) * sample_ratio))
            
            # 选择节点
            sampled_var_nodes = [node_id for node_id, _ in var_nodes_by_degree[:n_var_sample]]
            sampled_cons_nodes = [node_id for node_id, _ in cons_nodes_by_degree[:n_cons_sample]]
            
            # 创建子图
            subgraph = BipartiteGraphSampler._create_subgraph(
                graph, sampled_var_nodes, sampled_cons_nodes,
                f"{graph.graph_id}_degree_sampled_{sample_ratio}"
            )
            
            return subgraph
            
        except Exception as e:
            logger.error(f"度数采样失败: {e}")
            return None
    
    @staticmethod
    def neighborhood_sample(graph,
                          seed_nodes: List[str],
                          max_hops: int = 2,
                          max_nodes: int = 1000) -> 'BipartiteGraph':
        """
        邻域采样
        
        Args:
            graph: 原始二分图
            seed_nodes: 种子节点列表
            max_hops: 最大跳数
            max_nodes: 最大节点数
            
        Returns:
            采样后的子图
        """
        try:
            sampled_nodes = set(seed_nodes)
            current_layer = set(seed_nodes)
            
            for hop in range(max_hops):
                if len(sampled_nodes) >= max_nodes:
                    break
                
                next_layer = set()
                
                for node_id in current_layer:
                    # 获取邻居节点
                    if node_id in graph.variable_nodes:
                        neighbors = graph.variable_to_constraints.get(node_id, set())
                    elif node_id in graph.constraint_nodes:
                        neighbors = graph.constraint_to_variables.get(node_id, set())
                    else:
                        continue
                    
                    next_layer.update(neighbors)
                
                # 添加新节点，但限制总数
                remaining_capacity = max_nodes - len(sampled_nodes)
                if remaining_capacity > 0:
                    next_layer = next_layer - sampled_nodes
                    if len(next_layer) > remaining_capacity:
                        next_layer = set(random.sample(list(next_layer), remaining_capacity))
                    
                    sampled_nodes.update(next_layer)
                    current_layer = next_layer
                else:
                    break
            
            # 分离变量节点和约束节点
            sampled_var_nodes = [nid for nid in sampled_nodes if nid in graph.variable_nodes]
            sampled_cons_nodes = [nid for nid in sampled_nodes if nid in graph.constraint_nodes]
            
            # 创建子图
            subgraph = BipartiteGraphSampler._create_subgraph(
                graph, sampled_var_nodes, sampled_cons_nodes,
                f"{graph.graph_id}_neighborhood_{max_hops}hops"
            )
            
            return subgraph
            
        except Exception as e:
            logger.error(f"邻域采样失败: {e}")
            return None
    
    @staticmethod
    def _create_subgraph(original_graph,
                        var_node_ids: List[str],
                        cons_node_ids: List[str],
                        subgraph_id: str) -> 'BipartiteGraph':
        """
        创建子图
        
        Args:
            original_graph: 原始图
            var_node_ids: 变量节点ID列表
            cons_node_ids: 约束节点ID列表
            subgraph_id: 子图ID
            
        Returns:
            子图对象
        """
        from ..models.bipartite_graph.data_structures import BipartiteGraph
        
        # 创建新图
        subgraph = BipartiteGraph(
            graph_id=subgraph_id,
            source_problem_id=f"subgraph_of_{original_graph.source_problem_id}"
        )
        
        # 复制选中的变量节点
        for var_id in var_node_ids:
            if var_id in original_graph.variable_nodes:
                var_node = original_graph.variable_nodes[var_id]
                # 创建副本
                import copy
                new_var_node = copy.deepcopy(var_node)
                new_var_node.degree = 0  # 重新计算
                subgraph.add_variable_node(new_var_node)
        
        # 复制选中的约束节点
        for cons_id in cons_node_ids:
            if cons_id in original_graph.constraint_nodes:
                cons_node = original_graph.constraint_nodes[cons_id]
                # 创建副本
                import copy
                new_cons_node = copy.deepcopy(cons_node)
                new_cons_node.degree = 0  # 重新计算
                new_cons_node.lhs_coefficients = {}  # 重新填充
                subgraph.add_constraint_node(new_cons_node)
        
        # 复制相关的边
        for edge in original_graph.edges.values():
            if (edge.variable_node_id in var_node_ids and 
                edge.constraint_node_id in cons_node_ids):
                
                # 创建边的副本
                import copy
                new_edge = copy.deepcopy(edge)
                subgraph.add_edge(new_edge)
                
                # 更新约束节点的系数信息
                cons_node = subgraph.constraint_nodes[edge.constraint_node_id]
                cons_node.lhs_coefficients[edge.variable_node_id] = edge.coefficient
        
        # 更新度数和统计
        subgraph.update_node_degrees()
        subgraph.compute_statistics()
        
        # 添加子图元信息
        subgraph.metadata.update({
            'is_subgraph': True,
            'original_graph_id': original_graph.graph_id,
            'sampling_info': {
                'original_var_nodes': len(original_graph.variable_nodes),
                'original_cons_nodes': len(original_graph.constraint_nodes),
                'original_edges': len(original_graph.edges),
                'sampled_var_nodes': len(var_node_ids),
                'sampled_cons_nodes': len(cons_node_ids),
                'sampled_edges': len(subgraph.edges)
            }
        })
        
        return subgraph


class BipartiteGraphComparator:
    """二分图比较器"""
    
    @staticmethod
    def compute_graph_similarity(graph1, graph2) -> Dict[str, float]:
        """
        计算两个图的相似性
        
        Args:
            graph1: 第一个图
            graph2: 第二个图
            
        Returns:
            相似性指标字典
        """
        try:
            similarity = {}
            
            # 结构相似性
            similarity['structure_similarity'] = BipartiteGraphComparator._compute_structure_similarity(graph1, graph2)
            
            # 度数分布相似性
            similarity['degree_similarity'] = BipartiteGraphComparator._compute_degree_similarity(graph1, graph2)
            
            # 系数分布相似性
            similarity['coefficient_similarity'] = BipartiteGraphComparator._compute_coefficient_similarity(graph1, graph2)
            
            # 特征相似性
            similarity['feature_similarity'] = BipartiteGraphComparator._compute_feature_similarity(graph1, graph2)
            
            # 综合相似性
            similarity['overall_similarity'] = np.mean([
                similarity['structure_similarity'],
                similarity['degree_similarity'],
                similarity['coefficient_similarity'],
                similarity['feature_similarity']
            ])
            
            return similarity
            
        except Exception as e:
            logger.error(f"图相似性计算失败: {e}")
            return {'error': str(e)}
    
    @staticmethod
    def _compute_structure_similarity(graph1, graph2) -> float:
        """计算结构相似性"""
        try:
            # 比较基本结构统计
            stats1 = {
                'n_var': len(graph1.variable_nodes),
                'n_cons': len(graph1.constraint_nodes),
                'n_edges': len(graph1.edges),
                'density': graph1.statistics.density if hasattr(graph1, 'statistics') else 0
            }
            
            stats2 = {
                'n_var': len(graph2.variable_nodes),
                'n_cons': len(graph2.constraint_nodes),
                'n_edges': len(graph2.edges),
                'density': graph2.statistics.density if hasattr(graph2, 'statistics') else 0
            }
            
            # 计算相对差异
            similarities = []
            for key in stats1:
                val1, val2 = stats1[key], stats2[key]
                if val1 == 0 and val2 == 0:
                    similarities.append(1.0)
                elif val1 == 0 or val2 == 0:
                    similarities.append(0.0)
                else:
                    similarity = 1.0 - abs(val1 - val2) / max(val1, val2)
                    similarities.append(max(0, similarity))
            
            return np.mean(similarities)
            
        except Exception as e:
            logger.warning(f"结构相似性计算失败: {e}")
            return 0.0
    
    @staticmethod
    def _compute_degree_similarity(graph1, graph2) -> float:
        """计算度数分布相似性"""
        try:
            # 变量节点度数分布
            var_degrees1 = [node.degree for node in graph1.variable_nodes.values()]
            var_degrees2 = [node.degree for node in graph2.variable_nodes.values()]
            
            # 约束节点度数分布
            cons_degrees1 = [node.degree for node in graph1.constraint_nodes.values()]
            cons_degrees2 = [node.degree for node in graph2.constraint_nodes.values()]
            
            similarities = []
            
            # 比较变量度数分布
            if var_degrees1 and var_degrees2:
                var_sim = BipartiteGraphComparator._distribution_similarity(var_degrees1, var_degrees2)
                similarities.append(var_sim)
            
            # 比较约束度数分布
            if cons_degrees1 and cons_degrees2:
                cons_sim = BipartiteGraphComparator._distribution_similarity(cons_degrees1, cons_degrees2)
                similarities.append(cons_sim)
            
            return np.mean(similarities) if similarities else 0.0
            
        except Exception as e:
            logger.warning(f"度数相似性计算失败: {e}")
            return 0.0
    
    @staticmethod
    def _compute_coefficient_similarity(graph1, graph2) -> float:
        """计算系数分布相似性"""
        try:
            coeffs1 = [edge.coefficient for edge in graph1.edges.values()]
            coeffs2 = [edge.coefficient for edge in graph2.edges.values()]
            
            if not coeffs1 or not coeffs2:
                return 0.0
            
            return BipartiteGraphComparator._distribution_similarity(coeffs1, coeffs2)
            
        except Exception as e:
            logger.warning(f"系数相似性计算失败: {e}")
            return 0.0
    
    @staticmethod
    def _compute_feature_similarity(graph1, graph2) -> float:
        """计算特征相似性"""
        try:
            # 比较变量节点特征的统计量
            features1 = []
            features2 = []
            
            for node in graph1.variable_nodes.values():
                try:
                    features1.append(node.get_feature_vector())
                except:
                    continue
            
            for node in graph2.variable_nodes.values():
                try:
                    features2.append(node.get_feature_vector())
                except:
                    continue
            
            if not features1 or not features2:
                return 0.0
            
            # 计算特征均值的相似性
            mean_features1 = np.mean(features1, axis=0)
            mean_features2 = np.mean(features2, axis=0)
            
            # 计算余弦相似性
            dot_product = np.dot(mean_features1, mean_features2)
            norm1 = np.linalg.norm(mean_features1)
            norm2 = np.linalg.norm(mean_features2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            cosine_similarity = dot_product / (norm1 * norm2)
            return max(0, cosine_similarity)
            
        except Exception as e:
            logger.warning(f"特征相似性计算失败: {e}")
            return 0.0
    
    @staticmethod
    def _distribution_similarity(dist1: List[float], dist2: List[float]) -> float:
        """计算两个分布的相似性"""
        try:
            # 使用Wasserstein距离（或其近似）
            from scipy import stats
            
            # 计算统计量
            stat1 = {
                'mean': np.mean(dist1),
                'std': np.std(dist1),
                'min': np.min(dist1),
                'max': np.max(dist1)
            }
            
            stat2 = {
                'mean': np.mean(dist2),
                'std': np.std(dist2),
                'min': np.min(dist2),
                'max': np.max(dist2)
            }
            
            # 计算KS统计量
            ks_stat, _ = stats.ks_2samp(dist1, dist2)
            ks_similarity = 1.0 - ks_stat
            
            return max(0, ks_similarity)
            
        except:
            # 回退到简单的统计量比较
            try:
                mean_diff = abs(np.mean(dist1) - np.mean(dist2))
                max_mean = max(np.mean(dist1), np.mean(dist2))
                
                if max_mean == 0:
                    return 1.0
                
                return max(0, 1.0 - mean_diff / max_mean)
                
            except:
                return 0.0


# 便捷函数
def analyze_bipartite_graph(graph) -> Dict[str, Any]:
    """
    分析二分图（便捷函数）
    
    Args:
        graph: BipartiteGraph对象
        
    Returns:
        分析结果字典
    """
    analyzer = BipartiteGraphAnalyzer()
    
    analysis = {}
    analysis['graph_properties'] = analyzer.compute_graph_properties(graph)
    analysis['high_degree_nodes'] = analyzer.find_high_degree_nodes(graph)
    analysis['centrality_measures'] = analyzer.compute_centrality_measures(graph)
    
    return analysis


def sample_bipartite_graph(graph,
                          method: str = 'random',
                          sample_ratio: float = 0.5,
                          **kwargs) -> 'BipartiteGraph':
    """
    采样二分图（便捷函数）
    
    Args:
        graph: 原始图
        method: 采样方法 ('random', 'degree', 'neighborhood')
        sample_ratio: 采样比例
        **kwargs: 其他参数
        
    Returns:
        采样后的子图
    """
    sampler = BipartiteGraphSampler()
    
    if method == 'random':
        return sampler.random_node_sample(graph, sample_ratio, **kwargs)
    elif method == 'degree':
        return sampler.degree_based_sample(graph, sample_ratio, **kwargs)
    elif method == 'neighborhood':
        seed_nodes = kwargs.get('seed_nodes', [])
        if not seed_nodes:
            # 随机选择种子节点
            all_nodes = list(graph.variable_nodes.keys()) + list(graph.constraint_nodes.keys())
            seed_nodes = random.sample(all_nodes, min(5, len(all_nodes)))
        return sampler.neighborhood_sample(graph, seed_nodes, **kwargs)
    else:
        logger.error(f"不支持的采样方法: {method}")
        return None


def compare_bipartite_graphs(graph1, graph2) -> Dict[str, float]:
    """
    比较两个二分图（便捷函数）
    
    Args:
        graph1: 第一个图
        graph2: 第二个图
        
    Returns:
        相似性指标字典
    """
    comparator = BipartiteGraphComparator()
    return comparator.compute_graph_similarity(graph1, graph2)


if __name__ == "__main__":
    """测试图工具函数"""
    logger.info("图操作工具函数测试")
    print("✅ 图工具模块加载成功!")
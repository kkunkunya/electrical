"""
G2MILP评估模块
G2MILP Evaluation Module

实现多维度的模型和生成质量评估，包括：
1. 图结构相似度评估
2. MILP特征相似度评估  
3. 生成多样性评估
4. 训练质量监控
5. 性能基准对比
"""

import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
import json
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import networkx as nx
from scipy.stats import wasserstein_distance, ks_2samp
import warnings

logger = logging.getLogger(__name__)


@dataclass
class EvaluationConfig:
    """评估配置"""
    # 基础评估参数
    enable_graph_similarity: bool = True      # 图结构相似度评估
    enable_milp_similarity: bool = True       # MILP特征相似度评估
    enable_diversity_analysis: bool = True    # 多样性分析
    enable_training_monitoring: bool = True   # 训练监控
    
    # 相似度计算参数
    similarity_metrics: List[str] = None      # 相似度指标列表
    graph_topology_weight: float = 0.4       # 图拓扑权重
    node_feature_weight: float = 0.3         # 节点特征权重
    edge_feature_weight: float = 0.3         # 边特征权重
    
    # 多样性评估参数
    diversity_sample_size: int = 10           # 多样性评估样本数
    diversity_metrics: List[str] = None       # 多样性指标列表
    
    # 基准对比参数
    baseline_threshold: float = 0.7          # 基准阈值
    quality_benchmarks: Dict[str, float] = None  # 质量基准
    
    # 输出配置
    save_detailed_results: bool = True       # 保存详细结果
    generate_visualizations: bool = True     # 生成可视化
    output_dir: str = "output/demo4_g2milp/evaluation"
    
    def __post_init__(self):
        """初始化默认值"""
        if self.similarity_metrics is None:
            self.similarity_metrics = [
                'degree_distribution', 'clustering_coefficient', 
                'node_feature_similarity', 'edge_weight_similarity',
                'sparsity_similarity'
            ]
        
        if self.diversity_metrics is None:
            self.diversity_metrics = [
                'bias_diversity', 'degree_diversity', 'connection_diversity',
                'structural_diversity', 'feature_diversity'
            ]
        
        if self.quality_benchmarks is None:
            self.quality_benchmarks = {
                'similarity_score': 0.6,      # 相似度分数基准
                'diversity_score': 0.3,       # 多样性分数基准  
                'sparsity_preservation': 0.8,  # 稀疏性保持基准
                'training_convergence': 0.1    # 训练收敛基准
            }


class G2MILPEvaluator:
    """
    G2MILP综合评估器
    
    提供全面的模型性能、生成质量和训练过程评估
    """
    
    def __init__(self, config: EvaluationConfig = None):
        self.config = config or EvaluationConfig()
        self.evaluation_history = []
        
        # 创建输出目录
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"G2MILP评估器初始化完成 - 输出目录: {self.output_dir}")
    
    def evaluate_online_quality(self, 
                               model,
                               original_data: HeteroData,
                               epoch: int = 0,
                               num_samples: int = 3) -> Dict[str, float]:
        """
        在线质量评估 - 在训练过程中实时评估模型质量
        
        轻量级评估，不影响训练性能，主要评估：
        1. 约束有效性 - 生成约束的语法正确性
        2. 生成多样性 - 生成实例间的差异性
        3. 统计相似性 - 与原始图的宏观统计对比
        
        Args:
            model: G2MILP生成器模型
            original_data: 原始图数据
            epoch: 当前训练轮次
            num_samples: 评估样本数（保持较小以节省时间）
            
        Returns:
            质量评估指标字典
        """
        try:
            # 早期训练阶段跳过评估（模型还未充分学习）
            if epoch < 10:
                logger.debug(f"跳过早期训练阶段的质量评估 (epoch {epoch} < 10)")
                return {
                    'validity_score': 0.0, 
                    'diversity_score': 0.0, 
                    'similarity_score': 0.0,
                    'stability_score': 0.0,
                    'overall_quality': 0.0
                }
            
            # 设置模型为评估模式
            model.eval()
            quality_metrics = {}
            
            # 快速生成少量样本进行评估
            with torch.no_grad():
                generated_samples = []
                generation_infos = []
                
                for i in range(num_samples):
                    try:
                        # 使用模型的推理模式生成实例
                        result = model(original_data, mode="inference")
                        if isinstance(result, dict) and 'generated_data' in result:
                            generated_data = result['generated_data']
                            # 基础有效性检查
                            if isinstance(generated_data, HeteroData) and len(generated_data.x_dict) > 0:
                                generated_samples.append(generated_data)
                                generation_infos.append(result.get('generation_history', []))
                                logger.debug(f"在线评估样本{i+1}生成成功")
                            else:
                                logger.debug(f"在线评估样本{i+1}数据无效: 空数据或格式错误")
                        else:
                            logger.debug(f"在线评估样本{i+1}结果格式错误: {type(result)}")
                    except Exception as e:
                        logger.debug(f"在线评估样本{i+1}生成异常: {e}")
                        continue
                
                if not generated_samples:
                    logger.warning(f"在线评估：无法生成有效样本 (epoch {epoch}, 尝试样本数: {num_samples})")
                    return {
                        'validity_score': 0.0, 
                        'diversity_score': 0.0, 
                        'similarity_score': 0.0,
                        'stability_score': 0.0,
                        'overall_quality': 0.0
                    }
                
                # 1. 约束有效性评估
                validity_score = self._evaluate_constraint_validity(generated_samples)
                quality_metrics['validity_score'] = validity_score
                
                # 2. 生成多样性评估（轻量级）
                diversity_score = self._evaluate_lightweight_diversity(generated_samples, generation_infos)
                quality_metrics['diversity_score'] = diversity_score
                
                # 3. 统计相似性评估
                similarity_score = self._evaluate_statistical_similarity(original_data, generated_samples)
                quality_metrics['similarity_score'] = similarity_score
                
                # 4. 数值稳定性检查
                stability_score = self._evaluate_numerical_stability(generated_samples)
                quality_metrics['stability_score'] = stability_score
                
                # 5. 综合质量得分
                overall_quality = (validity_score * 0.3 + diversity_score * 0.2 + 
                                 similarity_score * 0.3 + stability_score * 0.2)
                quality_metrics['overall_quality'] = overall_quality
                
                # 记录评估历史
                self.evaluation_history.append({
                    'epoch': epoch,
                    'timestamp': datetime.now().isoformat(),
                    'metrics': quality_metrics,
                    'sample_count': len(generated_samples)
                })
                
            return quality_metrics
            
        except Exception as e:
            logger.error(f"在线质量评估失败: {e}")
            return {'validity_score': 0.0, 'diversity_score': 0.0, 'similarity_score': 0.0, 'overall_quality': 0.0}
        finally:
            # 恢复模型训练模式
            model.train()
    
    def _evaluate_constraint_validity(self, generated_samples: List[HeteroData]) -> float:
        """
        评估约束有效性
        
        检查生成的约束是否符合基本语法和逻辑规则
        """
        try:
            valid_count = 0
            total_count = 0
            
            for sample in generated_samples:
                # 检查基本图结构
                if 'constraint' in sample and 'variable' in sample:
                    constraint_features = sample['constraint'].x
                    variable_features = sample['variable'].x
                    
                    # 基本有效性检查
                    is_valid = True
                    
                    # 1. 特征值范围检查
                    if torch.isnan(constraint_features).any() or torch.isinf(constraint_features).any():
                        is_valid = False
                    
                    if torch.isnan(variable_features).any() or torch.isinf(variable_features).any():
                        is_valid = False
                    
                    # 2. 维度一致性检查
                    if constraint_features.size(1) != 16 or variable_features.size(1) != 9:
                        is_valid = False
                    
                    # 3. 边连接有效性检查
                    if hasattr(sample, 'edge_index_dict'):
                        for edge_type, edge_index in sample.edge_index_dict.items():
                            if edge_index.size(0) != 2:
                                is_valid = False
                            # 检查节点索引是否越界
                            if edge_type[0] == 'constraint':
                                if edge_index[0].max() >= constraint_features.size(0):
                                    is_valid = False
                            elif edge_type[2] == 'variable':
                                if edge_index[1].max() >= variable_features.size(0):
                                    is_valid = False
                    
                    if is_valid:
                        valid_count += 1
                    total_count += 1
            
            return valid_count / max(total_count, 1)
            
        except Exception as e:
            logger.warning(f"约束有效性评估失败: {e}")
            return 0.0
    
    def _evaluate_lightweight_diversity(self, generated_samples: List[HeteroData], 
                                      generation_infos: List[Dict]) -> float:
        """
        轻量级多样性评估
        
        快速计算生成样本间的多样性指标
        """
        try:
            if len(generated_samples) < 2:
                return 0.0
            
            diversity_scores = []
            
            # 1. 节点数多样性
            node_counts = []
            for sample in generated_samples:
                constraint_count = sample['constraint'].x.size(0) if 'constraint' in sample else 0
                variable_count = sample['variable'].x.size(0) if 'variable' in sample else 0
                node_counts.append(constraint_count + variable_count)
            
            if len(set(node_counts)) > 1:
                node_diversity = np.std(node_counts) / (np.mean(node_counts) + 1e-8)
                diversity_scores.append(min(node_diversity, 1.0))
            
            # 2. 边数多样性
            edge_counts = []
            for sample in generated_samples:
                total_edges = 0
                if hasattr(sample, 'edge_index_dict'):
                    for edge_index in sample.edge_index_dict.values():
                        total_edges += edge_index.size(1)
                edge_counts.append(total_edges)
            
            if len(set(edge_counts)) > 1:
                edge_diversity = np.std(edge_counts) / (np.mean(edge_counts) + 1e-8)
                diversity_scores.append(min(edge_diversity, 1.0))
            
            # 3. 特征多样性（约束特征的第一个维度）
            if len(generated_samples) >= 2:
                feature_diversity = []
                for i in range(len(generated_samples)):
                    for j in range(i+1, len(generated_samples)):
                        if 'constraint' in generated_samples[i] and 'constraint' in generated_samples[j]:
                            feat_i = generated_samples[i]['constraint'].x.mean(dim=0)
                            feat_j = generated_samples[j]['constraint'].x.mean(dim=0)
                            # 计算特征向量的余弦距离
                            cosine_sim = F.cosine_similarity(feat_i.unsqueeze(0), feat_j.unsqueeze(0))
                            feature_diversity.append(1.0 - cosine_sim.item())
                
                if feature_diversity:
                    diversity_scores.append(np.mean(feature_diversity))
            
            return np.mean(diversity_scores) if diversity_scores else 0.0
            
        except Exception as e:
            logger.warning(f"轻量级多样性评估失败: {e}")
            return 0.0
    
    def _evaluate_statistical_similarity(self, original_data: HeteroData, 
                                       generated_samples: List[HeteroData]) -> float:
        """
        统计相似性评估
        
        比较生成图与原始图在宏观统计特征上的相似性
        """
        try:
            # 获取原始图统计信息
            orig_constraint_count = original_data['constraint'].x.size(0)
            orig_variable_count = original_data['variable'].x.size(0)
            
            orig_edge_count = 0
            if hasattr(original_data, 'edge_index_dict'):
                for edge_index in original_data.edge_index_dict.values():
                    orig_edge_count += edge_index.size(1)
            
            # 计算生成样本的统计信息
            similarity_scores = []
            
            for sample in generated_samples:
                gen_constraint_count = sample['constraint'].x.size(0) if 'constraint' in sample else 0
                gen_variable_count = sample['variable'].x.size(0) if 'variable' in sample else 0
                
                gen_edge_count = 0
                if hasattr(sample, 'edge_index_dict'):
                    for edge_index in sample.edge_index_dict.values():
                        gen_edge_count += edge_index.size(1)
                
                # 计算相似度（基于相对差异）
                constraint_sim = 1.0 - abs(gen_constraint_count - orig_constraint_count) / max(orig_constraint_count, 1)
                variable_sim = 1.0 - abs(gen_variable_count - orig_variable_count) / max(orig_variable_count, 1)
                edge_sim = 1.0 - abs(gen_edge_count - orig_edge_count) / max(orig_edge_count, 1)
                
                sample_similarity = np.mean([constraint_sim, variable_sim, edge_sim])
                similarity_scores.append(max(0.0, sample_similarity))
            
            return np.mean(similarity_scores) if similarity_scores else 0.0
            
        except Exception as e:
            logger.warning(f"统计相似性评估失败: {e}")
            return 0.0
    
    def _evaluate_numerical_stability(self, generated_samples: List[HeteroData]) -> float:
        """
        数值稳定性评估
        
        检查生成的数值是否稳定（无NaN/Inf等异常值）
        """
        try:
            stable_count = 0
            total_count = 0
            
            for sample in generated_samples:
                is_stable = True
                
                # 检查约束节点特征
                if 'constraint' in sample:
                    constraint_features = sample['constraint'].x
                    if torch.isnan(constraint_features).any() or torch.isinf(constraint_features).any():
                        is_stable = False
                    if (constraint_features.abs() > 1000).any():  # 检查异常大的值
                        is_stable = False
                
                # 检查变量节点特征
                if 'variable' in sample:
                    variable_features = sample['variable'].x
                    if torch.isnan(variable_features).any() or torch.isinf(variable_features).any():
                        is_stable = False
                    if (variable_features.abs() > 1000).any():
                        is_stable = False
                
                # 检查边特征（如果存在）
                if hasattr(sample, 'edge_attr_dict'):
                    for edge_attr in sample.edge_attr_dict.values():
                        if edge_attr is not None:
                            if torch.isnan(edge_attr).any() or torch.isinf(edge_attr).any():
                                is_stable = False
                            if (edge_attr.abs() > 1000).any():
                                is_stable = False
                
                if is_stable:
                    stable_count += 1
                total_count += 1
            
            return stable_count / max(total_count, 1)
            
        except Exception as e:
            logger.warning(f"数值稳定性评估失败: {e}")
            return 0.0
    
    def evaluate_generation_quality(self, 
                                   original_data: HeteroData,
                                   generated_data_list: List[HeteroData],
                                   generation_info: List[Dict] = None) -> Dict[str, Any]:
        """
        评估生成质量
        
        Args:
            original_data: 原始图数据
            generated_data_list: 生成的图数据列表
            generation_info: 生成过程信息列表
            
        Returns:
            评估结果字典
        """
        logger.info(f"开始评估生成质量 - 样本数量: {len(generated_data_list)}")
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'sample_count': len(generated_data_list),
            'evaluation_config': asdict(self.config)
        }
        
        # 1. 图结构相似度评估
        if self.config.enable_graph_similarity:
            similarity_scores = self._evaluate_graph_similarity(original_data, generated_data_list)
            results['graph_similarity'] = similarity_scores
            logger.info(f"图结构相似度 - 平均分数: {np.mean(list(similarity_scores.values())):.4f}")
        
        # 2. MILP特征相似度评估
        if self.config.enable_milp_similarity:
            milp_scores = self._evaluate_milp_similarity(original_data, generated_data_list)
            results['milp_similarity'] = milp_scores
            logger.info(f"MILP特征相似度 - 平均分数: {np.mean(list(milp_scores.values())):.4f}")
        
        # 3. 生成多样性评估
        if self.config.enable_diversity_analysis:
            diversity_scores = self._evaluate_generation_diversity(generated_data_list, generation_info)
            results['diversity_analysis'] = diversity_scores
            logger.info(f"生成多样性 - 总体分数: {diversity_scores.get('overall_diversity_score', 0.0):.4f}")
        
        # 4. 综合质量评分
        overall_score = self._compute_overall_quality_score(results)
        results['overall_quality_score'] = overall_score
        
        # 5. 基准对比
        benchmark_results = self._compare_with_benchmarks(results)
        results['benchmark_comparison'] = benchmark_results
        
        # 6. 保存结果
        if self.config.save_detailed_results:
            self._save_evaluation_results(results)
        
        # 7. 生成可视化
        if self.config.generate_visualizations:
            self._generate_evaluation_visualizations(results, original_data, generated_data_list)
        
        # 记录评估历史
        self.evaluation_history.append(results)
        
        logger.info(f"生成质量评估完成 - 综合得分: {overall_score:.4f}")
        return results
    
    def _evaluate_graph_similarity(self, 
                                  original_data: HeteroData,
                                  generated_data_list: List[HeteroData]) -> Dict[str, float]:
        """评估图结构相似度"""
        try:
            similarity_scores = {}
            
            for metric in self.config.similarity_metrics:
                scores = []
                
                for gen_data in generated_data_list:
                    if metric == 'degree_distribution':
                        score = self._compute_degree_distribution_similarity(original_data, gen_data)
                    elif metric == 'clustering_coefficient':
                        score = self._compute_clustering_similarity(original_data, gen_data)
                    elif metric == 'node_feature_similarity':
                        score = self._compute_node_feature_similarity(original_data, gen_data)
                    elif metric == 'edge_weight_similarity':
                        score = self._compute_edge_weight_similarity(original_data, gen_data)
                    elif metric == 'sparsity_similarity':
                        score = self._compute_sparsity_similarity(original_data, gen_data)
                    else:
                        score = 0.0
                    
                    scores.append(score)
                
                similarity_scores[metric] = np.mean(scores)
            
            # 计算加权平均相似度
            if similarity_scores:
                weights = [1.0] * len(similarity_scores)  # 等权重
                weighted_avg = np.average(list(similarity_scores.values()), weights=weights)
                similarity_scores['weighted_average'] = weighted_avg
            
            return similarity_scores
            
        except Exception as e:
            logger.error(f"图结构相似度评估失败: {e}")
            return {}
    
    def _compute_degree_distribution_similarity(self, 
                                              original_data: HeteroData,
                                              generated_data: HeteroData) -> float:
        """计算度分布相似度"""
        try:
            # 提取约束节点的度分布
            orig_degrees = self._extract_constraint_degrees(original_data)
            gen_degrees = self._extract_constraint_degrees(generated_data)
            
            if len(orig_degrees) == 0 or len(gen_degrees) == 0:
                return 0.0
            
            # 使用Wasserstein距离计算分布相似度
            distance = wasserstein_distance(orig_degrees, gen_degrees)
            
            # 转换为相似度分数 (0-1)
            max_degree = max(max(orig_degrees), max(gen_degrees))
            similarity = 1.0 - min(distance / max_degree, 1.0)
            
            return float(similarity)
            
        except Exception as e:
            logger.warning(f"度分布相似度计算失败: {e}")
            return 0.0
    
    def _extract_constraint_degrees(self, data: HeteroData) -> List[int]:
        """提取约束节点的度"""
        try:
            if ('constraint', 'connects', 'variable') not in data.edge_index_dict:
                return []
            
            edge_index = data[('constraint', 'connects', 'variable')].edge_index
            constraint_ids = edge_index[0].cpu().numpy()
            
            # 计算每个约束的度
            unique_constraints, counts = np.unique(constraint_ids, return_counts=True)
            degrees = counts.tolist()
            
            return degrees
            
        except Exception as e:
            logger.warning(f"约束度提取失败: {e}")
            return []
    
    def _compute_clustering_similarity(self, 
                                     original_data: HeteroData,
                                     generated_data: HeteroData) -> float:
        """计算聚类系数相似度"""
        try:
            # 简化实现：使用变量节点的连接密度作为聚类指标
            orig_density = self._compute_graph_density(original_data)
            gen_density = self._compute_graph_density(generated_data)
            
            if orig_density == 0 and gen_density == 0:
                return 1.0
            
            # 计算密度相似度
            max_density = max(orig_density, gen_density)
            min_density = min(orig_density, gen_density)
            
            similarity = min_density / max_density if max_density > 0 else 0.0
            
            return float(similarity)
            
        except Exception as e:
            logger.warning(f"聚类相似度计算失败: {e}")
            return 0.0
    
    def _compute_graph_density(self, data: HeteroData) -> float:
        """计算图密度"""
        try:
            if ('constraint', 'connects', 'variable') not in data.edge_index_dict:
                return 0.0
            
            edge_index = data[('constraint', 'connects', 'variable')].edge_index
            n_edges = edge_index.size(1)
            
            n_constraints = data['constraint'].x.size(0)
            n_variables = data['variable'].x.size(0)
            max_edges = n_constraints * n_variables
            
            density = n_edges / max_edges if max_edges > 0 else 0.0
            
            return float(density)
            
        except Exception as e:
            logger.warning(f"图密度计算失败: {e}")
            return 0.0
    
    def _compute_node_feature_similarity(self, 
                                       original_data: HeteroData,
                                       generated_data: HeteroData) -> float:
        """计算节点特征相似度"""
        try:
            similarities = []
            
            for node_type in ['constraint', 'variable']:
                if node_type in original_data and node_type in generated_data:
                    orig_features = original_data[node_type].x
                    gen_features = generated_data[node_type].x
                    
                    # 计算特征统计量的相似度
                    orig_stats = torch.stack([orig_features.mean(dim=0), orig_features.std(dim=0)])
                    gen_stats = torch.stack([gen_features.mean(dim=0), gen_features.std(dim=0)])
                    
                    # 余弦相似度
                    cosine_sim = F.cosine_similarity(
                        orig_stats.flatten(), gen_stats.flatten(), dim=0
                    ).item()
                    
                    similarities.append(max(0.0, cosine_sim))  # 确保非负
            
            return float(np.mean(similarities)) if similarities else 0.0
            
        except Exception as e:
            logger.warning(f"节点特征相似度计算失败: {e}")
            return 0.0
    
    def _compute_edge_weight_similarity(self, 
                                      original_data: HeteroData,
                                      generated_data: HeteroData) -> float:
        """计算边权重相似度"""
        try:
            # 提取边权重（如果存在）
            orig_weights = self._extract_edge_weights(original_data)
            gen_weights = self._extract_edge_weights(generated_data)
            
            if len(orig_weights) == 0 or len(gen_weights) == 0:
                return 0.0
            
            # 使用统计量比较
            orig_mean = np.mean(orig_weights)
            gen_mean = np.mean(gen_weights)
            orig_std = np.std(orig_weights)
            gen_std = np.std(gen_weights)
            
            # 计算均值和标准差的相似度
            mean_similarity = 1.0 - abs(orig_mean - gen_mean) / (abs(orig_mean) + abs(gen_mean) + 1e-8)
            std_similarity = 1.0 - abs(orig_std - gen_std) / (abs(orig_std) + abs(gen_std) + 1e-8)
            
            # 加权平均
            similarity = 0.6 * mean_similarity + 0.4 * std_similarity
            
            return float(max(0.0, similarity))
            
        except Exception as e:
            logger.warning(f"边权重相似度计算失败: {e}")
            return 0.0
    
    def _extract_edge_weights(self, data: HeteroData) -> List[float]:
        """提取边权重"""
        try:
            edge_type = ('constraint', 'connects', 'variable')
            if edge_type not in data.edge_index_dict:
                return []
            
            # 如果有边特征，使用第一维作为权重
            if hasattr(data[edge_type], 'edge_attr') and data[edge_type].edge_attr is not None:
                weights = data[edge_type].edge_attr[:, 0].cpu().numpy().tolist()
            else:
                # 如果没有边特征，使用单位权重
                n_edges = data[edge_type].edge_index.size(1)
                weights = [1.0] * n_edges
            
            return weights
            
        except Exception as e:
            logger.warning(f"边权重提取失败: {e}")
            return []
    
    def _compute_sparsity_similarity(self, 
                                   original_data: HeteroData,
                                   generated_data: HeteroData) -> float:
        """计算稀疏性相似度"""
        try:
            orig_density = self._compute_graph_density(original_data)
            gen_density = self._compute_graph_density(generated_data)
            
            # 稀疏性 = 1 - 密度
            orig_sparsity = 1.0 - orig_density
            gen_sparsity = 1.0 - gen_density
            
            # 相似度计算
            if orig_sparsity == 0 and gen_sparsity == 0:
                return 1.0
            
            similarity = 1.0 - abs(orig_sparsity - gen_sparsity) / max(orig_sparsity, gen_sparsity, 1e-8)
            
            return float(max(0.0, similarity))
            
        except Exception as e:
            logger.warning(f"稀疏性相似度计算失败: {e}")
            return 0.0
    
    def _evaluate_milp_similarity(self, 
                                 original_data: HeteroData,
                                 generated_data_list: List[HeteroData]) -> Dict[str, float]:
        """评估MILP特征相似度"""
        try:
            milp_scores = {}
            
            # MILP关键特征
            features = ['constraint_count', 'variable_count', 'edge_count', 'avg_constraint_degree']
            
            for feature in features:
                orig_value = self._extract_milp_feature(original_data, feature)
                gen_values = [self._extract_milp_feature(gen_data, feature) for gen_data in generated_data_list]
                
                # 计算特征相似度
                similarities = []
                for gen_value in gen_values:
                    if orig_value == 0 and gen_value == 0:
                        sim = 1.0
                    else:
                        sim = 1.0 - abs(orig_value - gen_value) / max(abs(orig_value), abs(gen_value), 1e-8)
                    similarities.append(max(0.0, sim))
                
                milp_scores[feature] = np.mean(similarities)
            
            # 综合MILP相似度
            if milp_scores:
                milp_scores['overall_milp_similarity'] = np.mean(list(milp_scores.values()))
            
            return milp_scores
            
        except Exception as e:
            logger.error(f"MILP特征相似度评估失败: {e}")
            return {}
    
    def _extract_milp_feature(self, data: HeteroData, feature: str) -> float:
        """提取MILP特征"""
        try:
            if feature == 'constraint_count':
                return float(data['constraint'].x.size(0))
            elif feature == 'variable_count':
                return float(data['variable'].x.size(0))
            elif feature == 'edge_count':
                edge_type = ('constraint', 'connects', 'variable')
                if edge_type in data.edge_index_dict:
                    return float(data[edge_type].edge_index.size(1))
                return 0.0
            elif feature == 'avg_constraint_degree':
                degrees = self._extract_constraint_degrees(data)
                return float(np.mean(degrees)) if degrees else 0.0
            else:
                return 0.0
                
        except Exception as e:
            logger.warning(f"MILP特征提取失败 {feature}: {e}")
            return 0.0
    
    def _evaluate_generation_diversity(self, 
                                     generated_data_list: List[HeteroData],
                                     generation_info: List[Dict] = None) -> Dict[str, float]:
        """评估生成多样性"""
        try:
            diversity_scores = {}
            
            if len(generated_data_list) < 2:
                logger.warning("样本数量不足，无法评估多样性")
                return {'overall_diversity_score': 0.0}
            
            # 1. 结构多样性
            structural_diversity = self._compute_structural_diversity(generated_data_list)
            diversity_scores['structural_diversity'] = structural_diversity
            
            # 2. 特征多样性
            feature_diversity = self._compute_feature_diversity(generated_data_list)
            diversity_scores['feature_diversity'] = feature_diversity
            
            # 3. 生成过程多样性（如果有生成信息）
            if generation_info:
                process_diversity = self._compute_process_diversity(generation_info)
                diversity_scores.update(process_diversity)
            
            # 4. 综合多样性分数
            main_scores = [structural_diversity, feature_diversity]
            diversity_scores['overall_diversity_score'] = np.mean(main_scores)
            
            return diversity_scores
            
        except Exception as e:
            logger.error(f"生成多样性评估失败: {e}")
            return {'overall_diversity_score': 0.0}
    
    def _compute_structural_diversity(self, generated_data_list: List[HeteroData]) -> float:
        """计算结构多样性"""
        try:
            # 提取每个图的结构特征
            features = []
            for data in generated_data_list:
                feature_vector = [
                    self._compute_graph_density(data),
                    len(self._extract_constraint_degrees(data)),
                    np.mean(self._extract_constraint_degrees(data)) if self._extract_constraint_degrees(data) else 0.0,
                    np.std(self._extract_constraint_degrees(data)) if len(self._extract_constraint_degrees(data)) > 1 else 0.0
                ]
                features.append(feature_vector)
            
            # 计算特征向量间的多样性
            features = np.array(features)
            if features.shape[0] < 2:
                return 0.0
            
            # 使用标准差作为多样性指标
            diversity = np.mean(np.std(features, axis=0))
            
            return float(diversity)
            
        except Exception as e:
            logger.warning(f"结构多样性计算失败: {e}")
            return 0.0
    
    def _compute_feature_diversity(self, generated_data_list: List[HeteroData]) -> float:
        """计算特征多样性"""
        try:
            diversities = []
            
            for node_type in ['constraint', 'variable']:
                if all(node_type in data for data in generated_data_list):
                    # 提取每个图的节点特征统计量
                    feature_stats = []
                    for data in generated_data_list:
                        features = data[node_type].x
                        stats = torch.cat([features.mean(dim=0), features.std(dim=0)])
                        feature_stats.append(stats.cpu().numpy())
                    
                    # 计算统计量的多样性
                    feature_stats = np.array(feature_stats)
                    if feature_stats.shape[0] > 1:
                        diversity = np.mean(np.std(feature_stats, axis=0))
                        diversities.append(diversity)
            
            return float(np.mean(diversities)) if diversities else 0.0
            
        except Exception as e:
            logger.warning(f"特征多样性计算失败: {e}")
            return 0.0
    
    def _compute_process_diversity(self, generation_info: List[Dict]) -> Dict[str, float]:
        """计算生成过程多样性"""
        try:
            process_scores = {}
            
            # 提取关键生成指标
            all_biases = []
            all_degrees = []
            all_connections = []
            all_constraints = []
            
            for info in generation_info:
                if 'diversity_stats' in info:
                    stats = info['diversity_stats']
                    if 'avg_bias' in stats:
                        all_biases.append(stats['avg_bias'])
                    if 'avg_degree' in stats:
                        all_degrees.append(stats['avg_degree'])
                    if 'avg_connections' in stats:
                        all_connections.append(stats['avg_connections'])
                
                if 'generation_history' in info:
                    constraints = [h['masked_constraint_id'] for h in info['generation_history']]
                    all_constraints.extend(constraints)
            
            # 计算各指标的多样性
            if all_biases:
                process_scores['bias_diversity'] = float(np.std(all_biases))
            if all_degrees:
                process_scores['degree_diversity'] = float(np.std(all_degrees))
            if all_connections:
                process_scores['connection_diversity'] = float(np.std(all_connections))
            if all_constraints:
                unique_ratio = len(set(all_constraints)) / len(all_constraints)
                process_scores['constraint_diversity'] = float(unique_ratio)
            
            return process_scores
            
        except Exception as e:
            logger.warning(f"生成过程多样性计算失败: {e}")
            return {}
    
    def _compute_overall_quality_score(self, results: Dict[str, Any]) -> float:
        """计算综合质量评分"""
        try:
            scores = []
            weights = []
            
            # 图结构相似度 (权重: 0.4)
            if 'graph_similarity' in results and 'weighted_average' in results['graph_similarity']:
                scores.append(results['graph_similarity']['weighted_average'])
                weights.append(0.4)
            
            # MILP特征相似度 (权重: 0.3)
            if 'milp_similarity' in results and 'overall_milp_similarity' in results['milp_similarity']:
                scores.append(results['milp_similarity']['overall_milp_similarity'])
                weights.append(0.3)
            
            # 生成多样性 (权重: 0.3)
            if 'diversity_analysis' in results and 'overall_diversity_score' in results['diversity_analysis']:
                # 多样性分数需要适当缩放（过高的多样性可能表示质量差）
                diversity_score = results['diversity_analysis']['overall_diversity_score']
                scaled_diversity = min(diversity_score * 2.0, 1.0)  # 适度的多样性是好的
                scores.append(scaled_diversity)
                weights.append(0.3)
            
            if not scores:
                return 0.0
            
            # 加权平均
            overall_score = np.average(scores, weights=weights)
            
            return float(max(0.0, min(1.0, overall_score)))  # 限制在[0,1]范围内
            
        except Exception as e:
            logger.warning(f"综合质量评分计算失败: {e}")
            return 0.0
    
    def _compare_with_benchmarks(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """与基准进行对比"""
        try:
            benchmark_results = {}
            
            # 提取关键指标
            overall_score = results.get('overall_quality_score', 0.0)
            
            graph_sim = 0.0
            if 'graph_similarity' in results and 'weighted_average' in results['graph_similarity']:
                graph_sim = results['graph_similarity']['weighted_average']
            
            milp_sim = 0.0
            if 'milp_similarity' in results and 'overall_milp_similarity' in results['milp_similarity']:
                milp_sim = results['milp_similarity']['overall_milp_similarity']
            
            diversity_score = 0.0
            if 'diversity_analysis' in results and 'overall_diversity_score' in results['diversity_analysis']:
                diversity_score = results['diversity_analysis']['overall_diversity_score']
            
            # 与基准对比
            benchmarks = self.config.quality_benchmarks
            
            benchmark_results['overall_quality'] = {
                'score': overall_score,
                'benchmark': benchmarks.get('similarity_score', 0.6),
                'meets_benchmark': overall_score >= benchmarks.get('similarity_score', 0.6),
                'improvement_needed': max(0.0, benchmarks.get('similarity_score', 0.6) - overall_score)
            }
            
            benchmark_results['graph_similarity'] = {
                'score': graph_sim,
                'benchmark': benchmarks.get('similarity_score', 0.6),
                'meets_benchmark': graph_sim >= benchmarks.get('similarity_score', 0.6)
            }
            
            benchmark_results['milp_similarity'] = {
                'score': milp_sim,
                'benchmark': benchmarks.get('similarity_score', 0.6),
                'meets_benchmark': milp_sim >= benchmarks.get('similarity_score', 0.6)
            }
            
            benchmark_results['diversity'] = {
                'score': diversity_score,
                'benchmark': benchmarks.get('diversity_score', 0.3),
                'meets_benchmark': diversity_score >= benchmarks.get('diversity_score', 0.3)
            }
            
            # 综合评级
            passed_benchmarks = sum(1 for result in benchmark_results.values() 
                                  if isinstance(result, dict) and result.get('meets_benchmark', False))
            total_benchmarks = len([r for r in benchmark_results.values() if isinstance(r, dict)])
            
            if total_benchmarks > 0:
                benchmark_pass_rate = passed_benchmarks / total_benchmarks
                if benchmark_pass_rate >= 0.75:
                    grade = "A"
                elif benchmark_pass_rate >= 0.5:
                    grade = "B"
                elif benchmark_pass_rate >= 0.25:
                    grade = "C"
                else:
                    grade = "D"
            else:
                grade = "N/A"
            
            benchmark_results['summary'] = {
                'passed_benchmarks': passed_benchmarks,
                'total_benchmarks': total_benchmarks,
                'pass_rate': benchmark_pass_rate if total_benchmarks > 0 else 0.0,
                'grade': grade
            }
            
            return benchmark_results
            
        except Exception as e:
            logger.warning(f"基准对比失败: {e}")
            return {}
    
    def _save_evaluation_results(self, results: Dict[str, Any]):
        """保存评估结果"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = self.output_dir / f"evaluation_results_{timestamp}.json"
            
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"评估结果已保存: {results_file}")
            
        except Exception as e:
            logger.error(f"保存评估结果失败: {e}")
    
    def _generate_evaluation_visualizations(self, 
                                          results: Dict[str, Any],
                                          original_data: HeteroData,
                                          generated_data_list: List[HeteroData]):
        """生成评估可视化"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 创建综合评估报告
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('G2MILP Generation Quality Evaluation Report', fontsize=16)
            
            # 1. 相似度分数雷达图
            self._plot_similarity_radar(axes[0, 0], results)
            
            # 2. 多样性分析
            self._plot_diversity_analysis(axes[0, 1], results)
            
            # 3. 基准对比
            self._plot_benchmark_comparison(axes[1, 0], results)
            
            # 4. 质量趋势（如果有历史数据）
            self._plot_quality_trend(axes[1, 1])
            
            plt.tight_layout()
            viz_file = self.output_dir / f"evaluation_report_{timestamp}.png"
            plt.savefig(viz_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            # 生成详细的度分布对比图
            self._plot_degree_distribution_comparison(original_data, generated_data_list, timestamp)
            
            logger.info(f"评估可视化已生成: {viz_file}")
            
        except Exception as e:
            logger.warning(f"生成评估可视化失败: {e}")
    
    def _plot_similarity_radar(self, ax, results: Dict[str, Any]):
        """绘制相似度雷达图"""
        try:
            if 'graph_similarity' not in results:
                ax.text(0.5, 0.5, 'No similarity data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Graph Similarity Metrics')
                return
            
            similarity_data = results['graph_similarity']
            metrics = [k for k in similarity_data.keys() if k != 'weighted_average']
            scores = [similarity_data[k] for k in metrics]
            
            if not metrics:
                ax.text(0.5, 0.5, 'No metrics available', ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Graph Similarity Metrics')
                return
            
            # 简化的条形图代替雷达图
            ax.barh(metrics, scores, color='skyblue', alpha=0.7)
            ax.set_xlim(0, 1)
            ax.set_xlabel('Similarity Score')
            ax.set_title('Graph Similarity Metrics')
            ax.grid(True, alpha=0.3)
            
        except Exception as e:
            logger.warning(f"相似度雷达图绘制失败: {e}")
            ax.text(0.5, 0.5, f'Error: {str(e)[:50]}', ha='center', va='center', transform=ax.transAxes)
    
    def _plot_diversity_analysis(self, ax, results: Dict[str, Any]):
        """绘制多样性分析"""
        try:
            if 'diversity_analysis' not in results:
                ax.text(0.5, 0.5, 'No diversity data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Generation Diversity')
                return
            
            diversity_data = results['diversity_analysis']
            metrics = [k for k in diversity_data.keys() if k != 'overall_diversity_score']
            scores = [diversity_data[k] for k in metrics]
            
            if not metrics:
                ax.text(0.5, 0.5, 'No diversity metrics', ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Generation Diversity')
                return
            
            ax.bar(range(len(metrics)), scores, color='lightgreen', alpha=0.7)
            ax.set_xticks(range(len(metrics)))
            ax.set_xticklabels(metrics, rotation=45, ha='right')
            ax.set_ylabel('Diversity Score')
            ax.set_title('Generation Diversity Analysis')
            ax.grid(True, alpha=0.3)
            
        except Exception as e:
            logger.warning(f"多样性分析图绘制失败: {e}")
            ax.text(0.5, 0.5, f'Error: {str(e)[:50]}', ha='center', va='center', transform=ax.transAxes)
    
    def _plot_benchmark_comparison(self, ax, results: Dict[str, Any]):
        """绘制基准对比"""
        try:
            if 'benchmark_comparison' not in results:
                ax.text(0.5, 0.5, 'No benchmark data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Benchmark Comparison')
                return
            
            benchmark_data = results['benchmark_comparison']
            metrics = []
            scores = []
            benchmarks = []
            
            for key, value in benchmark_data.items():
                if isinstance(value, dict) and 'score' in value and 'benchmark' in value:
                    metrics.append(key.replace('_', ' ').title())
                    scores.append(value['score'])
                    benchmarks.append(value['benchmark'])
            
            if not metrics:
                ax.text(0.5, 0.5, 'No benchmark metrics', ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Benchmark Comparison')
                return
            
            x = np.arange(len(metrics))
            width = 0.35
            
            ax.bar(x - width/2, scores, width, label='Actual', color='lightblue', alpha=0.7)
            ax.bar(x + width/2, benchmarks, width, label='Benchmark', color='orange', alpha=0.7)
            
            ax.set_xlabel('Metrics')
            ax.set_ylabel('Score')
            ax.set_title('Performance vs Benchmarks')
            ax.set_xticks(x)
            ax.set_xticklabels(metrics, rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
        except Exception as e:
            logger.warning(f"基准对比图绘制失败: {e}")
            ax.text(0.5, 0.5, f'Error: {str(e)[:50]}', ha='center', va='center', transform=ax.transAxes)
    
    def _plot_quality_trend(self, ax):
        """绘制质量趋势"""
        try:
            if len(self.evaluation_history) < 2:
                ax.text(0.5, 0.5, 'Insufficient history data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Quality Trend')
                return
            
            scores = [result.get('overall_quality_score', 0.0) for result in self.evaluation_history]
            timestamps = range(len(scores))
            
            ax.plot(timestamps, scores, marker='o', linewidth=2, markersize=6, color='purple')
            ax.set_xlabel('Evaluation Session')
            ax.set_ylabel('Overall Quality Score')
            ax.set_title('Quality Improvement Trend')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1)
            
        except Exception as e:
            logger.warning(f"质量趋势图绘制失败: {e}")
            ax.text(0.5, 0.5, f'Error: {str(e)[:50]}', ha='center', va='center', transform=ax.transAxes)
    
    def _plot_degree_distribution_comparison(self, 
                                           original_data: HeteroData,
                                           generated_data_list: List[HeteroData],
                                           timestamp: str):
        """绘制度分布对比图"""
        try:
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            
            # 原始数据度分布
            orig_degrees = self._extract_constraint_degrees(original_data)
            if orig_degrees:
                ax.hist(orig_degrees, bins=20, alpha=0.7, label='Original', color='blue', density=True)
            
            # 生成数据度分布
            all_gen_degrees = []
            for gen_data in generated_data_list:
                gen_degrees = self._extract_constraint_degrees(gen_data)
                all_gen_degrees.extend(gen_degrees)
            
            if all_gen_degrees:
                ax.hist(all_gen_degrees, bins=20, alpha=0.7, label='Generated', color='red', density=True)
            
            ax.set_xlabel('Constraint Degree')
            ax.set_ylabel('Density')
            ax.set_title('Degree Distribution Comparison')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            degree_file = self.output_dir / f"degree_distribution_{timestamp}.png"
            plt.savefig(degree_file, dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.warning(f"度分布对比图绘制失败: {e}")


def create_evaluator(config: EvaluationConfig = None) -> G2MILPEvaluator:
    """
    创建G2MILP评估器的工厂函数
    
    Args:
        config: 评估配置
        
    Returns:
        G2MILP评估器实例
    """
    return G2MILPEvaluator(config)


if __name__ == "__main__":
    # 测试代码
    print("G2MILP评估模块测试")
    print("=" * 40)
    
    # 创建测试配置
    eval_config = EvaluationConfig(
        enable_graph_similarity=True,
        enable_milp_similarity=True,
        enable_diversity_analysis=True,
        diversity_sample_size=5
    )
    
    # 创建评估器
    evaluator = create_evaluator(eval_config)
    
    print(f"评估器配置:")
    print(f"- 图结构相似度: {eval_config.enable_graph_similarity}")
    print(f"- MILP特征相似度: {eval_config.enable_milp_similarity}")
    print(f"- 多样性分析: {eval_config.enable_diversity_analysis}")
    print(f"- 输出目录: {eval_config.output_dir}")
    print("评估器创建成功!")
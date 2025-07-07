"""
G2MILP推理模块
G2MILP Inference Module

实现G2MILP的推理过程，包括：
1. 基于训练好的模型生成新的MILP实例
2. 可控的生成参数（η、温度等）
3. 生成结果的分析和可视化
4. 与原始实例的对比分析
"""

import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import json
import pickle
import time
from pathlib import Path
from datetime import datetime
import copy
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import csr_matrix

from .generator import G2MILPGenerator
from ..g2milp_bipartite import BipartiteGraphRepresentation

logger = logging.getLogger(__name__)


@dataclass
class InferenceConfig:
    """推理配置"""
    # 生成参数
    eta: float = 0.1                    # 遮盖比例，控制相似度vs创新性
    num_iterations: Optional[int] = None # 迭代次数，None则根据η计算
    temperature: float = 1.0             # 采样温度
    sample_from_prior: bool = True       # 是否从先验分布采样
    
    # 生成策略
    constraint_selection_strategy: str = "random"  # random, degree_based, importance_based
    diversity_boost: bool = False        # 是否使用多样性增强
    num_diverse_samples: int = 5        # 多样性采样数量
    num_test_instances: int = 5         # 测试实例数量
    
    # 后处理
    apply_constraints_validation: bool = True   # 是否验证约束有效性
    normalize_weights: bool = True              # 是否归一化权重
    round_integer_variables: bool = True        # 是否舍入整数变量
    
    # 分析和比较
    compute_similarity_metrics: bool = True     # 计算相似度指标
    generate_comparison_report: bool = True     # 生成对比报告
    save_intermediate_states: bool = False      # 保存中间状态
    
    # 输出设置
    output_dir: str = "output/demo4_g2milp/inference"
    experiment_name: str = f"inference_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    save_generated_instances: bool = True
    
    # 设备
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class G2MILPInference:
    """
    G2MILP推理器
    
    使用训练好的G2MILP模型生成新的MILP实例
    """
    
    def __init__(self, 
                 model: G2MILPGenerator,
                 config: InferenceConfig = None):
        self.model = model
        self.config = config or InferenceConfig()
        
        # 设置设备
        self.device = torch.device(self.config.device)
        self.model = self.model.to(self.device)
        self.model.eval()  # 设置为推理模式
        
        # 创建输出目录
        self.output_dir = Path(self.config.output_dir) / self.config.experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 推理历史
        self.inference_history = []
        
        logger.info(f"G2MILP推理器初始化完成")
        logger.info(f"设备: {self.device}")
        logger.info(f"输出目录: {self.output_dir}")
    
    def generate_single_instance(self, 
                                source_data: HeteroData,
                                save_intermediate: bool = None,
                                dynamic_config: Dict = None) -> Dict[str, Any]:
        """
        生成单个MILP实例（增强版支持动态配置）
        
        Args:
            source_data: 源数据（有偏差的二分图）
            save_intermediate: 是否保存中间状态
            dynamic_config: 动态生成配置（用于多样性增强）
            
        Returns:
            生成结果字典
        """
        if save_intermediate is None:
            save_intermediate = self.config.save_intermediate_states
        
        logger.info("开始生成单个MILP实例")
        
        # 移动数据到设备
        source_data = source_data.to(self.device)
        
        # 应用动态配置（用于多样性增强）
        if dynamic_config is not None:
            effective_eta = dynamic_config.get('eta', self.config.eta)
            effective_temperature = dynamic_config.get('temperature', getattr(self.model.config, 'temperature', 1.0))
            constraint_selection_strategy = dynamic_config.get('constraint_selection_strategy', 'random')
            logger.info(f"使用动态配置: η={effective_eta:.3f}, temp={effective_temperature:.3f}, strategy={constraint_selection_strategy}")
        else:
            effective_eta = self.config.eta
            effective_temperature = getattr(self.model.config, 'temperature', 1.0)
            constraint_selection_strategy = 'random'
        
        # 确定迭代次数
        n_constraints = source_data['constraint'].x.size(0)
        if self.config.num_iterations is None:
            num_iterations = max(1, int(effective_eta * n_constraints))
        else:
            num_iterations = self.config.num_iterations
        
        logger.info(f"迭代次数: {num_iterations}, η = {effective_eta:.3f}")
        
        # 初始化生成过程
        current_data = copy.deepcopy(source_data)
        generation_steps = []
        similarity_evolution = []
        
        with torch.no_grad():
            for iteration in range(num_iterations):
                logger.debug(f"推理迭代 {iteration + 1}/{num_iterations}")
                
                # 记录迭代前状态
                if save_intermediate:
                    pre_iteration_state = self._extract_graph_state(current_data)
                
                # 单次生成迭代
                current_data, iteration_info = self.model.generate_single_iteration(
                    current_data
                )
                
                # 确保数据设备一致性
                current_data = self._ensure_device_consistency(current_data)
                
                # 计算与原始实例的相似度
                if self.config.compute_similarity_metrics:
                    similarity = self._compute_similarity(source_data, current_data)
                    similarity_evolution.append(similarity)
                    iteration_info['similarity'] = similarity
                
                # 记录生成步骤
                iteration_info['iteration'] = iteration
                generation_steps.append(iteration_info)
                
                # 保存中间状态
                if save_intermediate:
                    post_iteration_state = self._extract_graph_state(current_data)
                    self._save_intermediate_state(
                        iteration, pre_iteration_state, post_iteration_state, iteration_info
                    )
                
                logger.debug(f"完成迭代 {iteration + 1}")
        
        # 后处理
        if self.config.apply_constraints_validation:
            current_data = self._validate_and_fix_constraints(current_data)
        
        # 生成结果分析
        result = {
            'generated_data': current_data.cpu(),
            'source_data': source_data.cpu(),
            'generation_steps': generation_steps,
            'similarity_evolution': similarity_evolution,
            'num_iterations': num_iterations,
            'eta': self.config.eta,
            'generation_config': asdict(self.config),
            'generation_timestamp': datetime.now().isoformat()
        }
        
        # 详细分析
        if self.config.generate_comparison_report:
            analysis = self._analyze_generated_instance(
                source_data.cpu(), current_data.cpu(), generation_steps
            )
            result['analysis'] = analysis
        
        logger.info("单个MILP实例生成完成")
        
        return result
    
    def generate_multiple_instances(self, 
                                  source_data: HeteroData,
                                  num_instances: int = 5) -> List[Dict[str, Any]]:
        """
        生成多个MILP实例
        
        Args:
            source_data: 源数据
            num_instances: 生成实例数量
            
        Returns:
            生成结果列表
        """
        logger.info(f"开始生成 {num_instances} 个MILP实例")
        
        results = []
        
        for i in range(num_instances):
            logger.info(f"生成第 {i+1}/{num_instances} 个实例")
            
            # 改进的随机种子策略
            import time
            import os
            enhanced_seed = (
                int(time.time() * 1000000) % 1000000 +  # 微秒时间戳
                os.getpid() * 1000 +                    # 进程ID
                i * 1337 +                              # 实例序号
                hash(str(source_data)) % 10000          # 数据哈希
            ) % 2**32
            
            torch.manual_seed(enhanced_seed)
            np.random.seed(enhanced_seed % 2**31)
            
            # 为每个实例使用不同的生成参数
            dynamic_config = self._generate_dynamic_config(i, num_instances)
            
            result = self.generate_single_instance(
                source_data, 
                save_intermediate=False,
                dynamic_config=dynamic_config
            )
            result['instance_id'] = i
            result['dynamic_config'] = dynamic_config
            results.append(result)
        
        # 分析多实例生成结果
        multi_analysis = self._analyze_multiple_instances(results)
        
        # 保存结果
        if self.config.save_generated_instances:
            self._save_multiple_instances(results, multi_analysis)
        
        logger.info(f"完成生成 {num_instances} 个MILP实例")
        
        return results
    
    def generate_instances(self, 
                          source_data: HeteroData,
                          num_samples: int = 3) -> Dict[str, Any]:
        """
        生成MILP实例（训练时质量评估专用接口）
        
        这是为了与training.py中的质量评估接口兼容而添加的包装方法
        
        Args:
            source_data: 源数据（有偏差的二分图）
            num_samples: 生成实例数量
            
        Returns:
            格式化的生成结果字典，包含：
            - generated_instances: 生成的实例列表
            - generation_info: 生成过程信息
        """
        logger.info(f"🔍 质量评估模式：生成 {num_samples} 个测试实例")
        
        # 使用现有的generate_multiple_instances方法
        generation_results = self.generate_multiple_instances(
            source_data=source_data,
            num_instances=num_samples
        )
        
        # 提取生成的图数据
        generated_instances = []
        generation_info = {
            'num_generated': len(generation_results),
            'average_iterations': 0,
            'average_similarity': 0.0,
            'generation_success': True,
            'detailed_results': generation_results
        }
        
        # 处理生成结果
        total_iterations = 0
        total_similarity = 0.0
        
        for i, result in enumerate(generation_results):
            try:
                # 提取生成的图数据
                if 'generated_data' in result:
                    generated_instances.append(result['generated_data'])
                else:
                    logger.warning(f"实例 {i} 缺少 generated_data 字段")
                    generation_info['generation_success'] = False
                
                # 累计统计信息
                if 'generation_steps' in result:
                    total_iterations += len(result['generation_steps'])
                
                if 'final_similarity' in result:
                    total_similarity += result['final_similarity']
                    
            except Exception as e:
                logger.warning(f"处理生成结果 {i} 时出错: {e}")
                generation_info['generation_success'] = False
        
        # 计算平均值
        if len(generation_results) > 0:
            generation_info['average_iterations'] = total_iterations / len(generation_results)
            generation_info['average_similarity'] = total_similarity / len(generation_results)
        
        logger.info(f"✅ 质量评估生成完成：{len(generated_instances)} 个有效实例")
        
        return {
            'generated_instances': generated_instances,
            'generation_info': generation_info
        }
    
    def generate_with_diversity_boost(self, 
                                    source_data: HeteroData) -> List[Dict[str, Any]]:
        """
        使用多样性增强生成多个不同的实例
        
        Args:
            source_data: 源数据
            
        Returns:
            多样化生成结果
        """
        logger.info("开始多样性增强生成")
        
        results = []
        
        # 不同的η值
        eta_values = [0.05, 0.1, 0.15, 0.2, 0.3]
        
        # 不同的温度值
        temperature_values = [0.5, 1.0, 1.5, 2.0]
        
        # 组合不同参数
        param_combinations = []
        for eta in eta_values[:3]:  # 选择前3个η值
            for temp in temperature_values[:2]:  # 选择前2个温度值
                param_combinations.append((eta, temp))
        
        for i, (eta, temp) in enumerate(param_combinations):
            logger.info(f"多样性生成 {i+1}/{len(param_combinations)}: η={eta}, T={temp}")
            
            # 临时修改配置
            original_eta = self.config.eta
            original_temp = self.config.temperature
            
            self.config.eta = eta
            self.config.temperature = temp
            self.model.config.temperature = temp
            
            # 生成实例
            result = self.generate_single_instance(source_data, save_intermediate=False)
            result['diversity_params'] = {'eta': eta, 'temperature': temp}
            result['diversity_instance_id'] = i
            
            results.append(result)
            
            # 恢复原配置
            self.config.eta = original_eta
            self.config.temperature = original_temp
            self.model.config.temperature = original_temp
        
        # 分析多样性结果
        diversity_analysis = self._analyze_diversity_results(results)
        
        logger.info("多样性增强生成完成")
        
        return results
    
    def _ensure_device_consistency(self, data: HeteroData) -> HeteroData:
        """
        确保HeteroData中所有张量在正确设备上
        
        Args:
            data: 异构图数据
            
        Returns:
            设备一致的异构图数据
        """
        try:
            # 移动到目标设备
            data = data.to(self.device)
            
            # 验证设备一致性
            constraint_device = data['constraint'].x.device
            variable_device = data['variable'].x.device
            edge_device = data[('constraint', 'connects', 'variable')].edge_index.device
            
            # 统一设备比较（cuda:0和cuda被视为相同）
            devices_consistent = all(
                str(d).startswith('cuda') == str(self.device).startswith('cuda') 
                for d in [constraint_device, variable_device, edge_device]
            )
            
            if not devices_consistent:
                logger.debug(f"设备不一致检测到，正在修复：constraint={constraint_device}, variable={variable_device}, edge={edge_device}, target={self.device}")
                
                # 强制移动所有组件到目标设备
                for node_type in data.node_types:
                    if hasattr(data[node_type], 'x') and data[node_type].x is not None:
                        data[node_type].x = data[node_type].x.to(self.device)
                
                for edge_type in data.edge_types:
                    if hasattr(data[edge_type], 'edge_index') and data[edge_type].edge_index is not None:
                        data[edge_type].edge_index = data[edge_type].edge_index.to(self.device)
                    if hasattr(data[edge_type], 'edge_attr') and data[edge_type].edge_attr is not None:
                        data[edge_type].edge_attr = data[edge_type].edge_attr.to(self.device)
            
            return data
            
        except Exception as e:
            logger.error(f"设备一致性检查失败: {e}")
            # 尝试基本的设备移动
            return data.to(self.device)
    
    def _extract_graph_state(self, data: HeteroData) -> Dict[str, Any]:
        """提取图状态信息"""
        return {
            'n_constraints': data['constraint'].x.size(0),
            'n_variables': data['variable'].x.size(0),
            'n_edges': data[('constraint', 'connects', 'variable')].edge_index.size(1),
            'constraint_features_norm': torch.norm(data['constraint'].x).item(),
            'variable_features_norm': torch.norm(data['variable'].x).item(),
            'edge_weights_norm': torch.norm(
                data[('constraint', 'connects', 'variable')].edge_attr
            ).item() if hasattr(data[('constraint', 'connects', 'variable')], 'edge_attr') else 0.0
        }
    
    def _compute_similarity(self, source_data: HeteroData, generated_data: HeteroData) -> Dict[str, float]:
        """
        计算生成实例与源实例的多维度相似度（增强版）
        
        Args:
            source_data: 源图数据
            generated_data: 生成的图数据
            
        Returns:
            包含多个维度相似度分数的字典
        """
        similarity = {}
        
        # 1. 基础图结构相似度
        source_edges = source_data[('constraint', 'connects', 'variable')].edge_index
        generated_edges = generated_data[('constraint', 'connects', 'variable')].edge_index
        
        # 1.1 边数量相似度
        edge_count_similarity = 1.0 - abs(
            source_edges.size(1) - generated_edges.size(1)
        ) / max(source_edges.size(1), generated_edges.size(1))
        similarity['edge_count'] = edge_count_similarity
        
        # 1.2 图密度相似度
        n_constraints = source_data['constraint'].x.size(0)
        n_variables = source_data['variable'].x.size(0)
        max_possible_edges = n_constraints * n_variables
        
        source_density = source_edges.size(1) / max_possible_edges
        generated_density = generated_edges.size(1) / max_possible_edges
        density_similarity = 1.0 - abs(source_density - generated_density)
        similarity['density'] = density_similarity
        
        # 2. 度数分布相似度（增强版）
        # 2.1 约束节点度数分布
        source_constraint_degrees = torch.bincount(source_edges[0])
        generated_constraint_degrees = torch.bincount(generated_edges[0])
        constraint_degree_sim = self._compute_distribution_similarity(
            source_constraint_degrees, generated_constraint_degrees
        )
        similarity['constraint_degree_distribution'] = constraint_degree_sim
        
        # 2.2 变量节点度数分布
        source_variable_degrees = torch.bincount(source_edges[1])
        generated_variable_degrees = torch.bincount(generated_edges[1])
        variable_degree_sim = self._compute_distribution_similarity(
            source_variable_degrees, generated_variable_degrees
        )
        similarity['variable_degree_distribution'] = variable_degree_sim
        
        # 3. 图谱特性相似度
        try:
            spectral_sim = self._compute_spectral_similarity(source_data, generated_data)
            similarity.update(spectral_sim)
        except Exception as e:
            logger.warning(f"图谱相似度计算失败: {e}")
            similarity['spectral_similarity'] = 0.5
        
        # 4. 聚类系数相似度
        try:
            clustering_sim = self._compute_clustering_similarity(source_data, generated_data)
            similarity['clustering_coefficient'] = clustering_sim
        except Exception as e:
            logger.warning(f"聚类系数计算失败: {e}")
            similarity['clustering_coefficient'] = 0.5
        
        # 5. MILP特征相似度
        # 5.1 约束特征相似度（分维度计算）
        constraint_feature_sims = self._compute_feature_similarity_detailed(
            source_data['constraint'].x, generated_data['constraint'].x
        )
        for i, sim in enumerate(constraint_feature_sims):
            similarity[f'constraint_feature_dim_{i}'] = sim
        similarity['constraint_features_avg'] = np.mean(constraint_feature_sims)
        
        # 5.2 变量特征相似度（分维度计算）
        variable_feature_sims = self._compute_feature_similarity_detailed(
            source_data['variable'].x, generated_data['variable'].x
        )
        for i, sim in enumerate(variable_feature_sims):
            similarity[f'variable_feature_dim_{i}'] = sim
        similarity['variable_features_avg'] = np.mean(variable_feature_sims)
        
        # 6. 稀疏性模式相似度
        sparsity_sim = self._compute_sparsity_pattern_similarity(source_data, generated_data)
        similarity['sparsity_pattern'] = sparsity_sim
        
        # 7. 连接模式相似度
        connectivity_sim = self._compute_connectivity_pattern_similarity(source_data, generated_data)
        similarity['connectivity_pattern'] = connectivity_sim
        
        # 8. 综合相似度（加权平均）
        weights = {
            'edge_count': 0.15,
            'density': 0.10,
            'constraint_degree_distribution': 0.15,
            'variable_degree_distribution': 0.15,
            'spectral_similarity': 0.10,
            'clustering_coefficient': 0.05,
            'constraint_features_avg': 0.10,
            'variable_features_avg': 0.10,
            'sparsity_pattern': 0.05,
            'connectivity_pattern': 0.05
        }
        
        weighted_similarities = []
        for key, weight in weights.items():
            if key in similarity:
                weighted_similarities.append(similarity[key] * weight)
        
        similarity['overall'] = sum(weighted_similarities) if weighted_similarities else 0.5
        
        return similarity
    
    def _validate_and_fix_constraints(self, data: HeteroData) -> HeteroData:
        """验证和修复约束"""
        # 这里可以实现约束验证和修复逻辑
        # 目前只是返回原数据
        logger.debug("约束验证和修复（占位符实现）")
        return data
    
    def _analyze_generated_instance(self, 
                                  source_data: HeteroData,
                                  generated_data: HeteroData,
                                  generation_steps: List[Dict]) -> Dict[str, Any]:
        """分析生成的实例"""
        analysis = {}
        
        # 基本统计
        analysis['basic_stats'] = {
            'source': self._extract_graph_state(source_data),
            'generated': self._extract_graph_state(generated_data)
        }
        
        # 变化统计
        analysis['changes'] = {
            'edge_count_change': (
                generated_data[('constraint', 'connects', 'variable')].edge_index.size(1) - 
                source_data[('constraint', 'connects', 'variable')].edge_index.size(1)
            ),
            'avg_constraint_degree_change': self._compute_degree_change(source_data, generated_data)
        }
        
        # 生成过程分析
        analysis['generation_process'] = {
            'total_iterations': len(generation_steps),
            'modified_constraints': len(set(step['masked_constraint_id'] for step in generation_steps)),
            'avg_predicted_degree': np.mean([step['predicted_degree'] for step in generation_steps]),
            'avg_connections_per_iteration': np.mean([step['n_connections'] for step in generation_steps])
        }
        
        # 相似度分析
        if 'similarity' in generation_steps[-1]:
            final_similarity = generation_steps[-1]['similarity']
            analysis['final_similarity'] = final_similarity
        
        return analysis
    
    def _compute_degree_change(self, source_data: HeteroData, generated_data: HeteroData) -> float:
        """计算平均度数变化"""
        source_edges = source_data[('constraint', 'connects', 'variable')].edge_index
        generated_edges = generated_data[('constraint', 'connects', 'variable')].edge_index
        
        source_degrees = torch.bincount(source_edges[0]).float()
        generated_degrees = torch.bincount(generated_edges[0]).float()
        
        # 计算平均度数
        source_avg_degree = source_degrees.mean().item()
        generated_avg_degree = generated_degrees.mean().item()
        
        return generated_avg_degree - source_avg_degree
    
    def _analyze_multiple_instances(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析多个生成实例"""
        analysis = {
            'num_instances': len(results),
            'similarity_stats': {},
            'diversity_stats': {},
            'generation_stats': {}
        }
        
        # 相似度统计
        if results and 'analysis' in results[0] and 'final_similarity' in results[0]['analysis']:
            similarities = [
                r['analysis']['final_similarity']['overall'] 
                for r in results if 'analysis' in r
            ]
            
            analysis['similarity_stats'] = {
                'mean': np.mean(similarities),
                'std': np.std(similarities),
                'min': np.min(similarities),
                'max': np.max(similarities)
            }
        
        # 多样性统计（实例间差异）
        edge_counts = [
            r['generated_data'][('constraint', 'connects', 'variable')].edge_index.size(1)
            for r in results
        ]
        
        analysis['diversity_stats'] = {
            'edge_count_diversity': np.std(edge_counts) / np.mean(edge_counts) if np.mean(edge_counts) > 0 else 0,
            'edge_count_range': np.max(edge_counts) - np.min(edge_counts)
        }
        
        # 生成过程统计
        iteration_counts = [r['num_iterations'] for r in results]
        analysis['generation_stats'] = {
            'avg_iterations': np.mean(iteration_counts),
            'total_iterations': np.sum(iteration_counts)
        }
        
        return analysis
    
    def _analyze_diversity_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析多样性生成结果"""
        analysis = {
            'parameter_impact': {},
            'diversity_metrics': {}
        }
        
        # 参数影响分析
        eta_values = set()
        temp_values = set()
        
        for result in results:
            if 'diversity_params' in result:
                eta_values.add(result['diversity_params']['eta'])
                temp_values.add(result['diversity_params']['temperature'])
        
        analysis['parameter_impact'] = {
            'eta_range': (min(eta_values), max(eta_values)),
            'temperature_range': (min(temp_values), max(temp_values)),
            'num_parameter_combinations': len(results)
        }
        
        return analysis
    
    def _save_intermediate_state(self, iteration: int, pre_state: Dict, post_state: Dict, info: Dict):
        """保存中间状态"""
        intermediate_path = self.output_dir / "intermediate_states" 
        intermediate_path.mkdir(exist_ok=True)
        
        state_data = {
            'iteration': iteration,
            'pre_iteration_state': pre_state,
            'post_iteration_state': info,
            'iteration_info': info
        }
        
        with open(intermediate_path / f"iteration_{iteration:03d}.json", 'w') as f:
            json.dump(state_data, f, indent=2, default=str)
    
    def _save_multiple_instances(self, results: List[Dict[str, Any]], analysis: Dict[str, Any]):
        """保存多个实例的结果"""
        # 保存个别实例
        instances_dir = self.output_dir / "generated_instances"
        instances_dir.mkdir(exist_ok=True)
        
        for i, result in enumerate(results):
            instance_path = instances_dir / f"instance_{i:03d}.pkl"
            with open(instance_path, 'wb') as f:
                pickle.dump(result, f)
        
        # 保存汇总分析
        analysis_path = self.output_dir / "multi_instance_analysis.json"
        with open(analysis_path, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        logger.info(f"多实例结果已保存到: {self.output_dir}")
    
    def convert_to_bipartite_representation(self, 
                                          generated_data: HeteroData,
                                          instance_id: str = "generated") -> BipartiteGraphRepresentation:
        """
        将生成的HeteroData转换回BipartiteGraphRepresentation格式
        
        Args:
            generated_data: 生成的异构图数据
            instance_id: 实例ID
            
        Returns:
            二分图表示对象
        """
        # 这里需要实现从HeteroData到BipartiteGraphRepresentation的转换
        # 目前返回占位符
        logger.warning("BipartiteGraphRepresentation转换尚未实现")
        return None
    
    def save_results(self, results: Dict[str, Any], filename: str = "inference_results.pkl"):
        """保存推理结果"""
        result_path = self.output_dir / filename
        
        with open(result_path, 'wb') as f:
            pickle.dump(results, f)
        
        # 同时保存JSON格式的摘要
        summary_path = self.output_dir / filename.replace('.pkl', '_summary.json')
        
        summary = {
            'experiment_name': self.config.experiment_name,
            'inference_config': asdict(self.config),
            'timestamp': datetime.now().isoformat()
        }
        
        if 'analysis' in results:
            summary['analysis'] = results['analysis']
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"推理结果已保存: {result_path}")
    
    def _generate_dynamic_config(self, instance_id: int, total_instances: int) -> Dict[str, Any]:
        """
        为每个生成实例创建动态配置，增加多样性
        
        Args:
            instance_id: 当前实例ID
            total_instances: 总实例数
            
        Returns:
            动态配置字典
        """
        # η参数多样化：在基础值周围变化
        base_eta = self.config.eta
        eta_variations = [
            base_eta * 0.5,   # 更保守：更相似
            base_eta * 0.8,   # 略保守
            base_eta,         # 默认值
            base_eta * 1.5,   # 略激进：更创新
            base_eta * 2.0,   # 更激进
            base_eta * 3.0    # 极激进：很不同
        ]
        
        # 根据实例ID选择η值，确保覆盖不同的范围
        eta_index = instance_id % len(eta_variations)
        dynamic_eta = eta_variations[eta_index]
        
        # 添加小幅随机扰动
        eta_noise = np.random.uniform(-0.02, 0.02)
        dynamic_eta = max(0.01, min(0.8, dynamic_eta + eta_noise))
        
        # 温度参数多样化
        base_temp = getattr(self.model.config, 'temperature', 1.0)
        temp_variations = [0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0]
        temp_index = (instance_id * 2) % len(temp_variations)  # 不同的循环模式
        dynamic_temperature = temp_variations[temp_index]
        
        # 添加温度随机噪声
        temp_noise = np.random.uniform(-0.1, 0.1)
        dynamic_temperature = max(0.1, min(5.0, dynamic_temperature + temp_noise))
        
        # 约束选择策略多样化
        strategies = [
            'random',           # 随机选择
            'degree_based',     # 基于度数选择
            'centrality_based', # 基于中心性选择
            'progressive',      # 渐进式选择
            'inverse_degree'    # 反向度数选择
        ]
        strategy_index = (instance_id * 3) % len(strategies)  # 又一个不同的循环
        constraint_strategy = strategies[strategy_index]
        
        # 构建动态配置
        dynamic_config = {
            'eta': dynamic_eta,
            'temperature': dynamic_temperature,
            'constraint_selection_strategy': constraint_strategy,
            'instance_variation_id': instance_id,
            'randomization_seed': int(time.time() * 1000000 + instance_id) % 2**31
        }
        
        logger.debug(f"实例{instance_id}动态配置: η={dynamic_eta:.3f}, temp={dynamic_temperature:.3f}, strategy={constraint_strategy}")
        
        return dynamic_config
    
    def _compute_distribution_similarity(self, dist1: torch.Tensor, dist2: torch.Tensor) -> float:
        """计算两个分布的相似度（数值稳定性增强版）"""
        try:
            # 数值稳定性检查
            if not (torch.isfinite(dist1).all() and torch.isfinite(dist2).all()):
                logger.warning("分布包含非有限值")
                return 0.5
            
            # 填充到相同长度
            max_len = max(len(dist1), len(dist2))
            if len(dist1) < max_len:
                dist1 = F.pad(dist1, (0, max_len - len(dist1)))
            if len(dist2) < max_len:
                dist2 = F.pad(dist2, (0, max_len - len(dist2)))
            
            # 检查是否为零分布
            sum1 = dist1.sum().float()
            sum2 = dist2.sum().float()
            
            if sum1 < 1e-8 and sum2 < 1e-8:
                return 1.0  # 两个都是零分布，认为相似
            elif sum1 < 1e-8 or sum2 < 1e-8:
                return 0.0  # 一个是零分布，一个不是
            
            # 归一化为概率分布
            dist1_norm = dist1.float() / sum1
            dist2_norm = dist2.float() / sum2
            
            # 计算多种相似度指标的平均值
            similarities = []
            
            # 1. 余弦相似度（数值稳定版）
            try:
                norm1 = torch.norm(dist1_norm)
                norm2 = torch.norm(dist2_norm)
                
                if norm1 < 1e-8 or norm2 < 1e-8:
                    cosine_sim = 1.0 if (norm1 < 1e-8 and norm2 < 1e-8) else 0.0
                else:
                    cosine_sim = F.cosine_similarity(
                        dist1_norm.unsqueeze(0), dist2_norm.unsqueeze(0)
                    ).item()
                    
                    if not np.isfinite(cosine_sim):
                        cosine_sim = 0.5
                    else:
                        cosine_sim = max(0.0, min(1.0, cosine_sim))
                        
                similarities.append(cosine_sim)
            except Exception as e:
                logger.debug(f"余弦相似度计算失败: {e}")
                similarities.append(0.5)
            
            # 2. KL散度相似度（数值稳定版）
            try:
                # 添加平滑项以避免log(0)
                epsilon = 1e-10
                dist1_smooth = dist1_norm + epsilon
                dist2_smooth = dist2_norm + epsilon
                
                kl_div = F.kl_div(
                    torch.log(dist1_smooth), dist2_smooth, reduction='sum'
                ).item()
                
                if not np.isfinite(kl_div) or kl_div < 0:
                    kl_sim = 0.5
                else:
                    kl_sim = np.exp(-min(kl_div, 10.0))  # 限制最大KL散度
                    
                similarities.append(kl_sim)
            except Exception as e:
                logger.debug(f"KL散度计算失败: {e}")
                similarities.append(0.5)
            
            # 3. JS散度相似度（数值稳定版）
            try:
                epsilon = 1e-10
                dist1_smooth = dist1_norm + epsilon
                dist2_smooth = dist2_norm + epsilon
                m = (dist1_smooth + dist2_smooth) / 2
                
                js_div1 = F.kl_div(torch.log(dist1_smooth), m, reduction='sum')
                js_div2 = F.kl_div(torch.log(dist2_smooth), m, reduction='sum')
                js_div = 0.5 * (js_div1 + js_div2).item()
                
                if not np.isfinite(js_div) or js_div < 0:
                    js_sim = 0.5
                else:
                    js_sim = np.exp(-min(js_div, 10.0))  # 限制最大JS散度
                    
                similarities.append(js_sim)
            except Exception as e:
                logger.debug(f"JS散度计算失败: {e}")
                similarities.append(0.5)
            
            # 返回多种相似度的平均值
            if similarities:
                final_sim = sum(similarities) / len(similarities)
                return max(0.0, min(1.0, final_sim))
            else:
                return 0.5
            
        except Exception as e:
            logger.warning(f"分布相似度计算异常: {e}")
            return 0.5
    
    def _compute_spectral_similarity(self, source_data: HeteroData, generated_data: HeteroData) -> Dict[str, float]:
        """计算图谱特性相似度"""
        spectral_sim = {}
        
        try:
            import networkx as nx
            from scipy.sparse import csr_matrix
            from scipy.sparse.linalg import eigsh
            
            # 转换为networkx图进行谱分析
            def to_networkx_bipartite(data):
                G = nx.Graph()
                edges = data[('constraint', 'connects', 'variable')].edge_index.cpu().numpy()
                n_constraints = data['constraint'].x.size(0)
                
                # 添加节点（带类型标识）
                for i in range(n_constraints):
                    G.add_node(f'c_{i}', bipartite=0)
                for i in range(data['variable'].x.size(0)):
                    G.add_node(f'v_{i}', bipartite=1)
                
                # 添加边
                for i in range(edges.shape[1]):
                    G.add_edge(f'c_{edges[0, i]}', f'v_{edges[1, i]}')
                
                return G
            
            source_graph = to_networkx_bipartite(source_data)
            generated_graph = to_networkx_bipartite(generated_data)
            
            # 1. 拉普拉斯特征值相似度
            if len(source_graph.nodes()) > 1 and len(generated_graph.nodes()) > 1:
                source_laplacian = nx.laplacian_matrix(source_graph).astype(float)
                generated_laplacian = nx.laplacian_matrix(generated_graph).astype(float)
                
                # 计算前几个特征值
                k = min(10, min(source_laplacian.shape[0], generated_laplacian.shape[0]) - 1)
                if k > 0:
                    source_eigenvals = eigsh(source_laplacian, k=k, which='SM', return_eigenvectors=False)
                    generated_eigenvals = eigsh(generated_laplacian, k=k, which='SM', return_eigenvectors=False)
                    
                    # 归一化特征值
                    source_eigenvals = np.sort(source_eigenvals)
                    generated_eigenvals = np.sort(generated_eigenvals)
                    
                    # 计算特征值分布的相似度
                    eigenval_sim = 1.0 - np.mean(np.abs(source_eigenvals - generated_eigenvals)) / (
                        np.mean(np.abs(source_eigenvals)) + 1e-8
                    )
                    spectral_sim['eigenvalue_similarity'] = max(0.0, min(1.0, eigenval_sim))
                else:
                    spectral_sim['eigenvalue_similarity'] = 0.5
            else:
                spectral_sim['eigenvalue_similarity'] = 0.5
            
            spectral_sim['spectral_similarity'] = spectral_sim.get('eigenvalue_similarity', 0.5)
            
        except Exception as e:
            logger.warning(f"图谱分析需要networkx和scipy: {e}")
            spectral_sim = {'spectral_similarity': 0.5, 'eigenvalue_similarity': 0.5}
        
        return spectral_sim
    
    def _compute_clustering_similarity(self, source_data: HeteroData, generated_data: HeteroData) -> float:
        """计算聚类系数相似度"""
        try:
            import networkx as nx
            
            def compute_bipartite_clustering(data):
                G = nx.Graph()
                edges = data[('constraint', 'connects', 'variable')].edge_index.cpu().numpy()
                
                # 构建图
                for i in range(edges.shape[1]):
                    G.add_edge(f'c_{edges[0, i]}', f'v_{edges[1, i]}')
                
                if len(G.nodes()) < 3:
                    return 0.0
                
                # 对于二分图，计算约束节点的聚类系数
                clustering_coeffs = []
                for node in G.nodes():
                    if node.startswith('c_'):  # 约束节点
                        neighbors = list(G.neighbors(node))
                        if len(neighbors) < 2:
                            clustering_coeffs.append(0.0)
                            continue
                        
                        # 计算邻居间的连接数
                        possible_edges = len(neighbors) * (len(neighbors) - 1) // 2
                        actual_edges = 0
                        for i in range(len(neighbors)):
                            for j in range(i + 1, len(neighbors)):
                                if G.has_edge(neighbors[i], neighbors[j]):
                                    actual_edges += 1
                        
                        clustering_coeffs.append(actual_edges / possible_edges if possible_edges > 0 else 0.0)
                
                return np.mean(clustering_coeffs) if clustering_coeffs else 0.0
            
            source_clustering = compute_bipartite_clustering(source_data)
            generated_clustering = compute_bipartite_clustering(generated_data)
            
            # 计算聚类系数的相似度
            clustering_sim = 1.0 - abs(source_clustering - generated_clustering)
            return max(0.0, min(1.0, clustering_sim))
            
        except Exception as e:
            logger.warning(f"聚类系数计算失败: {e}")
            return 0.5
    
    def _compute_feature_similarity_detailed(self, features1: torch.Tensor, features2: torch.Tensor) -> List[float]:
        """计算特征的分维度相似度（数值稳定性增强版）"""
        similarities = []
        
        try:
            # 计算每个特征维度的相似度
            for dim in range(min(features1.size(1), features2.size(1))):
                dim1_values = features1[:, dim]
                dim2_values = features2[:, dim]
                
                # 数值稳定性检查
                dim1_numpy = dim1_values.cpu().numpy()
                dim2_numpy = dim2_values.cpu().numpy()
                
                # 检查是否包含NaN或无穷值
                if not (np.isfinite(dim1_numpy).all() and np.isfinite(dim2_numpy).all()):
                    logger.warning(f"特征维度 {dim} 包含非有限值，跳过该维度")
                    similarities.append(0.5)
                    continue
                
                # 1. 余弦相似度（数值稳定版）
                try:
                    # 检查向量是否为零向量
                    norm1 = torch.norm(dim1_values)
                    norm2 = torch.norm(dim2_values)
                    
                    if norm1 < 1e-8 or norm2 < 1e-8:
                        # 零向量情况：如果两个都是零向量则相似度为1，否则为0
                        cosine_sim = 1.0 if (norm1 < 1e-8 and norm2 < 1e-8) else 0.0
                    else:
                        cosine_sim = F.cosine_similarity(
                            dim1_values.unsqueeze(0), dim2_values.unsqueeze(0)
                        ).item()
                        
                        # 确保余弦相似度在有效范围内
                        if not np.isfinite(cosine_sim):
                            cosine_sim = 0.0
                        else:
                            cosine_sim = max(-1.0, min(1.0, cosine_sim))
                            # 转换到 [0, 1] 范围
                            cosine_sim = (cosine_sim + 1.0) / 2.0
                            
                except Exception as e:
                    logger.debug(f"余弦相似度计算失败 (dim {dim}): {e}")
                    cosine_sim = 0.5
                
                # 2. 皮尔逊相关系数（数值稳定版）
                try:
                    # 检查标准差是否为零
                    std1 = np.std(dim1_numpy)
                    std2 = np.std(dim2_numpy)
                    
                    if std1 < 1e-8 or std2 < 1e-8:
                        # 标准差为零的情况：检查均值是否相等
                        mean1 = np.mean(dim1_numpy)
                        mean2 = np.mean(dim2_numpy)
                        
                        if abs(mean1 - mean2) < 1e-8:
                            pearson_corr = 1.0  # 完全相同的常数值
                        else:
                            pearson_corr = 0.0  # 不同的常数值
                    else:
                        # 标准相关系数计算
                        corr_matrix = np.corrcoef(dim1_numpy, dim2_numpy)
                        
                        if corr_matrix.shape == (2, 2):
                            pearson_corr = corr_matrix[0, 1]
                        else:
                            pearson_corr = 0.0
                        
                        # 检查结果的有效性
                        if not np.isfinite(pearson_corr):
                            pearson_corr = 0.0
                        else:
                            # 转换到 [0, 1] 范围
                            pearson_corr = (pearson_corr + 1.0) / 2.0
                            
                except Exception as e:
                    logger.debug(f"皮尔逊相关系数计算失败 (dim {dim}): {e}")
                    pearson_corr = 0.5
                
                # 3. 分布相似度（基于统计特征，数值稳定版）
                try:
                    mean1 = dim1_values.mean().item()
                    mean2 = dim2_values.mean().item()
                    std1 = dim1_values.std().item()
                    std2 = dim2_values.std().item()
                    
                    # 均值相似度
                    if abs(mean1) + abs(mean2) < 1e-8:
                        mean_sim = 1.0  # 两个均值都接近0
                    else:
                        mean_sim = 1.0 - abs(mean1 - mean2) / (abs(mean1) + abs(mean2) + 1e-8)
                    
                    # 标准差相似度  
                    if std1 + std2 < 1e-8:
                        std_sim = 1.0  # 两个标准差都接近0
                    else:
                        std_sim = 1.0 - abs(std1 - std2) / (std1 + std2 + 1e-8)
                    
                    # 确保结果在有效范围内
                    mean_sim = max(0.0, min(1.0, mean_sim))
                    std_sim = max(0.0, min(1.0, std_sim))
                    
                except Exception as e:
                    logger.debug(f"分布相似度计算失败 (dim {dim}): {e}")
                    mean_sim = 0.5
                    std_sim = 0.5
                
                # 4. 综合相似度
                try:
                    dim_similarity = (cosine_sim + pearson_corr + mean_sim + std_sim) / 4.0
                    dim_similarity = max(0.0, min(1.0, dim_similarity))
                    
                    if not np.isfinite(dim_similarity):
                        dim_similarity = 0.5
                        
                    similarities.append(dim_similarity)
                    
                except Exception as e:
                    logger.debug(f"综合相似度计算失败 (dim {dim}): {e}")
                    similarities.append(0.5)
                
        except Exception as e:
            logger.warning(f"特征相似度计算异常: {e}")
            # 返回默认相似度
            similarities = [0.5] * min(features1.size(1), features2.size(1))
        
        # 最终有效性检查
        if not similarities:
            similarities = [0.5] * min(features1.size(1), features2.size(1))
        
        # 确保所有相似度都在有效范围内
        similarities = [max(0.0, min(1.0, sim)) for sim in similarities]
        
        return similarities
    
    def _compute_sparsity_pattern_similarity(self, source_data: HeteroData, generated_data: HeteroData) -> float:
        """计算稀疏性模式相似度"""
        try:
            source_edges = source_data[('constraint', 'connects', 'variable')].edge_index
            generated_edges = generated_data[('constraint', 'connects', 'variable')].edge_index
            
            n_constraints = source_data['constraint'].x.size(0)
            n_variables = source_data['variable'].x.size(0)
            
            # 构建稀疏性模式向量（每个约束的稀疏度）
            source_sparsity = torch.zeros(n_constraints)
            generated_sparsity = torch.zeros(n_constraints)
            
            for c in range(n_constraints):
                source_connections = (source_edges[0] == c).sum().item()
                generated_connections = (generated_edges[0] == c).sum().item()
                
                source_sparsity[c] = source_connections / n_variables
                generated_sparsity[c] = generated_connections / n_variables
            
            # 计算稀疏性模式的相似度
            sparsity_sim = F.cosine_similarity(
                source_sparsity.unsqueeze(0), generated_sparsity.unsqueeze(0)
            ).item()
            
            return max(0.0, min(1.0, sparsity_sim))
            
        except Exception as e:
            logger.warning(f"稀疏性模式计算异常: {e}")
            return 0.5
    
    def _compute_connectivity_pattern_similarity(self, source_data: HeteroData, generated_data: HeteroData) -> float:
        """计算连接模式相似度"""
        try:
            source_edges = source_data[('constraint', 'connects', 'variable')].edge_index
            generated_edges = generated_data[('constraint', 'connects', 'variable')].edge_index
            
            # 计算邻接矩阵的相似性
            n_constraints = source_data['constraint'].x.size(0)
            n_variables = source_data['variable'].x.size(0)
            
            # 构建二分图的块矩阵表示
            source_adj = torch.zeros(n_constraints, n_variables)
            generated_adj = torch.zeros(n_constraints, n_variables)
            
            source_adj[source_edges[0], source_edges[1]] = 1
            generated_adj[generated_edges[0], generated_edges[1]] = 1
            
            # 计算Jaccard相似度
            intersection = (source_adj * generated_adj).sum().item()
            union = ((source_adj + generated_adj) > 0).sum().item()
            
            jaccard_sim = intersection / union if union > 0 else 0.0
            
            return max(0.0, min(1.0, jaccard_sim))
            
        except Exception as e:
            logger.warning(f"连接模式计算异常: {e}")
            return 0.5
    
    def validate_milp_instances(self, source_data: HeteroData, generated_data: HeteroData, 
                               source_milp_instance=None, generated_milp_instance=None) -> Dict[str, Any]:
        """
        验证MILP实例的求解特性（最重要的质量评估）
        
        Args:
            source_data: 源图数据
            generated_data: 生成的图数据  
            source_milp_instance: 源MILP实例（可选）
            generated_milp_instance: 生成的MILP实例（可选）
            
        Returns:
            求解验证结果字典
        """
        validation_results = {
            'source_solve_results': {},
            'generated_solve_results': {},
            'comparative_analysis': {},
            'validation_summary': {}
        }
        
        try:
            # 如果没有提供MILP实例，从图数据重构
            if source_milp_instance is None:
                source_milp_instance = self._reconstruct_milp_from_graph(source_data)
            if generated_milp_instance is None:
                generated_milp_instance = self._reconstruct_milp_from_graph(generated_data)
            
            # 1. 求解源实例
            logger.info("求解源MILP实例...")
            source_solve_results = self._solve_milp_instance(source_milp_instance, "source")
            validation_results['source_solve_results'] = source_solve_results
            
            # 2. 求解生成实例  
            logger.info("求解生成的MILP实例...")
            generated_solve_results = self._solve_milp_instance(generated_milp_instance, "generated")
            validation_results['generated_solve_results'] = generated_solve_results
            
            # 3. 比较分析
            logger.info("进行求解特性比较分析...")
            comparative_analysis = self._compare_solve_results(source_solve_results, generated_solve_results)
            validation_results['comparative_analysis'] = comparative_analysis
            
            # 4. 生成验证摘要
            validation_summary = self._generate_validation_summary(
                source_solve_results, generated_solve_results, comparative_analysis
            )
            validation_results['validation_summary'] = validation_summary
            
            logger.info("MILP实例求解验证完成")
            
        except Exception as e:
            logger.error(f"MILP求解验证失败: {e}")
            validation_results['error'] = str(e)
            validation_results['validation_summary'] = {
                'is_valid': False,
                'error_message': str(e)
            }
        
        return validation_results
    
    def _reconstruct_milp_from_graph(self, graph_data: HeteroData) -> Dict[str, Any]:
        """从图数据重构MILP实例"""
        try:
            edges = graph_data[('constraint', 'connects', 'variable')].edge_index
            n_constraints = graph_data['constraint'].x.size(0)
            n_variables = graph_data['variable'].x.size(0)
            
            # 构建约束矩阵
            import scipy.sparse as sp
            constraint_matrix = sp.lil_matrix((n_constraints, n_variables))
            
            # 设置非零元素（使用图数据中的边权重，如果有的话）
            for i in range(edges.size(1)):
                constraint_idx = edges[0, i].item()
                variable_idx = edges[1, i].item()
                
                # 如果有边权重，使用边权重；否则使用默认值
                if hasattr(graph_data[('constraint', 'connects', 'variable')], 'edge_attr'):
                    weight = graph_data[('constraint', 'connects', 'variable')].edge_attr[i].item()
                else:
                    weight = 1.0  # 默认权重
                
                constraint_matrix[constraint_idx, variable_idx] = weight
            
            # 从节点特征中提取MILP参数
            constraint_features = graph_data['constraint'].x.cpu().numpy()
            variable_features = graph_data['variable'].x.cpu().numpy()
            
            # 提取目标函数系数（假设在变量特征的某一维）
            objective_coeffs = variable_features[:, 0] if variable_features.shape[1] > 0 else np.ones(n_variables)
            
            # 提取右端项（假设在约束特征的某一维）  
            rhs_values = constraint_features[:, 0] if constraint_features.shape[1] > 0 else np.ones(n_constraints)
            
            # 变量界限（假设在变量特征中）
            if variable_features.shape[1] > 2:
                lower_bounds = variable_features[:, 1] 
                upper_bounds = variable_features[:, 2]
            else:
                lower_bounds = np.zeros(n_variables)
                upper_bounds = np.ones(n_variables) * 1000  # 大的上界
            
            milp_instance = {
                'constraint_matrix': constraint_matrix.tocsr(),
                'objective_coeffs': objective_coeffs,
                'rhs_values': rhs_values,
                'lower_bounds': lower_bounds,
                'upper_bounds': upper_bounds,
                'variable_types': ['continuous'] * n_variables,  # 简化：假设都是连续变量
                'constraint_senses': ['<='] * n_constraints,      # 简化：假设都是<=约束
                'n_constraints': n_constraints,
                'n_variables': n_variables
            }
            
            return milp_instance
            
        except Exception as e:
            logger.error(f"从图数据重构MILP实例失败: {e}")
            raise
    
    def _solve_milp_instance(self, milp_instance: Dict[str, Any], instance_name: str) -> Dict[str, Any]:
        """使用多种求解器求解MILP实例"""
        solve_results = {
            'instance_name': instance_name,
            'solvers_used': [],
            'solve_attempts': {},
            'best_result': None,
            'solving_statistics': {}
        }
        
        # 尝试使用的求解器列表（按优先级排序）
        solvers_to_try = ['CVXPY_DEFAULT', 'CVXPY_ECOS', 'CVXPY_SCS']
        
        try:
            # 添加商业求解器（如果可用）
            try:
                import gurobipy
                solvers_to_try.insert(0, 'GUROBI')
            except ImportError:
                pass
            
            try:
                import cplex
                solvers_to_try.insert(0, 'CPLEX')
            except ImportError:
                pass
        except:
            pass
        
        best_solve_time = float('inf')
        best_objective = None
        
        for solver_name in solvers_to_try[:3]:  # 限制尝试的求解器数量
            try:
                logger.debug(f"尝试使用求解器: {solver_name}")
                solve_attempt = self._solve_with_specific_solver(milp_instance, solver_name)
                solve_results['solve_attempts'][solver_name] = solve_attempt
                solve_results['solvers_used'].append(solver_name)
                
                # 更新最佳结果
                if solve_attempt['status'] == 'optimal' and solve_attempt['solve_time'] < best_solve_time:
                    best_solve_time = solve_attempt['solve_time']
                    best_objective = solve_attempt['objective_value']
                    solve_results['best_result'] = solve_attempt
                    solve_results['best_result']['solver'] = solver_name
                
            except Exception as e:
                logger.warning(f"求解器 {solver_name} 失败: {e}")
                solve_results['solve_attempts'][solver_name] = {
                    'status': 'error',
                    'error_message': str(e),
                    'solve_time': None,
                    'objective_value': None
                }
        
        # 生成求解统计
        solve_results['solving_statistics'] = self._compute_solving_statistics(solve_results)
        
        return solve_results
    
    def _solve_with_specific_solver(self, milp_instance: Dict[str, Any], solver_name: str) -> Dict[str, Any]:
        """使用特定求解器求解"""
        import time
        start_time = time.time()
        
        try:
            if solver_name.startswith('CVXPY'):
                return self._solve_with_cvxpy(milp_instance, solver_name)
            elif solver_name == 'GUROBI':
                return self._solve_with_gurobi(milp_instance)
            elif solver_name == 'CPLEX':
                return self._solve_with_cplex(milp_instance)
            else:
                raise ValueError(f"不支持的求解器: {solver_name}")
                
        except Exception as e:
            solve_time = time.time() - start_time
            return {
                'status': 'error',
                'error_message': str(e),
                'solve_time': solve_time,
                'objective_value': None,
                'solution': None
            }
    
    def _solve_with_cvxpy(self, milp_instance: Dict[str, Any], solver_name: str) -> Dict[str, Any]:
        """使用CVXPY求解"""
        import cvxpy as cp
        import time
        
        start_time = time.time()
        
        try:
            # 创建变量
            n_vars = milp_instance['n_variables']
            x = cp.Variable(n_vars)
            
            # 目标函数
            objective = cp.Minimize(milp_instance['objective_coeffs'] @ x)
            
            # 约束
            constraints = []
            
            # 主要约束 Ax <= b
            A = milp_instance['constraint_matrix']
            b = milp_instance['rhs_values']
            constraints.append(A @ x <= b)
            
            # 变量界限
            constraints.append(x >= milp_instance['lower_bounds'])
            constraints.append(x <= milp_instance['upper_bounds'])
            
            # 创建问题
            problem = cp.Problem(objective, constraints)
            
            # 选择求解器
            if solver_name == 'CVXPY_ECOS':
                solver = cp.ECOS
            elif solver_name == 'CVXPY_SCS':
                solver = cp.SCS
            else:
                solver = None  # 使用默认求解器
            
            # 求解
            if solver:
                problem.solve(solver=solver, verbose=False)
            else:
                problem.solve(verbose=False)
            
            solve_time = time.time() - start_time
            
            # 检查状态
            if problem.status == cp.OPTIMAL:
                return {
                    'status': 'optimal',
                    'objective_value': problem.value,
                    'solve_time': solve_time,
                    'solution': x.value,
                    'num_variables': n_vars,
                    'num_constraints': milp_instance['n_constraints']
                }
            elif problem.status == cp.INFEASIBLE:
                return {
                    'status': 'infeasible',
                    'objective_value': None,
                    'solve_time': solve_time,
                    'solution': None,
                    'num_variables': n_vars,
                    'num_constraints': milp_instance['n_constraints']
                }
            else:
                return {
                    'status': 'unknown',
                    'objective_value': problem.value,
                    'solve_time': solve_time,
                    'solution': x.value if x.value is not None else None,
                    'num_variables': n_vars,
                    'num_constraints': milp_instance['n_constraints']
                }
                
        except Exception as e:
            solve_time = time.time() - start_time
            raise Exception(f"CVXPY求解失败: {e}")
    
    def _solve_with_gurobi(self, milp_instance: Dict[str, Any]) -> Dict[str, Any]:
        """使用Gurobi求解（如果可用）"""
        try:
            import gurobipy as gp
            from gurobipy import GRB
            import time
            
            start_time = time.time()
            
            # 创建模型
            model = gp.Model("milp_validation")
            model.setParam('OutputFlag', 0)  # 静默模式
            model.setParam('TimeLimit', 60)  # 60秒时间限制
            
            # 创建变量
            n_vars = milp_instance['n_variables']
            x = model.addVars(n_vars, lb=milp_instance['lower_bounds'], 
                             ub=milp_instance['upper_bounds'], name="x")
            
            # 设置目标函数
            obj_coeffs = milp_instance['objective_coeffs']
            model.setObjective(gp.quicksum(obj_coeffs[i] * x[i] for i in range(n_vars)), GRB.MINIMIZE)
            
            # 添加约束
            A = milp_instance['constraint_matrix']
            b = milp_instance['rhs_values']
            
            for i in range(milp_instance['n_constraints']):
                row = A.getrow(i)
                lhs = gp.quicksum(row.data[j] * x[row.indices[j]] for j in range(len(row.data)))
                model.addConstr(lhs <= b[i], f"constraint_{i}")
            
            # 求解
            model.optimize()
            solve_time = time.time() - start_time
            
            if model.status == GRB.OPTIMAL:
                solution = [x[i].x for i in range(n_vars)]
                return {
                    'status': 'optimal',
                    'objective_value': model.objVal,
                    'solve_time': solve_time,
                    'solution': solution,
                    'num_variables': n_vars,
                    'num_constraints': milp_instance['n_constraints'],
                    'mip_gap': model.MIPGap if hasattr(model, 'MIPGap') else 0.0
                }
            elif model.status == GRB.INFEASIBLE:
                return {
                    'status': 'infeasible',
                    'objective_value': None,
                    'solve_time': solve_time,
                    'solution': None,
                    'num_variables': n_vars,
                    'num_constraints': milp_instance['n_constraints']
                }
            else:
                return {
                    'status': 'unknown',
                    'objective_value': model.objVal if model.objVal != GRB.INFINITY else None,
                    'solve_time': solve_time,
                    'solution': None,
                    'num_variables': n_vars,
                    'num_constraints': milp_instance['n_constraints']
                }
                
        except Exception as e:
            raise Exception(f"Gurobi求解失败: {e}")
    
    def _solve_with_cplex(self, milp_instance: Dict[str, Any]) -> Dict[str, Any]:
        """使用CPLEX求解（占位符实现）"""
        # 这里可以实现CPLEX求解器集成
        # 当前返回未实现状态
        return {
            'status': 'not_implemented',
            'error_message': 'CPLEX求解器集成尚未实现',
            'solve_time': 0.0,
            'objective_value': None,
            'solution': None
        }
    
    def _compute_solving_statistics(self, solve_results: Dict[str, Any]) -> Dict[str, Any]:
        """计算求解统计信息"""
        statistics = {
            'num_solvers_tried': len(solve_results['solvers_used']),
            'successful_solvers': [],
            'failed_solvers': [],
            'optimal_solutions_found': 0,
            'infeasible_results': 0,
            'average_solve_time': 0.0,
            'solve_time_variance': 0.0
        }
        
        solve_times = []
        
        for solver, result in solve_results['solve_attempts'].items():
            if result['status'] == 'optimal':
                statistics['successful_solvers'].append(solver)
                statistics['optimal_solutions_found'] += 1
                if result['solve_time'] is not None:
                    solve_times.append(result['solve_time'])
            elif result['status'] == 'infeasible':
                statistics['infeasible_results'] += 1
            else:
                statistics['failed_solvers'].append(solver)
        
        if solve_times:
            statistics['average_solve_time'] = np.mean(solve_times)
            statistics['solve_time_variance'] = np.var(solve_times)
        
        return statistics
    
    def _compare_solve_results(self, source_results: Dict[str, Any], 
                              generated_results: Dict[str, Any]) -> Dict[str, Any]:
        """比较两个实例的求解结果"""
        comparison = {
            'feasibility_consistency': False,
            'objective_value_comparison': {},
            'solve_time_comparison': {},
            'difficulty_assessment': {},
            'quality_score': 0.0
        }
        
        try:
            source_best = source_results.get('best_result')
            generated_best = generated_results.get('best_result')
            
            # 1. 可行性一致性检查
            if source_best and generated_best:
                source_feasible = source_best['status'] == 'optimal'
                generated_feasible = generated_best['status'] == 'optimal'
                comparison['feasibility_consistency'] = source_feasible == generated_feasible
                
                # 2. 目标值比较（如果都可行）
                if source_feasible and generated_feasible:
                    source_obj = source_best['objective_value']
                    generated_obj = generated_best['objective_value']
                    
                    comparison['objective_value_comparison'] = {
                        'source_objective': source_obj,
                        'generated_objective': generated_obj,
                        'relative_difference': abs(source_obj - generated_obj) / (abs(source_obj) + 1e-8),
                        'objectives_similar': abs(source_obj - generated_obj) / (abs(source_obj) + 1e-8) < 0.5
                    }
                    
                    # 3. 求解时间比较
                    source_time = source_best['solve_time']
                    generated_time = generated_best['solve_time']
                    
                    comparison['solve_time_comparison'] = {
                        'source_solve_time': source_time,
                        'generated_solve_time': generated_time,
                        'time_ratio': generated_time / (source_time + 1e-8),
                        'similar_difficulty': 0.1 <= (generated_time / (source_time + 1e-8)) <= 10.0
                    }
                    
                    # 4. 难度评估
                    comparison['difficulty_assessment'] = {
                        'source_complexity_score': self._assess_instance_complexity(source_results),
                        'generated_complexity_score': self._assess_instance_complexity(generated_results),
                        'complexity_preserved': True  # 简化评估
                    }
                    
                    # 5. 综合质量评分
                    quality_components = [
                        1.0 if comparison['feasibility_consistency'] else 0.0,
                        0.8 if comparison['objective_value_comparison']['objectives_similar'] else 0.2,
                        0.8 if comparison['solve_time_comparison']['similar_difficulty'] else 0.2,
                        0.8 if comparison['difficulty_assessment']['complexity_preserved'] else 0.2
                    ]
                    comparison['quality_score'] = np.mean(quality_components)
            
        except Exception as e:
            logger.warning(f"求解结果比较失败: {e}")
            comparison['error'] = str(e)
            comparison['quality_score'] = 0.0
        
        return comparison
    
    def _assess_instance_complexity(self, solve_results: Dict[str, Any]) -> float:
        """评估MILP实例的复杂度"""
        try:
            best_result = solve_results.get('best_result')
            if not best_result:
                return 0.5
            
            # 基于求解时间、变量数、约束数等因素评估复杂度
            solve_time = best_result.get('solve_time', 0.0)
            num_vars = best_result.get('num_variables', 0)
            num_constraints = best_result.get('num_constraints', 0)
            
            # 简化的复杂度评分
            time_score = min(1.0, solve_time / 10.0)  # 10秒为高复杂度基准
            size_score = min(1.0, (num_vars + num_constraints) / 10000.0)  # 10k为高复杂度基准
            
            complexity_score = (time_score + size_score) / 2.0
            return complexity_score
            
        except Exception as e:
            logger.warning(f"复杂度评估失败: {e}")
            return 0.5
    
    def _generate_validation_summary(self, source_results: Dict[str, Any], 
                                   generated_results: Dict[str, Any],
                                   comparative_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """生成验证摘要"""
        summary = {
            'is_valid': False,
            'validation_score': 0.0,
            'key_findings': [],
            'recommendations': [],
            'detailed_assessment': {}
        }
        
        try:
            # 基本验证检查
            source_solvable = source_results.get('best_result', {}).get('status') == 'optimal'
            generated_solvable = generated_results.get('best_result', {}).get('status') == 'optimal'
            
            summary['detailed_assessment'] = {
                'source_solvable': source_solvable,
                'generated_solvable': generated_solvable,
                'feasibility_consistent': comparative_analysis.get('feasibility_consistency', False),
                'quality_score': comparative_analysis.get('quality_score', 0.0)
            }
            
            # 关键发现
            if source_solvable and generated_solvable:
                summary['key_findings'].append("✅ 源实例和生成实例都可以成功求解")
                if comparative_analysis.get('feasibility_consistency'):
                    summary['key_findings'].append("✅ 可行性状态一致")
            elif not source_solvable and not generated_solvable:
                summary['key_findings'].append("⚠️ 源实例和生成实例都不可行（可能是期望的）")
            else:
                summary['key_findings'].append("❌ 可行性状态不一致，生成质量有问题")
            
            # 验证分数计算
            quality_score = comparative_analysis.get('quality_score', 0.0)
            summary['validation_score'] = quality_score
            
            # 验证判定
            if quality_score >= 0.7:
                summary['is_valid'] = True
                summary['key_findings'].append(f"🎉 生成质量优秀 (得分: {quality_score:.3f})")
                summary['recommendations'].append("生成的MILP实例质量良好，可用于研究和测试")
            elif quality_score >= 0.5:
                summary['is_valid'] = True
                summary['key_findings'].append(f"⚠️ 生成质量一般 (得分: {quality_score:.3f})")
                summary['recommendations'].append("生成的实例基本可用，但建议进一步优化模型")
            else:
                summary['is_valid'] = False
                summary['key_findings'].append(f"❌ 生成质量不佳 (得分: {quality_score:.3f})")
                summary['recommendations'].append("需要重新训练模型或调整生成参数")
            
        except Exception as e:
            logger.error(f"验证摘要生成失败: {e}")
            summary['error'] = str(e)
            summary['key_findings'].append(f"❌ 验证过程出现错误: {e}")
        
        return summary


def create_inference_engine(model: G2MILPGenerator,
                          config: InferenceConfig = None) -> G2MILPInference:
    """
    创建G2MILP推理器的工厂函数
    
    Args:
        model: 训练好的G2MILP生成器
        config: 推理配置
        
    Returns:
        G2MILP推理器实例
    """
    return G2MILPInference(model, config)


if __name__ == "__main__":
    # 测试代码
    print("G2MILP推理模块测试")
    print("=" * 40)
    
    # 创建测试配置
    inference_config = InferenceConfig(
        eta=0.1,
        temperature=1.0,
        sample_from_prior=True,
        compute_similarity_metrics=True
    )
    
    print(f"推理配置:")
    print(f"- η (eta): {inference_config.eta}")
    print(f"- Temperature: {inference_config.temperature}")
    print(f"- Sample from prior: {inference_config.sample_from_prior}")
    print(f"- Device: {inference_config.device}")
    print("推理器配置创建成功!")
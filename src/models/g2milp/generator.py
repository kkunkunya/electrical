"""
G2MILP生成器模块
G2MILP Generator Module

G2MILP的主生成器，集成编码器、解码器和遮盖过程，实现完整的MILP实例生成流程。
根据论文3.4节的训练和推理过程实现。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
import numpy as np
import logging
from typing import Dict, Tuple, Optional, List, Any
from dataclasses import dataclass
import copy
from pathlib import Path
import json
from datetime import datetime

from .encoder import G2MILPEncoder, EncoderConfig
from .decoder import G2MILPDecoder, DecoderConfig
from .masking import ConstraintMasker, MaskingConfig

logger = logging.getLogger(__name__)


@dataclass
class GeneratorConfig:
    """生成器配置"""
    # 网络架构
    encoder_config: EncoderConfig = None
    decoder_config: DecoderConfig = None
    masking_config: MaskingConfig = None
    
    # 损失函数权重（重新平衡，优先连接预测）
    alpha_bias: float = 0.5      # 偏置预测损失权重（降低）
    alpha_degree: float = 0.5    # 度数预测损失权重（降低）  
    alpha_logits: float = 2.0    # 连接预测损失权重（提高，核心任务）
    alpha_weights: float = 0.5   # 权重预测损失权重（降低）
    beta_kl: float = 1.0         # KL散度损失权重
    
    # 训练参数
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0
    
    # 推理参数
    eta: float = 0.1             # 遮盖比例η，控制相似度vs创新性
    num_inference_iterations: Optional[int] = None  # 推理迭代次数，None则根据η计算
    sample_from_prior: bool = True  # 推理时是否从先验分布采样
    temperature: float = 1.0     # 基础采样温度
    
    # VAE采样增强参数（大幅改进）
    use_dynamic_temperature: bool = True    # 是否使用动态温度
    temperature_range: Tuple[float, float] = (0.3, 3.0)  # 扩大温度范围
    use_spherical_sampling: bool = True     # 启用球面采样（更均匀分布）
    noise_injection_strength: float = 0.15  # 增强噪声注入强度
    
    # 推理多样性增强
    use_dynamic_eta: bool = True            # 动态η参数
    eta_range: Tuple[float, float] = (0.05, 0.4)  # η参数范围
    diversity_boost_factor: float = 1.5     # 多样性提升因子
    use_constraint_diversity: bool = True   # 约束选择多样性
    
    # 稀疏性正则化参数（增强版）
    use_sparsity_regularization: bool = True
    sparsity_weight: float = 0.05  # 提高稀疏性正则化权重
    target_sparsity: float = 0.1
    
    # 课程学习参数（新增）
    use_curriculum_learning: bool = True
    curriculum_kl_warmup_epochs: int = 200  # KL预热期延长
    curriculum_kl_annealing_epochs: int = 600  # KL退火期延长
    
    # 设备配置
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class G2MILPGenerator(nn.Module):
    """
    G2MILP生成器主类
    
    实现完整的Masked VAE范式MILP实例生成，包括：
    1. 训练阶段：重建被遮盖的约束
    2. 推理阶段：生成新的MILP实例
    """
    
    def __init__(self, 
                 constraint_feature_dim: int = 16,
                 variable_feature_dim: int = 9,
                 edge_feature_dim: int = 8,
                 config: GeneratorConfig = None):
        super().__init__()
        
        self.config = config or GeneratorConfig()
        self.constraint_feature_dim = constraint_feature_dim
        self.variable_feature_dim = variable_feature_dim
        self.edge_feature_dim = edge_feature_dim
        
        # 初始化子配置
        if self.config.encoder_config is None:
            self.config.encoder_config = EncoderConfig()
        if self.config.decoder_config is None:
            self.config.decoder_config = DecoderConfig()
        if self.config.masking_config is None:
            self.config.masking_config = MaskingConfig()
        
        # 初始化组件
        self.encoder = G2MILPEncoder(
            constraint_feature_dim=constraint_feature_dim,
            variable_feature_dim=variable_feature_dim,
            edge_feature_dim=edge_feature_dim,
            config=self.config.encoder_config
        )
        
        self.decoder = G2MILPDecoder(
            constraint_feature_dim=constraint_feature_dim,
            variable_feature_dim=variable_feature_dim,
            config=self.config.decoder_config
        )
        
        self.masker = ConstraintMasker(self.config.masking_config)
        
        # 训练状态跟踪（用于课程学习）
        self.current_epoch = 0
        self.training_step = 0
        
        # 移动到指定设备
        self.to(self.config.device)
        
        logger.info(f"G2MILP生成器初始化完成 - 设备: {self.config.device}")
    
    def forward(self, data: HeteroData, mode: str = "train") -> Dict[str, Any]:
        """
        前向传播
        
        Args:
            data: 输入的异构图数据
            mode: 模式 ("train" 或 "inference")
            
        Returns:
            结果字典
        """
        if mode == "train":
            return self._forward_train(data)
        elif mode == "inference":
            return self._forward_inference(data)
        else:
            raise ValueError(f"不支持的模式: {mode}")
    
    def _forward_train(self, data: HeteroData) -> Dict[str, Any]:
        """
        训练模式前向传播
        
        实现论文中的重建任务：给定遮盖图和潜变量，重建原始约束
        """
        # 确保所有输入数据都在正确的设备上
        data = data.to(self.config.device)
        
        # 1. 选择要遮盖的约束
        constraint_id = self.masker.select_constraint_to_mask(data)
        
        # 2. 进行遮盖
        masked_data, mask_info = self.masker.mask_constraint(data, constraint_id)
        masked_data = masked_data.to(self.config.device)
        
        
        # 3. 编码
        z_dict, mu_dict, logvar_dict = self.encoder.encode(data, sample=True)
        
        # 4. 解码
        predicted_results = self.decoder(masked_data, z_dict, constraint_id)
        
        # 5. 计算损失
        losses = self._compute_training_losses(
            data, masked_data, mask_info, predicted_results, mu_dict, logvar_dict
        )
        
        return {
            'losses': losses,
            'predicted_results': predicted_results,
            'mask_info': mask_info,
            'z_dict': z_dict,
            'mu_dict': mu_dict,
            'logvar_dict': logvar_dict
        }
    
    def _forward_inference(self, data: HeteroData) -> Dict[str, Any]:
        """
        推理模式前向传播（增强版：支持动态η和多样性策略）
        
        实现论文算法2：迭代遮盖-生成过程 + 多样性增强
        """
        # 动态确定迭代次数
        n_constraints = data['constraint'].x.size(0)
        
        if self.config.num_inference_iterations is None:
            # 使用动态η参数（如果启用）
            if getattr(self.config, 'use_dynamic_eta', False):
                eta_range = getattr(self.config, 'eta_range', (0.05, 0.4))
                dynamic_eta = np.random.uniform(eta_range[0], eta_range[1])
                logger.debug(f"使用动态η参数: {dynamic_eta:.3f}")
            else:
                dynamic_eta = self.config.eta
            
            base_iterations = int(dynamic_eta * n_constraints)
            
            # 多样性提升因子
            diversity_factor = getattr(self.config, 'diversity_boost_factor', 1.0)
            num_iterations = int(base_iterations * diversity_factor)
        else:
            num_iterations = self.config.num_inference_iterations
            dynamic_eta = self.config.eta
        
        num_iterations = max(3, num_iterations)  # 至少迭代3次，增强多样性
        
        logger.info(f"开始推理生成 - 迭代次数: {num_iterations} (η={dynamic_eta:.3f})")
        
        # 初始化生成数据
        generated_data = copy.deepcopy(data).to(self.config.device)
        generation_history = []
        
        # 迭代生成过程（增强版）
        used_constraints = set()  # 跟踪已使用的约束，增强多样性
        
        for iteration in range(num_iterations):
            logger.debug(f"推理迭代 {iteration + 1}/{num_iterations}")
            
            # 1. 智能选择约束进行遮盖（多样性增强）
            if getattr(self.config, 'use_constraint_diversity', False) and len(used_constraints) < n_constraints:
                # 优先选择未使用过的约束
                available_constraints = list(range(n_constraints))
                unused_constraints = [c for c in available_constraints if c not in used_constraints]
                
                if unused_constraints:
                    # 从未使用的约束中随机选择
                    constraint_id = np.random.choice(unused_constraints)
                    logger.debug(f"选择未使用约束: {constraint_id}")
                else:
                    # 所有约束都用过了，重新开始
                    constraint_id = self.masker.select_constraint_to_mask(generated_data)
                    used_constraints.clear()  # 重置使用记录
                    logger.debug(f"重置约束选择记录，选择: {constraint_id}")
                
                used_constraints.add(constraint_id)
            else:
                # 标准随机选择
                constraint_id = self.masker.select_constraint_to_mask(generated_data)
            
            # 2. 遮盖约束
            masked_data, mask_info = self.masker.mask_constraint(generated_data, constraint_id)
            
            # 3. 从先验分布采样潜变量（多样性增强）
            if self.config.sample_from_prior:
                # 为每次迭代使用不同的温度，增强多样性
                if getattr(self.config, 'use_dynamic_temperature', False):
                    temp_range = getattr(self.config, 'temperature_range', (0.3, 3.0))
                    iteration_temperature = np.random.uniform(temp_range[0], temp_range[1])
                    logger.debug(f"迭代 {iteration+1} 使用温度: {iteration_temperature:.2f}")
                    z_dict = self._sample_from_prior(generated_data, dynamic_temperature=iteration_temperature)
                else:
                    z_dict = self._sample_from_prior(generated_data)
            else:
                # 使用编码器编码（也可以添加温度调节）
                z_dict, _, _ = self.encoder.encode(generated_data, sample=True)
            
            # 4. 解码生成新约束
            predicted_results = self.decoder(masked_data, z_dict, constraint_id)
            
            # 5. 用生成的约束替换原约束
            generated_data = self.masker.unmask_constraint(
                masked_data, mask_info, predicted_results
            )
            
            # 记录生成历史（增强版，包含多样性信息）
            iteration_info = {
                'iteration': iteration,
                'masked_constraint_id': constraint_id,
                'predicted_bias': predicted_results['predicted_bias'].item(),
                'predicted_degree': predicted_results['predicted_degree'].item(),
                'n_connections': predicted_results['connection_mask'].sum().item()
            }
            
            # 添加多样性相关信息
            if getattr(self.config, 'use_dynamic_temperature', False):
                iteration_info['temperature'] = iteration_temperature
            if getattr(self.config, 'use_constraint_diversity', False):
                iteration_info['constraint_reused'] = constraint_id in [h['masked_constraint_id'] for h in generation_history]
            
            generation_history.append(iteration_info)
        
        logger.info(f"推理生成完成")
        
        return {
            'generated_data': generated_data,
            'generation_history': generation_history,
            'num_iterations': num_iterations,
            'eta': dynamic_eta,  # 使用实际的动态η值
            'diversity_stats': self._compute_diversity_stats(generation_history)
        }
    
    def _sample_from_prior(self, data: HeteroData, dynamic_temperature: float = None) -> Dict[str, torch.Tensor]:
        """
        从先验分布采样潜变量（增强版）
        
        Args:
            data: 图数据
            dynamic_temperature: 动态温度参数，如果为None则使用配置中的温度
            
        Returns:
            采样的潜变量字典
        """
        latent_dim = self.config.encoder_config.latent_dim
        device = self.config.device
        
        # 动态温度调整
        if dynamic_temperature is not None:
            temperature = dynamic_temperature
        else:
            # 从配置中获取温度，支持动态范围
            base_temp = getattr(self.config, 'temperature', 1.0)
            temp_range = getattr(self.config, 'temperature_range', (0.5, 2.0))
            if hasattr(self.config, 'use_dynamic_temperature') and self.config.use_dynamic_temperature:
                # 在训练时，温度在范围内随机采样
                temperature = np.random.uniform(temp_range[0], temp_range[1])
            else:
                temperature = base_temp
        
        # 球面采样选项（更均匀的分布）
        use_spherical = getattr(self.config, 'use_spherical_sampling', False)
        
        if use_spherical:
            # 球面采样：先采样方向，再采样半径
            z_dict = {}
            for node_type in ['constraint', 'variable']:
                n_nodes = data[node_type].x.size(0)
                # 采样标准高斯，然后归一化到单位球面
                raw_samples = torch.randn(n_nodes, latent_dim, device=device)
                normalized = F.normalize(raw_samples, p=2, dim=-1)
                
                # 采样半径（使用Chi分布近似高维球面）
                radii = torch.sqrt(torch.tensor(latent_dim, device=device, dtype=torch.float)) * temperature
                radii = radii * torch.pow(torch.rand(n_nodes, 1, device=device), 1.0 / latent_dim)
                
                z_dict[node_type] = normalized * radii
        else:
            # 标准高斯采样（增强版）
            z_dict = {
                'constraint': torch.randn(
                    data['constraint'].x.size(0), latent_dim,
                    device=device
                ) * temperature,
                'variable': torch.randn(
                    data['variable'].x.size(0), latent_dim,
                    device=device
                ) * temperature
            }
        
        # 噪声注入选项
        noise_injection = getattr(self.config, 'noise_injection_strength', 0.0)
        if noise_injection > 0.0:
            for node_type in z_dict:
                noise = torch.randn_like(z_dict[node_type]) * noise_injection
                z_dict[node_type] = z_dict[node_type] + noise
        
        return z_dict
    
    def _compute_diversity_stats(self, generation_history: List[Dict]) -> Dict[str, float]:
        """
        计算生成过程的多样性统计信息
        
        Args:
            generation_history: 生成历史记录
            
        Returns:
            多样性统计字典
        """
        if not generation_history:
            return {}
        
        try:
            # 提取关键指标
            biases = [h['predicted_bias'] for h in generation_history]
            degrees = [h['predicted_degree'] for h in generation_history]
            connections = [h['n_connections'] for h in generation_history]
            constraint_ids = [h['masked_constraint_id'] for h in generation_history]
            
            # 计算多样性指标
            stats = {
                'bias_std': float(np.std(biases)),
                'bias_range': float(np.max(biases) - np.min(biases)),
                'degree_std': float(np.std(degrees)),
                'degree_range': float(np.max(degrees) - np.min(degrees)),
                'connection_std': float(np.std(connections)),
                'connection_range': float(np.max(connections) - np.min(connections)),
                'unique_constraints_ratio': len(set(constraint_ids)) / len(constraint_ids),
                'avg_bias': float(np.mean(biases)),
                'avg_degree': float(np.mean(degrees)),
                'avg_connections': float(np.mean(connections))
            }
            
            # 如果有温度信息，添加温度多样性
            if 'temperature' in generation_history[0]:
                temperatures = [h['temperature'] for h in generation_history if 'temperature' in h]
                if temperatures:
                    stats['temperature_std'] = float(np.std(temperatures))
                    stats['temperature_range'] = float(np.max(temperatures) - np.min(temperatures))
            
            return stats
            
        except Exception as e:
            logger.warning(f"计算多样性统计失败: {e}")
            return {}
    
    def _compute_dynamic_kl_weight(self) -> float:
        """
        计算动态KL散度权重（课程学习策略）
        
        策略：
        1. 预热期（0-200 epochs）：权重为0，专注重构任务
        2. 退火期（200-800 epochs）：权重从0逐渐增长到目标值
        3. 稳定期（800+ epochs）：权重保持目标值
        
        Returns:
            动态计算的KL权重
        """
        if not hasattr(self.config, 'use_curriculum_learning') or not self.config.use_curriculum_learning:
            return self.config.beta_kl
        
        warmup_epochs = getattr(self.config, 'curriculum_kl_warmup_epochs', 200)
        annealing_epochs = getattr(self.config, 'curriculum_kl_annealing_epochs', 600)
        total_curriculum_epochs = warmup_epochs + annealing_epochs
        
        if self.current_epoch < warmup_epochs:
            # 预热期：KL权重为0
            return 0.0
        elif self.current_epoch < total_curriculum_epochs:
            # 退火期：使用余弦退火策略
            progress = (self.current_epoch - warmup_epochs) / annealing_epochs
            # 余弦退火：从0平滑增长到beta_kl
            import math
            cosine_factor = 0.5 * (1 - math.cos(math.pi * progress))
            dynamic_weight = self.config.beta_kl * cosine_factor
            return dynamic_weight
        else:
            # 稳定期：使用目标权重
            return self.config.beta_kl
    
    def update_training_state(self, epoch: int, step: int = None):
        """
        更新训练状态（用于课程学习）
        
        Args:
            epoch: 当前epoch
            step: 当前训练步数（可选）
        """
        self.current_epoch = epoch
        if step is not None:
            self.training_step = step
    
    def _compute_training_losses(self, 
                                original_data: HeteroData,
                                masked_data: HeteroData,
                                mask_info: Dict,
                                predicted_results: Dict,
                                mu_dict: Dict[str, torch.Tensor],
                                logvar_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        计算训练损失（简化稳定版本）
        
        重新设计损失函数以解决梯度爆炸和训练不稳定问题
        """
        losses = {}
        device = self.config.device
        
        constraint_id = mask_info['masked_constraint_id']
        original_edges = mask_info['original_edges']
        
        # 1. 偏置预测损失（简化稳定版）
        if 'predicted_bias' in predicted_results:
            try:
                target_bias = self._extract_target_bias(original_data, constraint_id)
                predicted_bias = predicted_results['predicted_bias']
                
                # 标准化处理，防止数值过大
                target_bias = torch.clamp(target_bias, min=-10.0, max=10.0)
                predicted_bias = torch.clamp(predicted_bias, min=-10.0, max=10.0)
                
                # 简单的MSE损失，但加权较小
                bias_loss = F.mse_loss(predicted_bias, target_bias)
                bias_loss = torch.clamp(bias_loss, max=1.0)  # 强制裁剪
                losses['bias'] = 0.1 * bias_loss  # 降低权重
                
            except Exception as e:
                logger.debug(f"偏置损失计算失败: {e}")
                losses['bias'] = torch.tensor(0.05, device=device)
        
        # 2. 度数预测损失（简化稳定版）
        if 'predicted_degree' in predicted_results:
            try:
                target_degree = len(original_edges)
                predicted_degree = predicted_results['predicted_degree'].float()
                
                # 归一化到合理范围
                target_degree_norm = target_degree / 50.0  # 归一化
                predicted_degree_norm = predicted_degree / 50.0
                
                # 确保维度匹配：target需要与predicted同维度
                target_tensor = torch.tensor(target_degree_norm, device=device).view_as(predicted_degree_norm)
                
                # 简单MSE，强制裁剪
                degree_loss = F.mse_loss(predicted_degree_norm, target_tensor)
                degree_loss = torch.clamp(degree_loss, max=0.5)  # 严格裁剪
                losses['degree'] = 0.2 * degree_loss  # 大幅降低权重
                
            except Exception as e:
                logger.debug(f"度数损失计算失败: {e}")
                losses['degree'] = torch.tensor(0.1, device=device)
        
        # 3. 连接预测损失（极简稳定版）
        if 'connection_logits' in predicted_results:
            try:
                target_connections = self._create_target_connections(
                    original_data, original_edges, constraint_id
                )
                connection_logits = predicted_results['connection_logits']
                
                # 强制裁剪logits到安全范围
                connection_logits = torch.clamp(connection_logits, min=-3.0, max=3.0)
                
                # 简单密度损失（最稳定的方法）
                pred_density = torch.sigmoid(connection_logits).mean()
                target_density = target_connections.float().mean()
                
                # MSE密度损失，强制裁剪
                logits_loss = F.mse_loss(pred_density, target_density)
                logits_loss = torch.clamp(logits_loss, max=0.2)  # 极严格裁剪
                losses['logits'] = 0.5 * logits_loss  # 降低权重
                
            except Exception as e:
                logger.debug(f"连接损失计算失败: {e}")
                losses['logits'] = torch.tensor(0.1, device=device)
        
        # 4. 权重预测损失（超简化版）
        if 'predicted_weights' in predicted_results and len(original_edges) > 0:
            try:
                predicted_weights = predicted_results['predicted_weights']
                # 简单的L2正则化，鼓励小权重
                weights_loss = (predicted_weights ** 2).mean()
                weights_loss = torch.clamp(weights_loss, max=0.1)
                losses['weights'] = 0.05 * weights_loss  # 极小权重
                
            except Exception as e:
                logger.debug(f"权重损失计算失败: {e}")
                losses['weights'] = torch.tensor(0.02, device=device)
        
        # 5. KL散度损失（超简化版）
        try:
            kl_loss = self.encoder.compute_kl_divergence(mu_dict, logvar_dict)
            kl_loss = torch.clamp(kl_loss, max=1.0)  # 强制裁剪
            
            # 使用固定的极小权重，避免KL坍塌
            losses['kl'] = 0.001 * kl_loss  # 极小权重，避免主导训练
            
        except Exception as e:
            logger.debug(f"KL散度计算失败: {e}")
            losses['kl'] = torch.tensor(0.001, device=device)
        
        # 总损失计算（极简版）
        total_loss = torch.tensor(0.0, device=device)
        
        for loss_name, loss_value in losses.items():
            if torch.isfinite(loss_value):
                total_loss = total_loss + loss_value
        
        # 强制总损失裁剪，防止梯度爆炸
        total_loss = torch.clamp(total_loss, max=2.0)  # 严格控制总损失
        losses['total'] = total_loss
        
        return losses
    
    def _create_target_connections(self, 
                                  original_data: HeteroData,
                                  original_edges: List[Tuple],
                                  constraint_id: int) -> torch.Tensor:
        """创建目标连接向量"""
        n_variables = original_data['variable'].x.size(0)
        target_connections = torch.zeros(n_variables, 1, device=self.config.device)
        
        # 标记原始连接的变量
        for edge in original_edges:
            if len(edge) >= 2:
                var_id = edge[1]  # (constraint_id, variable_id, ...)
                if var_id < n_variables:
                    target_connections[var_id, 0] = 1.0
        
        return target_connections
    
    def _extract_target_bias(self, original_data: HeteroData, constraint_id: int) -> torch.Tensor:
        """
        从原始数据中提取目标约束的偏置值
        
        Args:
            original_data: 原始图数据
            constraint_id: 约束节点ID
            
        Returns:
            目标偏置值的张量
        """
        try:
            # 方法1: 尝试从约束节点特征中提取偏置（第4个特征通常是RHS值）
            if constraint_id < original_data['constraint'].x.size(0):
                constraint_features = original_data['constraint'].x[constraint_id]
                # 假设第4个特征是RHS偏置值（根据data_structures.py中的特征向量定义）
                if constraint_features.size(0) >= 5:  # 确保有足够的特征
                    rhs_value = constraint_features[4].float()  # 第5个特征位置（索引4）
                else:
                    # 备用方案：使用约束特征的绝对值均值作为偏置
                    rhs_value = torch.abs(constraint_features).mean()
                    
                # 确保返回正确的形状和设备
                target_bias = rhs_value.unsqueeze(0).unsqueeze(0).to(self.config.device)
                
                # 数值稳定性检查
                if not torch.isfinite(target_bias):
                    logger.warning(f"提取的偏置值非有限: {target_bias}, 使用默认值")
                    target_bias = torch.tensor([[1.0]], device=self.config.device, dtype=torch.float)
                    
                return target_bias
            else:
                logger.warning(f"约束ID {constraint_id} 超出范围，使用默认偏置")
                return torch.tensor([[1.0]], device=self.config.device, dtype=torch.float)
                
        except Exception as e:
            logger.warning(f"提取目标偏置失败: {e}，使用默认值")
            return torch.tensor([[1.0]], device=self.config.device, dtype=torch.float)
    
    def _extract_target_weights(self, original_data: HeteroData, original_edges: List[Tuple], 
                               target_length: int = None) -> torch.Tensor:
        """
        从原始数据中提取目标约束的权重系数（改进版：交集损失+伪正例惩罚）
        
        新策略：
        1. 只对预测为真且真实也为真的连接（交集）计算损失
        2. 对多余的预测连接（伪正例）施加惩罚项
        3. 防止维度对齐时的噪声引入
        
        Args:
            original_data: 原始图数据
            original_edges: 原始边列表，格式为 [(constraint_id, variable_id, weight), ...]
            target_length: 目标权重张量的长度（用于维度对齐）
            
        Returns:
            目标权重的张量，维度与predicted_weights匹配
        """
        try:
            if not original_edges:
                # 如果没有原始边，根据target_length生成默认权重
                length = target_length if target_length is not None else 1
                return torch.ones(length, 1, device=self.config.device, dtype=torch.float)
            
            # 改进策略：根据预测连接数量确定目标长度
            # 这样可以避免强制维度对齐带来的噪声
            if target_length is None:
                target_length = len(original_edges)
            
            # 从边信息中提取权重（仅从有效边中提取）
            edge_weights_dict = {}  # {variable_id: weight}
            for edge in original_edges:
                if len(edge) >= 3:
                    try:
                        var_id = edge[1]
                        weight = float(edge[2])
                        edge_weights_dict[var_id] = weight
                    except (ValueError, TypeError, IndexError):
                        if len(edge) >= 2:
                            var_id = edge[1]
                            edge_weights_dict[var_id] = 1.0
            
            if not edge_weights_dict:
                # 由于无法提取有效权重，返回默认值
                return torch.ones(target_length, 1, device=self.config.device, dtype=torch.float)
            
            # 根据目标长度构建权重张量
            if target_length <= len(edge_weights_dict):
                # 预测连接数量不超过原始连接数：使用原始权重
                weights = list(edge_weights_dict.values())[:target_length]
            else:
                # 预测连接数量超过原始连接数：使用策略扩展
                original_weights = list(edge_weights_dict.values())
                
                if len(original_weights) == 1:
                    # 单边情况：使用相同权重 + 小幅随机扰动
                    base_weight = original_weights[0]
                    weights = []
                    for i in range(target_length):
                        # 添加±5%的随机扰动，避免完全相同
                        noise = np.random.uniform(-0.05, 0.05)
                        perturbed_weight = base_weight * (1.0 + noise)
                        weights.append(perturbed_weight)
                else:
                    # 多边情况：使用高斯混合模型生成新权重
                    weights = []
                    mean_weight = np.mean(original_weights)
                    std_weight = max(np.std(original_weights), 0.1 * abs(mean_weight))  # 防止标准差过小
                    
                    # 先使用原始权重
                    weights.extend(original_weights)
                    
                    # 然后生成额外的权重
                    for i in range(target_length - len(original_weights)):
                        new_weight = np.random.normal(mean_weight, std_weight)
                        weights.append(new_weight)
            
            # 转换为张量
            weights_tensor = torch.tensor(weights, device=self.config.device, dtype=torch.float)
            
            # 确保形状为 [length, 1]
            if weights_tensor.dim() == 1:
                weights_tensor = weights_tensor.unsqueeze(-1)
            
            # 数值稳定性检查和清理
            if not torch.isfinite(weights_tensor).all():
                logger.warning("权重包含非有限值，进行清理")
                finite_mask = torch.isfinite(weights_tensor)
                if finite_mask.any():
                    # 使用有限权重的平均值替换非有限值
                    valid_weights = weights_tensor[finite_mask]
                    replacement_value = valid_weights.mean()
                    weights_tensor = torch.where(finite_mask, weights_tensor, replacement_value)
                else:
                    # 所有权重都无效，使用默认值
                    weights_tensor = torch.ones_like(weights_tensor)
            
            return weights_tensor
            
        except Exception as e:
            logger.warning(f"提取目标权重失败: {e}，使用默认权重")
            length = target_length if target_length is not None else 1
            return torch.ones(length, 1, device=self.config.device, dtype=torch.float)
    
    def compute_total_loss(self, losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        """计算总损失"""
        return losses['total']
    
    def generate_single_iteration(self, data: HeteroData) -> Tuple[HeteroData, Dict]:
        """
        单次生成迭代
        
        Args:
            data: 输入图数据
            
        Returns:
            (generated_data, iteration_info)
        """
        with torch.no_grad():
            # 切换到推理模式
            self.eval()
            
            # 确保输入数据在正确设备上
            target_device = self.config.device
            data = data.to(target_device)
            
            # 选择约束进行遮盖
            constraint_id = self.masker.select_constraint_to_mask(data)
            
            # 遮盖
            masked_data, mask_info = self.masker.mask_constraint(data, constraint_id)
            masked_data = masked_data.to(target_device)
            
            # 从先验分布采样
            z_dict = self._sample_from_prior(data)
            
            # 解码
            predicted_results = self.decoder(masked_data, z_dict, constraint_id)
            
            # 重建
            generated_data = self.masker.unmask_constraint(
                masked_data, mask_info, predicted_results
            )
            
            # 确保生成数据在正确设备上
            generated_data = generated_data.to(target_device)
            
            iteration_info = {
                'masked_constraint_id': constraint_id,
                'predicted_bias': predicted_results['predicted_bias'].cpu().item(),
                'predicted_degree': predicted_results['predicted_degree'].cpu().item(),
                'n_connections': predicted_results['connection_mask'].sum().cpu().item()
            }
            
            # 保持数据在原设备上，不强制转CPU
            return generated_data, iteration_info
    
    def save_model(self, filepath: str) -> bool:
        """保存模型"""
        try:
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            # 保存模型状态
            torch.save({
                'model_state_dict': self.state_dict(),
                'config': self.config,
                'constraint_feature_dim': self.constraint_feature_dim,
                'variable_feature_dim': self.variable_feature_dim,
                'edge_feature_dim': self.edge_feature_dim
            }, filepath)
            
            logger.info(f"模型已保存: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"模型保存失败: {e}")
            return False
    
    @classmethod
    def load_model(cls, filepath: str) -> 'G2MILPGenerator':
        """加载模型"""
        try:
            checkpoint = torch.load(filepath, map_location='cpu')
            
            # 创建模型实例
            model = cls(
                constraint_feature_dim=checkpoint['constraint_feature_dim'],
                variable_feature_dim=checkpoint['variable_feature_dim'],
                edge_feature_dim=checkpoint['edge_feature_dim'],
                config=checkpoint['config']
            )
            
            # 加载状态
            model.load_state_dict(checkpoint['model_state_dict'])
            
            logger.info(f"模型已加载: {filepath}")
            return model
            
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise
    
    def _compute_sparsity_loss(self, original_data: HeteroData, predicted_results: Dict) -> torch.Tensor:
        """
        计算稀疏性正则化损失
        
        目标：防止生成的图过于密集，保持MILP问题的稀疏性特征
        
        Args:
            original_data: 原始图数据
            predicted_results: 预测结果
            
        Returns:
            稀疏性损失值
        """
        try:
            # 获取图的基本信息
            n_constraints = original_data['constraint'].x.size(0)
            n_variables = original_data['variable'].x.size(0)
            max_possible_edges = n_constraints * n_variables
            
            # 计算原始图的稀疏度作为目标
            if ('constraint', 'connects', 'variable') in original_data.edge_index_dict:
                original_edges = original_data[('constraint', 'connects', 'variable')].edge_index
                original_edge_count = original_edges.size(1)
                target_sparsity = original_edge_count / max_possible_edges
            else:
                target_sparsity = getattr(self.config, 'target_sparsity', 0.1)
            
            # 计算预测的边密度
            if 'connection_mask' in predicted_results:
                predicted_mask = predicted_results['connection_mask']
                predicted_edge_density = predicted_mask.float().mean()
            elif 'connection_logits' in predicted_results:
                # 使用sigmoid将logits转换为概率，然后计算期望边数
                connection_probs = torch.sigmoid(predicted_results['connection_logits'])
                predicted_edge_density = connection_probs.mean()
            else:
                # 备用方案：使用predicted_degree
                if 'predicted_degree' in predicted_results:
                    predicted_degree = predicted_results['predicted_degree'].float()
                    # 假设度数均匀分布到所有约束
                    avg_degree_per_constraint = predicted_degree / n_constraints
                    predicted_edge_density = avg_degree_per_constraint / n_variables
                else:
                    return torch.tensor(0.0, device=self.config.device)
            
            # 计算稀疏性损失 - 使用MSE或L1
            target_density = torch.tensor(target_sparsity, device=self.config.device, dtype=torch.float)
            
            # 使用平滑L1损失（Huber Loss），对异常值不敏感
            sparsity_loss = F.smooth_l1_loss(predicted_edge_density, target_density)
            
            # 添加额外的惩罚项：如果预测密度过高，增加额外惩罚
            if predicted_edge_density > target_density * 2.0:
                excess_penalty = (predicted_edge_density - target_density * 2.0) ** 2
                sparsity_loss = sparsity_loss + 0.5 * excess_penalty
            
            return sparsity_loss
            
        except Exception as e:
            logger.warning(f"稀疏性损失计算异常: {e}")
            return torch.tensor(0.0, device=self.config.device)


def create_generator(constraint_feature_dim: int = 16,
                    variable_feature_dim: int = 9,
                    edge_feature_dim: int = 8,
                    config: GeneratorConfig = None) -> G2MILPGenerator:
    """
    创建G2MILP生成器的工厂函数
    
    Args:
        constraint_feature_dim: 约束节点特征维度
        variable_feature_dim: 变量节点特征维度
        edge_feature_dim: 边特征维度
        config: 生成器配置
        
    Returns:
        G2MILP生成器实例
    """
    if config is None:
        config = GeneratorConfig()
    
    return G2MILPGenerator(
        constraint_feature_dim=constraint_feature_dim,
        variable_feature_dim=variable_feature_dim,
        edge_feature_dim=edge_feature_dim,
        config=config
    )


if __name__ == "__main__":
    # 测试代码
    print("G2MILP生成器模块测试")
    print("=" * 40)
    
    # 创建测试配置
    config = GeneratorConfig(
        eta=0.1,
        alpha_bias=1.0,
        alpha_degree=1.0,
        alpha_logits=1.0,
        alpha_weights=1.0,
        beta_kl=0.1
    )
    
    # 创建生成器
    generator = create_generator(config=config)
    
    # 打印模型信息
    total_params = sum(p.numel() for p in generator.parameters())
    trainable_params = sum(p.numel() for p in generator.parameters() if p.requires_grad)
    
    print(f"生成器总参数数量: {total_params:,}")
    print(f"可训练参数数量: {trainable_params:,}")
    print(f"设备: {config.device}")
    print(f"η参数: {config.eta}")
    print("生成器创建成功!")
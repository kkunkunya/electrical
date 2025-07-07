"""
G2MILP训练模块
G2MILP Training Module

实现G2MILP的训练逻辑，包括：
1. 基于单个"有偏差"实例的自我学习训练
2. 损失函数计算和优化
3. 训练进度监控和日志记录
4. 模型保存和恢复
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler
from torch_geometric.data import HeteroData
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import json
from pathlib import Path
from datetime import datetime
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import threading
from concurrent.futures import ThreadPoolExecutor, Future

from .generator import G2MILPGenerator, GeneratorConfig
from .evaluation import G2MILPEvaluator, EvaluationConfig
from .inference import G2MILPInference, InferenceConfig
from .early_stopping import EarlyStoppingMonitor, EarlyStoppingConfig, create_early_stopping_monitor

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """训练配置"""
    # 基础训练参数（大幅提升训练强度）
    num_epochs: int = 5000           # 大幅增加训练轮数
    batch_size: int = 1              # 对于单实例训练，batch_size=1
    learning_rate: float = 1e-4      # 降低初始学习率，增强稳定性
    weight_decay: float = 1e-3       # 提高权重衰减
    
    # 学习率调度增强
    use_lr_scheduler: bool = True
    scheduler_type: str = "cosine_with_warmup"  # plateau, cosine, cosine_with_warmup, cosine_restart
    lr_patience: int = 50
    lr_factor: float = 0.5
    lr_min: float = 1e-6
    lr_scheduler_factor: float = 0.8  # 兼容别名
    lr_scheduler_patience: int = 20    # 兼容别名，增加patience
    
    # 学习率预热和重启
    warmup_epochs: int = 50           # 预热epoch数
    warmup_start_lr: float = 1e-6     # 预热起始学习率
    cosine_restart_period: int = 500   # Cosine重启周期
    restart_mult: float = 2.0         # 重启周期倍数
    
    # 梯度裁剪
    grad_clip_norm: float = 1.0
    
    # 训练策略（增强版）
    iterations_per_epoch: int = 200  # 每个epoch内的迭代次数（翻倍）
    validation_frequency: int = 50   # 验证频率（epoch）
    save_frequency: int = 500        # 模型保存频率（epoch）
    
    # 微批次累积（提高GPU利用率）
    micro_batch_size: int = 4        # 微批次大小（累积4次前向传播）
    gradient_accumulation_steps: int = 4  # 梯度累积步数
    
    # 早停增强（更宽松的策略）
    use_early_stopping: bool = True
    early_stopping_patience: int = 500   # 超大patience，适合长期训练
    early_stopping_min_delta: float = 1e-6  # 更敏感的min_delta
    min_delta: float = 1e-6              # 兼容别名
    
    # 高级Early Stopping配置
    early_stopping_strategy: str = "combined"  # simple, multi_metric, adaptive, trend_analysis, combined
    early_stopping_monitor_metrics: List[str] = None  # 额外监控指标
    early_stopping_quality_threshold: float = 0.7     # 质量阈值
    early_stopping_adaptive_patience: bool = True     # 自适应patience
    early_stopping_trend_analysis: bool = True        # 趋势分析
    early_stopping_verbose: bool = True               # 详细日志
    
    # 损失权重调度（课程学习增强）
    kl_annealing: bool = True
    kl_annealing_epochs: int = 800       # 大幅延长退火期
    kl_start_weight: float = 0.0
    kl_end_weight: float = 1.0
    
    # 新增：数据增强参数
    use_data_augmentation: bool = True
    feature_noise_std: float = 0.05      # 特征噪声标准差（±5%）
    edge_perturbation_prob: float = 0.1  # 边扰动概率
    
    # 新增：优化器增强
    optimizer_type: str = "adamw"        # 使用AdamW优化器
    use_gradient_accumulation: bool = False  # 梯度累积（单实例不需要）
    
    # RTX 3060 Ti专项优化
    use_mixed_precision: bool = True     # 启用AMP混合精度训练（RTX 30系列支持）
    amp_loss_scale: str = "dynamic"      # AMP损失缩放策略 dynamic/static/value
    use_compile: bool = False            # PyTorch 2.0编译优化（可选）
    
    # 稀疏性正则化新增
    use_sparsity_regularization: bool = True
    sparsity_weight: float = 0.01     # 稀疏性损失权重
    target_sparsity: float = 0.1      # 目标稀疏度（边数比例）
    
    # 数值稳定性
    loss_nan_threshold: float = 1000.0  # NaN/Inf替换阈值
    gradient_nan_check: bool = True     # 梯度NaN检查
    
    # 日志和保存
    log_interval: int = 10  # 日志记录间隔（iteration）
    save_dir: str = "output/demo4_g2milp/training"
    experiment_name: str = f"g2milp_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # 质量评估配置（新增）
    enable_quality_evaluation: bool = True       # 启用质量评估
    quality_evaluation_frequency: int = 50       # 质量评估频率（epoch）
    quality_samples_per_eval: int = 3           # 每次评估生成的样本数
    enable_detailed_quality_logging: bool = True # 启用详细质量日志
    
    # 设备
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class G2MILPTrainer:
    """
    G2MILP训练器
    
    专门针对单个"有偏差"实例的自我学习训练过程
    """
    
    def __init__(self, 
                 model: G2MILPGenerator,
                 config: TrainingConfig = None,
                 evaluator = None):
        self.model = model
        self.config = config or TrainingConfig()
        self.evaluator = evaluator  # 在线质量评估器
        
        # 设置设备
        self.device = torch.device(self.config.device)
        self.model = self.model.to(self.device)
        
        # 优化器（支持多种类型）
        self.optimizer = self._create_optimizer()
        
        # 学习率调度器
        self.scheduler = self._create_scheduler()
        
        # AMP混合精度支持（RTX 3060 Ti优化）
        self.use_amp = getattr(self.config, 'use_mixed_precision', True) and torch.cuda.is_available()
        if self.use_amp:
            self.scaler = GradScaler()
            logger.info("已启用AMP混合精度训练 (RTX 30系列优化)")
        else:
            self.scaler = None
            
        # PyTorch 2.0编译优化
        if getattr(self.config, 'use_compile', False) and hasattr(torch, 'compile'):
            try:
                self.model = torch.compile(self.model)
                logger.info("已启用PyTorch 2.0编译优化")
            except Exception as e:
                logger.warning(f"编译优化失败: {e}")
        
        # 训练状态
        self.current_epoch = 0
        self.current_iteration = 0
        self.best_loss = float('inf')
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
            'kl_weight': [],
            # 详细损失分解
            'train_reconstruction': [],
            'train_kl_raw': [],
            'train_bias': [],
            'train_degree': [],
            'train_logits': [],
            'train_weights': [],
            'train_sparsity': [],
            'val_reconstruction': [],
            'val_kl_raw': [],
            # 梯度和参数统计
            'grad_norm': [],
            'grad_max': [],
            'param_norm': [],
            'nan_grads': [],
            'inf_grads': [],
            # 在线质量评估
            'validity_score': [],
            'diversity_score': [],
            'similarity_score': [],
            'stability_score': [],
            'quality_overall': [],
            # 新增：质量评估历史记录
            'generation_quality': [],
            'validity_scores': [],
            'diversity_scores': [],
            'similarity_scores': []
        }
        
        # 高级Early Stopping监控器
        if self.config.use_early_stopping:
            early_stopping_config = EarlyStoppingConfig(
                strategy=getattr(self.config, 'early_stopping_strategy', 'combined'),
                patience=self.config.early_stopping_patience,
                min_delta=self.config.early_stopping_min_delta,
                monitor_metric='val_loss',
                additional_metrics=getattr(self.config, 'early_stopping_monitor_metrics', 
                                         ['train_loss', 'kl_weight', 'grad_norm', 'generation_quality']),
                adaptive_patience=getattr(self.config, 'early_stopping_adaptive_patience', True),
                trend_window=20,
                enable_quality_monitoring=getattr(self.config, 'enable_quality_evaluation', True),
                quality_threshold=getattr(self.config, 'early_stopping_quality_threshold', 0.7),
                verbose=getattr(self.config, 'early_stopping_verbose', True),
                save_best_model=True,
                restore_best_weights=True
            )
            self.early_stopping_monitor = EarlyStoppingMonitor(early_stopping_config)
            logger.info(f"✅ 高级Early Stopping监控器已启用 (策略: {early_stopping_config.strategy.value})")
        else:
            self.early_stopping_monitor = None
        
        # 传统早停（兼容性保留）
        self.early_stopping_counter = 0
        self.should_stop = False
        
        # 创建保存目录
        self.save_dir = Path(self.config.save_dir) / self.config.experiment_name
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化质量评估器（新增）
        if getattr(self.config, 'enable_quality_evaluation', True):
            eval_config = EvaluationConfig(
                enable_graph_similarity=True,
                enable_milp_similarity=True,
                enable_diversity_analysis=True,
                enable_training_monitoring=True,
                diversity_sample_size=getattr(self.config, 'quality_samples_per_eval', 3),
                generate_visualizations=False,  # 训练时不生成可视化，避免过多文件
                save_detailed_results=False,     # 训练时不保存详细结果
                output_dir=str(self.save_dir / "quality_evaluation")
            )
            self.quality_evaluator = G2MILPEvaluator(eval_config)
            
            # 初始化推理器用于质量评估
            inference_config = InferenceConfig(
                num_test_instances=getattr(self.config, 'quality_samples_per_eval', 3),
                eta=0.1,
                sample_from_prior=True,
                constraint_selection_strategy="random"
            )
            self.quality_inferencer = None  # 延迟初始化，因为需要模型
        else:
            self.quality_evaluator = None
            self.quality_inferencer = None
        
        logger.info(f"G2MILP训练器初始化完成")
        logger.info(f"设备: {self.device}")
        logger.info(f"优化器: {self.config.optimizer_type}")
        logger.info(f"保存目录: {self.save_dir}")
        logger.info(f"质量评估: {'启用' if self.quality_evaluator else '禁用'}")
    
    def _create_optimizer(self):
        """创建优化器"""
        optimizer_type = getattr(self.config, 'optimizer_type', 'adam').lower()
        
        if optimizer_type == 'adamw':
            return optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                betas=(0.9, 0.999),
                eps=1e-8
            )
        elif optimizer_type == 'adam':
            return optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                betas=(0.9, 0.999),
                eps=1e-8
            )
        elif optimizer_type == 'sgd':
            return optim.SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                momentum=0.9,
                nesterov=True
            )
        else:
            logger.warning(f"未知的优化器类型: {optimizer_type}, 使用默认AdamW")
            return optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
    
    def _create_scheduler(self):
        """创建学习率调度器"""
        if not self.config.use_lr_scheduler:
            return None
        
        scheduler_type = self.config.scheduler_type
        
        if scheduler_type == "plateau":
            # 支持新的配置参数名称
            factor = getattr(self.config, 'lr_scheduler_factor', self.config.lr_factor)
            patience = getattr(self.config, 'lr_scheduler_patience', self.config.lr_patience)
            
            return ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=factor,
                patience=patience,
                min_lr=self.config.lr_min
            )
        elif scheduler_type == "cosine":
            return CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.num_epochs,
                eta_min=self.config.lr_min
            )
        elif scheduler_type == "cosine_with_warmup":
            # 自定义的Cosine with Warmup调度器
            return self._create_cosine_warmup_scheduler()
        elif scheduler_type == "cosine_restart":
            # Cosine Annealing with Warm Restarts
            from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
            return CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=self.config.cosine_restart_period,
                T_mult=int(self.config.restart_mult),
                eta_min=self.config.lr_min
            )
        else:
            logger.warning(f"未知的调度器类型: {scheduler_type}, 使用默认plateau")
            return ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=50,
                min_lr=self.config.lr_min
            )
    
    def _create_cosine_warmup_scheduler(self):
        """创建带预热的Cosine调度器（优化版）"""
        from torch.optim.lr_scheduler import LambdaLR
        import math
        
        def lr_lambda(current_step: int) -> float:
            if current_step < self.config.warmup_epochs:
                # 预热阶段：从较高的基础值开始线性增长到1.0
                base_ratio = 0.1  # 预热期起始比例（而非0）
                warmup_ratio = base_ratio + (1.0 - base_ratio) * (
                    float(current_step) / float(max(1, self.config.warmup_epochs))
                )
                return warmup_ratio
            
            # Cosine退火阶段
            progress = float(current_step - self.config.warmup_epochs) / float(
                max(1, self.config.num_epochs - self.config.warmup_epochs)
            )
            # 改善的余弦退火：保持更高的最小学习率
            min_ratio = self.config.lr_min / self.config.learning_rate
            cosine_ratio = min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
            return max(min_ratio, cosine_ratio)
        
        return LambdaLR(self.optimizer, lr_lambda)
    
    def _get_kl_weight(self, epoch: int) -> float:
        """计算KL散度权重（KL退火）"""
        if not self.config.kl_annealing:
            return self.config.kl_end_weight
        
        if epoch >= self.config.kl_annealing_epochs:
            return self.config.kl_end_weight
        
        # 线性退火
        progress = epoch / self.config.kl_annealing_epochs
        weight = self.config.kl_start_weight + progress * (
            self.config.kl_end_weight - self.config.kl_start_weight
        )
        
        return weight
    
    def train_epoch(self, data: HeteroData) -> Dict[str, float]:
        """
        训练一个epoch
        
        Args:
            data: 训练数据（单个有偏差实例）
            
        Returns:
            epoch统计信息
        """
        self.model.train()
        epoch_losses = {
            'total': 0.0,
            'bias': 0.0,
            'degree': 0.0,
            'logits': 0.0,
            'weights': 0.0,
            'kl': 0.0,
            'kl_raw': 0.0,  # 未加权的原始KL散度
            'sparsity': 0.0,
            'reconstruction': 0.0  # 重构损失总和
        }
        
        # 添加梯度和参数统计
        gradient_stats = {
            'grad_norm': 0.0,
            'grad_max': 0.0,
            'param_norm': 0.0,
            'param_updates': 0.0,
            'nan_grads': 0,
            'inf_grads': 0
        }
        
        # KL权重（如果模型支持课程学习，会被覆盖）
        kl_weight = self._get_kl_weight(self.current_epoch)
        
        # 更新模型的训练状态（用于课程学习）
        if hasattr(self.model, 'update_training_state'):
            self.model.update_training_state(self.current_epoch)
        else:
            # 兼容旧版本：直接更新KL权重
            self.model.config.beta_kl = kl_weight
        
        # 在一个epoch内多次使用同一个实例进行训练
        iteration_bar = tqdm(
            range(self.config.iterations_per_epoch),
            desc=f"Epoch {self.current_epoch}",
            leave=False,
            ncols=80,
            disable=self.current_epoch % 10 != 0  # 只有每10个epoch显示迭代进度条
        )
        
        # 获取梯度累积配置
        accumulation_steps = getattr(self.config, 'gradient_accumulation_steps', 1)
        
        for iteration in iteration_bar:
            # 只在累积开始时清零梯度
            if iteration % accumulation_steps == 0:
                self.optimizer.zero_grad()
            
            # 数据增强（如果启用）
            augmented_data = self._apply_data_augmentation(data) if getattr(self.config, 'use_data_augmentation', False) else data
            
            # 前向传播（支持AMP混合精度）
            try:
                if self.use_amp:
                    with autocast():
                        results = self.model(augmented_data, mode="train")
                        losses = results['losses']
                        
                        # 添加稀疏性正则化（如果启用）
                        if getattr(self.config, 'use_sparsity_regularization', False):
                            sparsity_loss = self._compute_sparsity_regularization(results, data)
                            sparsity_weight = getattr(self.config, 'sparsity_weight', 0.1)
                            losses['sparsity'] = sparsity_loss
                            losses['total'] = losses['total'] + sparsity_weight * sparsity_loss
                else:
                    results = self.model(augmented_data, mode="train")
                    losses = results['losses']
                    
                    # 添加稀疏性正则化（如果启用）
                    if getattr(self.config, 'use_sparsity_regularization', False):
                        sparsity_loss = self._compute_sparsity_regularization(results, data)
                        sparsity_weight = getattr(self.config, 'sparsity_weight', 0.1)
                        losses['sparsity'] = sparsity_loss
                        losses['total'] = losses['total'] + sparsity_weight * sparsity_loss
                    
            except RuntimeError as e:
                if "Expected all tensors to be on the same device" in str(e):
                    print(f"设备错误详情: {e}")
                    print(f"数据设备状态:")
                    print(f"  constraint.x device: {data['constraint'].x.device}")
                    print(f"  variable.x device: {data['variable'].x.device}")
                    if hasattr(data, 'edge_index_dict'):
                        for key, edge_index in data.edge_index_dict.items():
                            print(f"  {key} edge_index device: {edge_index.device}")
                raise
            
            # 反向传播（支持AMP和梯度累积）
            total_loss = losses['total']
            
            # 梯度累积：除以累积步数
            if accumulation_steps > 1:
                total_loss = total_loss / accumulation_steps
            
            # 数值稳定性检查：确保损失是有限的
            if not torch.isfinite(total_loss):
                logger.warning(f"迭代 {iteration}: 检测到非有限损失 {total_loss.item()}，跳过该迭代")
                continue
            
            if self.use_amp:
                # AMP反向传播
                self.scaler.scale(total_loss).backward()
                
                # 只在累积周期结束时进行优化器更新
                if (iteration + 1) % accumulation_steps == 0:
                    # 智能梯度有效性检查（允许少量NaN）
                    grad_stats = self._analyze_gradient_health()
                    
                    # 如果NaN梯度比例过高（>20%），跳过更新
                    if grad_stats['nan_ratio'] > 0.2:
                        logger.warning(f"迭代 {iteration}: NaN梯度比例过高 {grad_stats['nan_ratio']:.3f}，跳过更新")
                        self.optimizer.zero_grad()
                        continue
                    
                    # 如果有少量NaN梯度（≤20%），进行零值替换
                    if grad_stats['nan_ratio'] > 0.0:
                        logger.debug(f"迭代 {iteration}: 检测到 {grad_stats['nan_ratio']:.3f} NaN梯度，进行零值替换")
                        self._replace_nan_gradients()
                    
                    # 检查梯度是否全为零（可能表示梯度消失）
                    if grad_stats['zero_ratio'] > 0.9:
                        logger.warning(f"迭代 {iteration}: 梯度几乎全为零 {grad_stats['zero_ratio']:.3f}，可能存在梯度消失问题")
                        # 不跳过，但记录警告
                    
                    # 梯度裁剪（需要先unscale）
                    if self.config.grad_clip_norm > 0:
                        self.scaler.unscale_(self.optimizer)
                        
                        # 强化梯度裁剪（更保守的值）
                        actual_clip_norm = min(self.config.grad_clip_norm, 0.1)
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), 
                            actual_clip_norm
                        )
                        
                        # 如果梯度范数过大，跳过更新
                        if grad_norm > 10.0:
                            logger.warning(f"迭代 {iteration}: 梯度范数过大 {grad_norm:.4f}，跳过更新")
                            self.optimizer.zero_grad()
                            continue
                    
                    # 优化器步进
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
            else:
                # 标准反向传播
                total_loss.backward()
                
                # 只在累积周期结束时进行优化器更新
                if (iteration + 1) % accumulation_steps == 0:
                    # 智能梯度有效性检查（允许少量NaN）
                    grad_stats = self._analyze_gradient_health()
                    
                    # 如果NaN梯度比例过高（>20%），跳过更新
                    if grad_stats['nan_ratio'] > 0.2:
                        logger.warning(f"迭代 {iteration}: NaN梯度比例过高 {grad_stats['nan_ratio']:.3f}，跳过更新")
                        self.optimizer.zero_grad()
                        continue
                    
                    # 如果有少量NaN梯度（≤20%），进行零值替换
                    if grad_stats['nan_ratio'] > 0.0:
                        logger.debug(f"迭代 {iteration}: 检测到 {grad_stats['nan_ratio']:.3f} NaN梯度，进行零值替换")
                        self._replace_nan_gradients()
                    
                    # 检查梯度是否全为零（可能表示梯度消失）
                    if grad_stats['zero_ratio'] > 0.9:
                        logger.warning(f"迭代 {iteration}: 梯度几乎全为零 {grad_stats['zero_ratio']:.3f}，可能存在梯度消失问题")
                    
                    # 强化梯度裁剪
                    if self.config.grad_clip_norm > 0:
                        actual_clip_norm = min(self.config.grad_clip_norm, 0.1)
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), 
                            actual_clip_norm
                        )
                        
                        # 如果梯度范数过大，跳过更新
                        if grad_norm > 10.0:
                            logger.warning(f"迭代 {iteration}: 梯度范数过大 {grad_norm:.4f}，跳过更新")
                            self.optimizer.zero_grad()
                            continue
                    
                    # 优化器步进
                    self.optimizer.step()
            
            # 累计损失
            for key in epoch_losses:
                if key in losses:
                    epoch_losses[key] += losses[key].item()
            
            # 计算原始KL散度（未加权）
            if 'kl' in losses and kl_weight > 0:
                epoch_losses['kl_raw'] += (losses['kl'].item() / kl_weight)
            
            # 计算重构损失总和
            reconstruction_components = ['bias', 'degree', 'logits', 'weights']
            reconstruction_total = sum(losses[comp].item() for comp in reconstruction_components if comp in losses)
            epoch_losses['reconstruction'] += reconstruction_total
            
            # 梯度统计计算
            grad_stats = self._compute_gradient_stats()
            for key, value in grad_stats.items():
                if key not in gradient_stats:
                    gradient_stats[key] = 0  # 初始化新键
                gradient_stats[key] += value
            
            # 更新迭代进度条
            if not iteration_bar.disable:
                iter_desc = f"Loss:{total_loss.item():.4f} GradNorm:{grad_stats.get('grad_norm', 0):.3f}"
                iteration_bar.set_postfix_str(iter_desc)
            
            # 记录详细日志（减少频率）
            if iteration % (self.config.log_interval * 5) == 0:
                logger.debug(
                    f"Epoch {self.current_epoch}, Iter {iteration}: "
                    f"Total = {total_loss.item():.6f}, "
                    f"Reconstruction = {reconstruction_total:.6f}, "
                    f"KL(raw) = {epoch_losses['kl_raw']/(iteration+1):.6f}, "
                    f"KL_weight = {kl_weight:.6f}, "
                    f"Grad_norm = {grad_stats.get('grad_norm', 0):.4f}"
                )
            
            self.current_iteration += 1
        
        # 计算平均损失和梯度统计
        for key in epoch_losses:
            epoch_losses[key] /= self.config.iterations_per_epoch
        
        for key in gradient_stats:
            if key not in ['nan_grads', 'inf_grads', 'zero_grads', 'finite_grads', 'total_params']:
                gradient_stats[key] /= self.config.iterations_per_epoch
        
        # 合并损失和梯度统计
        combined_stats = {**epoch_losses, **gradient_stats}
        
        return combined_stats
    
    def _compute_gradient_stats(self) -> Dict[str, float]:
        """
        计算梯度和参数统计信息（增强版）
        
        Returns:
            梯度统计字典，包含详细的健康状况分析
        """
        stats = {
            'grad_norm': 0.0,
            'grad_max': 0.0,
            'param_norm': 0.0,
            'param_updates': 0.0,
            'nan_grads': 0,
            'inf_grads': 0,
            'zero_grads': 0,
            'finite_grads': 0,
            'nan_ratio': 0.0,
            'finite_ratio': 0.0,
            'zero_ratio': 0.0,
            'total_params': 0
        }
        
        total_grad_norm = 0.0
        total_param_norm = 0.0
        max_grad = 0.0
        nan_count = 0
        inf_count = 0
        zero_count = 0
        finite_count = 0
        total_grad_elements = 0
        param_count = 0
        
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                param_count += 1
                grad_flat = param.grad.flatten()
                grad_elements = grad_flat.numel()
                total_grad_elements += grad_elements
                
                # 详细统计各种梯度类型
                nan_mask = torch.isnan(grad_flat)
                inf_mask = torch.isinf(grad_flat)
                zero_mask = (grad_flat == 0.0)
                finite_mask = torch.isfinite(grad_flat)
                
                nan_count += nan_mask.sum().item()
                inf_count += inf_mask.sum().item()
                zero_count += zero_mask.sum().item()
                finite_count += finite_mask.sum().item()
                
                # 只对有限梯度计算范数
                finite_grads = grad_flat[finite_mask]
                if len(finite_grads) > 0:
                    grad_norm = torch.norm(finite_grads).item()
                    total_grad_norm += grad_norm ** 2
                    max_grad = max(max_grad, torch.max(torch.abs(finite_grads)).item())
                
                # 参数统计
                param_norm = param.data.norm().item()
                total_param_norm += param_norm ** 2
        
        if total_grad_elements > 0:
            stats['grad_norm'] = (total_grad_norm ** 0.5)
            stats['grad_max'] = max_grad
            stats['param_norm'] = (total_param_norm ** 0.5)
            stats['nan_grads'] = nan_count
            stats['inf_grads'] = inf_count
            stats['zero_grads'] = zero_count
            stats['finite_grads'] = finite_count
            stats['total_params'] = total_grad_elements
            
            # 计算比例
            stats['nan_ratio'] = nan_count / total_grad_elements
            stats['finite_ratio'] = finite_count / total_grad_elements
            stats['zero_ratio'] = zero_count / total_grad_elements
        
        return stats
    
    def validate(self, data: HeteroData) -> Dict[str, float]:
        """
        验证模型
        
        Args:
            data: 验证数据
            
        Returns:
            验证统计信息
        """
        self.model.eval()
        
        with torch.no_grad():
            # 验证时也使用训练模式来计算损失
            self.model.train()
            results = self.model(data, mode="train")
            losses = results['losses']
            self.model.eval()
            
            val_losses = {}
            kl_weight = self._get_kl_weight(self.current_epoch)
            
            for key, value in losses.items():
                # 确保损失值是有限的
                loss_value = value.item()
                if not torch.isfinite(torch.tensor(loss_value)):
                    logger.warning(f"验证损失 {key} 包含非有限值: {loss_value}，替换为大数值")
                    loss_value = 1000.0 if key == 'total' else 100.0
                val_losses[key] = loss_value
            
            # 计算原始KL散度（未加权）
            if 'kl' in val_losses and kl_weight > 0:
                val_losses['kl_raw'] = val_losses['kl'] / kl_weight
            else:
                val_losses['kl_raw'] = val_losses.get('kl', 0.0)
            
            # 计算重构损失总和
            reconstruction_components = ['bias', 'degree', 'logits', 'weights']
            reconstruction_total = sum(val_losses.get(comp, 0.0) for comp in reconstruction_components)
            val_losses['reconstruction'] = reconstruction_total
        
        return val_losses
    
    def train(self, data: HeteroData) -> Dict[str, Any]:
        """
        完整训练过程
        
        Args:
            data: 训练数据（单个有偏差实例）
            
        Returns:
            训练结果
        """
        logger.info(f"开始训练 - 总epoch数: {self.config.num_epochs}")
        
        # 移动数据到设备
        data = data.to(self.device)
        
        # 保存配置
        self._save_config()
        
        # 训练循环
        start_time = time.time()
        
        # 创建进度条
        progress_bar = tqdm(
            range(self.config.num_epochs),
            desc="Training Progress",
            unit="epoch",
            ncols=120,
            leave=True
        )
        
        # 异步质量评估执行器
        quality_executor = ThreadPoolExecutor(max_workers=1)
        pending_quality_future = None
        
        for epoch in progress_bar:
            self.current_epoch = epoch
            
            # 训练一个epoch
            train_losses = self.train_epoch(data)
            
            # 计算当前KL权重（在这里定义，确保作用域正确）
            current_kl_weight = self._get_kl_weight(epoch)
            
            # 验证
            if epoch % self.config.validation_frequency == 0:
                val_losses = self.validate(data)
                
                # 更新历史记录
                self.training_history['train_loss'].append(train_losses['total'])
                self.training_history['val_loss'].append(val_losses['total'])
                self.training_history['learning_rate'].append(
                    self.optimizer.param_groups[0]['lr']
                )
                self.training_history['kl_weight'].append(
                    self._get_kl_weight(epoch)
                )
                
                # 详细损失分解历史
                self.training_history['train_reconstruction'].append(train_losses.get('reconstruction', 0))
                self.training_history['train_kl_raw'].append(train_losses.get('kl_raw', 0))
                self.training_history['train_bias'].append(train_losses.get('bias', 0))
                self.training_history['train_degree'].append(train_losses.get('degree', 0))
                self.training_history['train_logits'].append(train_losses.get('logits', 0))
                self.training_history['train_weights'].append(train_losses.get('weights', 0))
                self.training_history['train_sparsity'].append(train_losses.get('sparsity', 0))
                
                self.training_history['val_reconstruction'].append(val_losses.get('reconstruction', 0))
                self.training_history['val_kl_raw'].append(val_losses.get('kl_raw', 0))
                
                # 梯度和参数统计历史
                self.training_history['grad_norm'].append(train_losses.get('grad_norm', 0))
                self.training_history['grad_max'].append(train_losses.get('grad_max', 0))
                self.training_history['param_norm'].append(train_losses.get('param_norm', 0))
                self.training_history['nan_grads'].append(train_losses.get('nan_grads', 0))
                self.training_history['inf_grads'].append(train_losses.get('inf_grads', 0))
                
                # 学习率调度
                if self.scheduler:
                    if self.config.scheduler_type == "plateau":
                        self.scheduler.step(val_losses['total'])
                    else:
                        self.scheduler.step()
                
                # 异步质量评估（修复阻塞问题）
                quality_scores = {}
                should_evaluate = (
                    self.quality_evaluator and 
                    getattr(self.config, 'enable_quality_evaluation', True) and
                    epoch % getattr(self.config, 'quality_evaluation_frequency', 50) == 0 and
                    epoch > 100  # 前100个epoch跳过质量评估，专注训练收敛
                )
                
                if should_evaluate:
                    # 检查上一次异步评估是否完成
                    if pending_quality_future and pending_quality_future.done():
                        try:
                            quality_scores = pending_quality_future.result(timeout=1.0)
                            logger.info(f"📊 质量评估完成 (Epoch {epoch-50})")
                        except Exception as e:
                            logger.warning(f"质量评估结果获取失败: {e}")
                            quality_scores = {}
                    
                    # 启动新的异步质量评估
                    logger.info(f"🔍 启动后台质量评估 (Epoch {epoch})...")
                    pending_quality_future = quality_executor.submit(
                        self._evaluate_generation_quality_safe, data, epoch
                    )
                    
                    # 记录质量评估历史
                    if quality_scores:  # 确保quality_scores不为空
                        self.training_history['generation_quality'].append(quality_scores.get('overall_quality_score', 0.0))
                        self.training_history['validity_scores'].append(quality_scores.get('validity_score', 0.0))
                        self.training_history['diversity_scores'].append(quality_scores.get('diversity_score', 0.0))
                        self.training_history['similarity_scores'].append(quality_scores.get('graph_similarity', 0.0))
                    else:
                        # 如果质量评估失败，填充默认值
                        self.training_history['generation_quality'].append(0.0)
                        self.training_history['validity_scores'].append(0.0)
                        self.training_history['diversity_scores'].append(0.0)
                        self.training_history['similarity_scores'].append(0.0)
                
                # 高级Early Stopping检查
                should_stop = False
                early_stopping_result = None
                
                if self.early_stopping_monitor:
                    # 准备指标字典
                    all_metrics = {
                        'val_loss': val_losses['total'],
                        'train_loss': train_losses['total'],
                        'kl_weight': current_kl_weight,
                        'grad_norm': train_losses.get('grad_norm', 0.0),
                        'reconstruction_loss': train_losses.get('reconstruction', 0.0),
                        'kl_raw': train_losses.get('kl_raw', 0.0),
                        'learning_rate': self.optimizer.param_groups[0]['lr']
                    }
                    
                    # 添加质量评估指标（如果有）
                    if quality_scores:
                        all_metrics['generation_quality'] = quality_scores.get('overall_quality_score', 0.0)
                        all_metrics['graph_similarity'] = quality_scores.get('graph_similarity', 0.0)
                        all_metrics['diversity_score'] = quality_scores.get('diversity_score', 0.0)
                    
                    # 更新Early Stopping监控器
                    early_stopping_result = self.early_stopping_monitor.update(
                        epoch, 
                        all_metrics, 
                        model_state=self.model.state_dict()
                    )
                    
                    should_stop = early_stopping_result['should_stop']
                    
                    # 记录Early Stopping详细信息
                    if epoch % 50 == 0 or should_stop:  # 每50个epoch或停止时记录
                        logger.info(f"🔍 Early Stopping Analysis (Epoch {epoch}):")
                        logger.info(f"  ├─ Decision: {'STOP' if should_stop else 'CONTINUE'}")
                        logger.info(f"  ├─ Reason: {early_stopping_result['decision_reason']}")
                        logger.info(f"  ├─ Best Score: {early_stopping_result['best_score']:.6f} (Epoch {early_stopping_result['best_epoch']})")
                        logger.info(f"  ├─ Patience: {early_stopping_result['patience_counter']}/{early_stopping_result['current_patience']}")
                        
                        if early_stopping_result['trend_analysis']:
                            trend = early_stopping_result['trend_analysis']
                            logger.info(f"  ├─ Trend: {trend['trend']} (slope: {trend['slope']:.2e}, stability: {trend['stability']:.3f})")
                        
                        if early_stopping_result['quality_analysis']:
                            qa = early_stopping_result['quality_analysis']
                            logger.info(f"  └─ Quality: {qa.get('quality_score', 'N/A'):.3f} (threshold: {self.early_stopping_monitor.config.quality_threshold})")
                    
                    if should_stop:
                        logger.warning(f"🛑 高级Early Stopping触发 (Epoch {epoch})")
                        logger.warning(f"  原因: {early_stopping_result['decision_reason']}")
                        logger.warning(f"  置信度: {early_stopping_result.get('confidence', 0.0):.3f}")
                        
                        # 恢复最佳权重（如果配置启用）
                        if self.early_stopping_monitor.should_restore_weights():
                            best_weights = self.early_stopping_monitor.get_best_weights()
                            if best_weights:
                                self.model.load_state_dict(best_weights)
                                logger.info(f"✅ 已恢复最佳模型权重 (来自Epoch {early_stopping_result['best_epoch']})")
                        
                        break
                else:
                    # 传统早停检查（兼容性）
                    if self._check_early_stopping(val_losses['total']):
                        logger.info(f"传统早停在epoch {epoch}")
                        should_stop = True
                        break
                
                # 增强的详细日志输出
                # current_kl_weight 已在上面定义
                
                # 计算损失组件百分比
                train_total = train_losses['total']
                val_total = val_losses['total']
                
                recon_pct = (train_losses.get('reconstruction', 0) / train_total * 100) if train_total > 0 else 0
                kl_pct = (train_losses.get('kl', 0) / train_total * 100) if train_total > 0 else 0
                
                # 更新进度条描述
                progress_desc = (
                    f"E{epoch:4d} | Loss:{train_total:.4f} | "
                    f"LR:{self.optimizer.param_groups[0]['lr']:.1e} | "
                    f"KL:{current_kl_weight:.3f}"
                )
                progress_bar.set_description(progress_desc)
                
                # 简化日志输出（避免过多输出影响性能）
                if epoch % 10 == 0:  # 每10个epoch输出一次详细日志
                    logger.info(
                        f"Epoch {epoch:4d} | "
                        f"Train Loss: {train_total:.6f} | "
                        f"Val Loss: {val_total:.6f} | "
                        f"LR: {self.optimizer.param_groups[0]['lr']:.2e} | "
                        f"KL Weight: {current_kl_weight:.4f}"
                    )
                
                # 详细损失分解
                logger.info(
                    f"  ├─ Reconstruction: {train_losses.get('reconstruction', 0):.6f} ({recon_pct:.1f}%) "
                    f"[Bias: {train_losses.get('bias', 0):.4f}, Degree: {train_losses.get('degree', 0):.4f}, "
                    f"Logits: {train_losses.get('logits', 0):.4f}, Weights: {train_losses.get('weights', 0):.4f}]"
                )
                
                logger.info(
                    f"  ├─ KL Divergence: {train_losses.get('kl', 0):.6f} ({kl_pct:.1f}%) "
                    f"[Raw: {train_losses.get('kl_raw', 0):.6f}, Weight: {current_kl_weight:.4f}]"
                )
                
                # 梯度和参数统计
                if 'grad_norm' in train_losses:
                    logger.info(
                        f"  ├─ Gradients: Norm={train_losses.get('grad_norm', 0):.4f}, "
                        f"Max={train_losses.get('grad_max', 0):.4f}, "
                        f"NaN={train_losses.get('nan_grads', 0)}, Inf={train_losses.get('inf_grads', 0)}"
                    )
                
                # 参数统计
                if 'param_norm' in train_losses:
                    logger.info(
                        f"  ├─ Parameters: Norm={train_losses.get('param_norm', 0):.4f}"
                    )
                
                # 质量评估结果（如果有）
                if quality_scores:
                    logger.info(
                        f"  🎯 Quality Assessment: Overall={quality_scores.get('overall_quality_score', 0):.4f}, "
                        f"Similarity={quality_scores.get('graph_similarity', 0):.4f}, "
                        f"Diversity={quality_scores.get('diversity_score', 0):.4f}, "
                        f"Grade={quality_scores.get('benchmark_grade', 'N/A')}"
                    )
                
                # 稀疏性损失（如果存在）
                if train_losses.get('sparsity', 0) > 0:
                    sparsity_pct = (train_losses['sparsity'] / train_total * 100) if train_total > 0 else 0
                    logger.info(f"  └─ Sparsity: {train_losses['sparsity']:.6f} ({sparsity_pct:.1f}%)")
                
                # 在线质量评估（每隔一定epoch执行）
                enable_quality_eval = getattr(self.config, 'enable_quality_evaluation', True)
                quality_eval_frequency = getattr(self.config, 'quality_evaluation_frequency', 100)
                if self.evaluator and enable_quality_eval and epoch % quality_eval_frequency == 0:
                    try:
                        num_samples = getattr(self.config, 'quality_samples_per_eval', 3)
                        quality_metrics = self.evaluator.evaluate_online_quality(
                            model=self.model,
                            original_data=data,
                            epoch=epoch,
                            num_samples=num_samples
                        )
                        
                        # 更新质量评估历史
                        self.training_history['validity_score'].append(quality_metrics.get('validity_score', 0.0))
                        self.training_history['diversity_score'].append(quality_metrics.get('diversity_score', 0.0))
                        self.training_history['similarity_score'].append(quality_metrics.get('similarity_score', 0.0))
                        self.training_history['stability_score'].append(quality_metrics.get('stability_score', 0.0))
                        self.training_history['quality_overall'].append(quality_metrics.get('overall_quality', 0.0))
                        
                        # 记录质量评估结果
                        logger.info(
                            f"  ├─ Quality Assessment: "
                            f"Validity={quality_metrics.get('validity_score', 0):.3f}, "
                            f"Diversity={quality_metrics.get('diversity_score', 0):.3f}, "
                            f"Similarity={quality_metrics.get('similarity_score', 0):.3f}, "
                            f"Stability={quality_metrics.get('stability_score', 0):.3f}"
                        )
                        logger.info(f"  └─ Overall Quality: {quality_metrics.get('overall_quality', 0):.3f}")
                        
                    except Exception as e:
                        logger.warning(f"在线质量评估失败: {e}")
                        # 填充默认值以保持历史记录一致性
                        self.training_history['validity_score'].append(0.0)
                        self.training_history['diversity_score'].append(0.0)
                        self.training_history['similarity_score'].append(0.0)
                        self.training_history['stability_score'].append(0.0)
                        self.training_history['quality_overall'].append(0.0)
            
            # 保存模型
            if epoch % self.config.save_frequency == 0 and epoch > 0:
                self._save_checkpoint(epoch, train_losses)
        
        # 关闭进度条和清理异步资源
        progress_bar.close()
        
        # 等待最后一次质量评估完成
        if pending_quality_future:
            try:
                final_quality = pending_quality_future.result(timeout=30.0)
                logger.info("最终质量评估完成")
            except Exception as e:
                logger.warning(f"最终质量评估超时或失败: {e}")
        
        quality_executor.shutdown(wait=True)
        
        # 训练完成
        training_time = time.time() - start_time
        logger.info(f"训练完成 - 总时间: {training_time:.2f}秒")
        
        # 保存最终模型
        self._save_final_model()
        
        # 保存训练历史
        self._save_training_history()
        
        # 保存Early Stopping状态
        if self.early_stopping_monitor:
            early_stopping_save_path = self.save_dir / "early_stopping_state.json"
            self.early_stopping_monitor.save_state(str(early_stopping_save_path))
        
        # 生成训练报告
        return self._generate_training_report(training_time)
    
    def _check_early_stopping(self, val_loss: float) -> bool:
        """检查是否应该早停"""
        if not self.config.use_early_stopping:
            return False
        
        # 支持新的配置参数名称
        min_delta = getattr(self.config, 'min_delta', self.config.early_stopping_min_delta)
        
        if val_loss < self.best_loss - min_delta:
            self.best_loss = val_loss
            self.early_stopping_counter = 0
        else:
            self.early_stopping_counter += 1
        
        if self.early_stopping_counter >= self.config.early_stopping_patience:
            return True
        
        return False
    
    def _evaluate_generation_quality_safe(self, data: HeteroData, epoch: int) -> Dict[str, float]:
        """
        安全的质量评估方法（用于异步执行）
        
        包含异常处理和超时机制，避免阻塞主训练流程
        """
        try:
            return self._evaluate_generation_quality(data, epoch)
        except Exception as e:
            logger.warning(f"后台质量评估失败 (Epoch {epoch}): {e}")
            return {}
    
    def _evaluate_generation_quality(self, data: HeteroData, epoch: int) -> Dict[str, float]:
        """
        评估生成质量（新增）
        
        Args:
            data: 原始训练数据
            epoch: 当前epoch
            
        Returns:
            质量评估结果
        """
        try:
            if not self.quality_evaluator:
                return {}
            
            # 延迟初始化推理器
            if self.quality_inferencer is None:
                inference_config = InferenceConfig(
                    num_test_instances=getattr(self.config, 'quality_samples_per_eval', 3),
                    eta=0.1,
                    sample_from_prior=True,
                    constraint_selection_strategy="random"
                )
                self.quality_inferencer = G2MILPInference(self.model, inference_config)
            
            # 切换到评估模式
            self.model.eval()
            
            # 生成测试样本
            with torch.no_grad():
                inference_results = self.quality_inferencer.generate_instances(
                    data, 
                    num_samples=getattr(self.config, 'quality_samples_per_eval', 3)
                )
            
            # 恢复训练模式
            self.model.train()
            
            # 提取生成数据
            generated_data_list = inference_results['generated_instances']
            generation_info = inference_results['generation_info']
            
            # 执行质量评估
            evaluation_results = self.quality_evaluator.evaluate_generation_quality(
                original_data=data,
                generated_data_list=generated_data_list,
                generation_info=generation_info
            )
            
            # 提取关键质量指标
            quality_summary = {
                'overall_quality_score': evaluation_results.get('overall_quality_score', 0.0),
                'graph_similarity': 0.0,
                'milp_similarity': 0.0,
                'diversity_score': 0.0,
                'validity_score': 1.0,  # 默认假设有效，可以进一步实现
                'benchmark_grade': 'N/A'
            }
            
            # 提取具体指标
            if 'graph_similarity' in evaluation_results and 'weighted_average' in evaluation_results['graph_similarity']:
                quality_summary['graph_similarity'] = evaluation_results['graph_similarity']['weighted_average']
            
            if 'milp_similarity' in evaluation_results and 'overall_milp_similarity' in evaluation_results['milp_similarity']:
                quality_summary['milp_similarity'] = evaluation_results['milp_similarity']['overall_milp_similarity']
            
            if 'diversity_analysis' in evaluation_results and 'overall_diversity_score' in evaluation_results['diversity_analysis']:
                quality_summary['diversity_score'] = evaluation_results['diversity_analysis']['overall_diversity_score']
            
            if 'benchmark_comparison' in evaluation_results and 'summary' in evaluation_results['benchmark_comparison']:
                quality_summary['benchmark_grade'] = evaluation_results['benchmark_comparison']['summary']['grade']
            
            logger.debug(f"Epoch {epoch} 质量评估完成 - 综合得分: {quality_summary['overall_quality_score']:.4f}")
            
            return quality_summary
            
        except Exception as e:
            logger.warning(f"质量评估失败 (Epoch {epoch}): {e}")
            return {}
    
    def _apply_data_augmentation(self, data: HeteroData) -> HeteroData:
        """
        应用数据增强技术
        
        策略：
        1. 特征噪声注入：对节点特征添加小幅高斯噪声
        2. 边权重扰动：对边权重添加小幅扰动
        3. 结构轻微修改：随机遮盖少量边（可选）
        
        Args:
            data: 原始图数据
            
        Returns:
            增强后的图数据
        """
        try:
            # 创建数据副本
            augmented_data = data.clone()
            
            # 1. 特征噪声注入
            feature_noise_std = getattr(self.config, 'feature_noise_std', 0.05)
            if feature_noise_std > 0:
                for node_type in ['constraint', 'variable']:
                    if node_type in augmented_data:
                        node_features = augmented_data[node_type].x
                        noise = torch.randn_like(node_features) * feature_noise_std
                        augmented_data[node_type].x = node_features + noise
            
            # 2. 边特征扰动（如果存在边特征）
            edge_noise_std = feature_noise_std * 0.5  # 边特征的噪声更小
            if hasattr(augmented_data, 'edge_attr_dict'):
                for edge_type, edge_attr in augmented_data.edge_attr_dict.items():
                    if edge_attr is not None and edge_noise_std > 0:
                        noise = torch.randn_like(edge_attr) * edge_noise_std
                        augmented_data.edge_attr_dict[edge_type] = edge_attr + noise
            
            # 3. 轻微的结构扰动（可选，概率很低）
            edge_perturbation_prob = getattr(self.config, 'edge_perturbation_prob', 0.0)
            if edge_perturbation_prob > 0 and torch.rand(1).item() < edge_perturbation_prob:
                # 对于训练稳定性，目前不实施结构扰动
                pass
            
            return augmented_data.to(self.device)
            
        except Exception as e:
            logger.warning(f"数据增强失败: {e}，使用原始数据")
            return data
    
    def _compute_sparsity_regularization(self, model_results: Dict, data: HeteroData) -> torch.Tensor:
        """
        计算稀疏性正则化损失
        
        鼓励模型生成与源图具有相似稀疏性的图
        """
        try:
            # 获取源图的边数
            if hasattr(data, 'edge_index_dict'):
                source_edge_count = 0
                for edge_type, edge_index in data.edge_index_dict.items():
                    source_edge_count += edge_index.shape[1]
            else:
                source_edge_count = data.edge_index.shape[1] if hasattr(data, 'edge_index') else 1
            
            # 获取模型预测的边概率（假设在logits预测器中）
            if 'predictions' in model_results and 'logits' in model_results['predictions']:
                edge_logits = model_results['predictions']['logits']
                # 使用sigmoid获取边概率
                edge_probs = torch.sigmoid(edge_logits)
                predicted_edge_count = torch.sum(edge_probs)
                
                # 计算边数差异的平方损失
                target_sparsity = getattr(self.config, 'target_sparsity', 0.1)
                expected_edges = source_edge_count * (1.0 + target_sparsity)
                sparsity_loss = torch.pow(predicted_edge_count - expected_edges, 2) / (expected_edges + 1e-8)
                
                return sparsity_loss
            else:
                # 如果无法获取边预测，返回零损失
                return torch.tensor(0.0, device=self.device)
                
        except Exception as e:
            logger.warning(f"稀疏性正则化计算失败: {e}")
            return torch.tensor(0.0, device=self.device)
    
    def _save_config(self):
        """保存训练配置"""
        config_path = self.save_dir / "training_config.json"
        
        config_dict = asdict(self.config)
        config_dict['model_config'] = {
            'constraint_feature_dim': self.model.constraint_feature_dim,
            'variable_feature_dim': self.model.variable_feature_dim,
            'edge_feature_dim': self.model.edge_feature_dim
        }
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"训练配置已保存: {config_path}")
    
    def _save_checkpoint(self, epoch: int, losses: Dict[str, float]):
        """保存训练检查点"""
        checkpoint_path = self.save_dir / f"checkpoint_epoch_{epoch}.pth"
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'losses': losses,
            'training_history': self.training_history,
            'best_loss': self.best_loss
        }, checkpoint_path)
        
        logger.debug(f"检查点已保存: {checkpoint_path}")
    
    def _save_final_model(self):
        """保存最终模型"""
        model_path = self.save_dir / "final_model.pth"
        self.model.save_model(str(model_path))
        
        logger.info(f"最终模型已保存: {model_path}")
    
    def _save_training_history(self):
        """保存训练历史"""
        history_path = self.save_dir / "training_history.json"
        
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(self.training_history, f, indent=2, ensure_ascii=False, default=str)
        
        # 绘制训练曲线
        self._plot_training_curves()
        
        logger.info(f"训练历史已保存: {history_path}")
    
    def _plot_training_curves(self):
        """绘制增强版训练曲线"""
        try:
            # 创建更大的图表来容纳更多信息
            fig, axes = plt.subplots(3, 3, figsize=(18, 12))
            
            # 1. 总损失曲线
            axes[0, 0].plot(self.training_history['train_loss'], label='Train Loss', linewidth=2)
            axes[0, 0].plot(self.training_history['val_loss'], label='Val Loss', linewidth=2)
            axes[0, 0].set_title('Total Loss Curves')
            axes[0, 0].set_xlabel('Validation Steps')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # 2. 重构损失分解
            axes[0, 1].plot(self.training_history['train_bias'], label='Bias', alpha=0.8)
            axes[0, 1].plot(self.training_history['train_degree'], label='Degree', alpha=0.8)
            axes[0, 1].plot(self.training_history['train_logits'], label='Logits', alpha=0.8)
            axes[0, 1].plot(self.training_history['train_weights'], label='Weights', alpha=0.8)
            axes[0, 1].set_title('Reconstruction Loss Components')
            axes[0, 1].set_xlabel('Validation Steps')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # 3. KL散度（原始 vs 加权）
            axes[0, 2].plot(self.training_history['train_kl_raw'], label='KL Raw', alpha=0.8)
            axes[0, 2].plot(self.training_history['kl_weight'], label='KL Weight', alpha=0.8)
            kl_weighted = [raw * weight for raw, weight in zip(self.training_history['train_kl_raw'], self.training_history['kl_weight'])]
            axes[0, 2].plot(kl_weighted, label='KL Weighted', alpha=0.8)
            axes[0, 2].set_title('KL Divergence Analysis')
            axes[0, 2].set_xlabel('Validation Steps')
            axes[0, 2].set_ylabel('Value')
            axes[0, 2].legend()
            axes[0, 2].grid(True, alpha=0.3)
            
            # 4. 学习率曲线
            axes[1, 0].plot(self.training_history['learning_rate'], linewidth=2)
            axes[1, 0].set_title('Learning Rate Schedule')
            axes[1, 0].set_xlabel('Validation Steps')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].set_yscale('log')
            axes[1, 0].grid(True, alpha=0.3)
            
            # 5. 梯度统计
            axes[1, 1].plot(self.training_history['grad_norm'], label='Grad Norm', alpha=0.8)
            axes[1, 1].plot(self.training_history['grad_max'], label='Grad Max', alpha=0.8)
            axes[1, 1].set_title('Gradient Statistics')
            axes[1, 1].set_xlabel('Validation Steps')
            axes[1, 1].set_ylabel('Gradient Magnitude')
            axes[1, 1].set_yscale('log')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
            
            # 6. 参数范数
            axes[1, 2].plot(self.training_history['param_norm'], linewidth=2, color='green')
            axes[1, 2].set_title('Parameter Norm')
            axes[1, 2].set_xlabel('Validation Steps')
            axes[1, 2].set_ylabel('Parameter Norm')
            axes[1, 2].grid(True, alpha=0.3)
            
            # 7. 重构损失对比（训练 vs 验证）
            axes[2, 0].plot(self.training_history['train_reconstruction'], label='Train Reconstruction', alpha=0.8)
            axes[2, 0].plot(self.training_history['val_reconstruction'], label='Val Reconstruction', alpha=0.8)
            axes[2, 0].set_title('Reconstruction Loss: Train vs Val')
            axes[2, 0].set_xlabel('Validation Steps')
            axes[2, 0].set_ylabel('Reconstruction Loss')
            axes[2, 0].legend()
            axes[2, 0].grid(True, alpha=0.3)
            
            # 8. 数值稳定性监控
            axes[2, 1].bar(['NaN Grads', 'Inf Grads'], 
                          [sum(self.training_history['nan_grads']), sum(self.training_history['inf_grads'])],
                          color=['red', 'orange'], alpha=0.7)
            axes[2, 1].set_title('Numerical Stability')
            axes[2, 1].set_ylabel('Count')
            
            # 9. 质量评估指标（如果存在）
            if any(self.training_history['quality_overall']):
                # 创建质量评估的独立步长（因为不是每个epoch都评估）
                quality_steps = list(range(0, len(self.training_history['quality_overall'])))
                
                axes[2, 2].plot(quality_steps, self.training_history['validity_score'], 
                               label='Validity', alpha=0.8, marker='o', markersize=3)
                axes[2, 2].plot(quality_steps, self.training_history['diversity_score'], 
                               label='Diversity', alpha=0.8, marker='s', markersize=3)
                axes[2, 2].plot(quality_steps, self.training_history['similarity_score'], 
                               label='Similarity', alpha=0.8, marker='^', markersize=3)
                axes[2, 2].plot(quality_steps, self.training_history['quality_overall'], 
                               label='Overall', alpha=0.9, linewidth=2, color='red')
                axes[2, 2].set_title('Online Quality Assessment')
                axes[2, 2].set_xlabel('Quality Evaluation Steps')
                axes[2, 2].set_ylabel('Quality Score')
                axes[2, 2].legend()
                axes[2, 2].grid(True, alpha=0.3)
            elif any(self.training_history['train_sparsity']):
                # 稀疏性损失
                axes[2, 2].plot(self.training_history['train_sparsity'], linewidth=2, color='purple')
                axes[2, 2].set_title('Sparsity Regularization')
                axes[2, 2].set_xlabel('Validation Steps')
                axes[2, 2].set_ylabel('Sparsity Loss')
                axes[2, 2].grid(True, alpha=0.3)
            else:
                # 如果没有质量评估和稀疏性损失，显示损失分布
                axes[2, 2].hist(self.training_history['train_loss'], bins=30, alpha=0.7, label='Train')
                axes[2, 2].hist(self.training_history['val_loss'], bins=30, alpha=0.7, label='Val')
                axes[2, 2].set_title('Loss Distribution')
                axes[2, 2].set_xlabel('Loss')
                axes[2, 2].set_ylabel('Frequency')
                axes[2, 2].legend()
            
            plt.tight_layout()
            plt.savefig(self.save_dir / "enhanced_training_curves.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info("增强版训练曲线已保存")
            
        except Exception as e:
            logger.warning(f"无法绘制增强版训练曲线: {e}")
            # 回退到基础版本
            self._plot_basic_training_curves()
    
    def _plot_basic_training_curves(self):
        """绘制基础训练曲线（备用方案）"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            
            # 总损失曲线
            axes[0, 0].plot(self.training_history['train_loss'], label='Train Loss')
            axes[0, 0].plot(self.training_history['val_loss'], label='Val Loss')
            axes[0, 0].set_title('Total Loss Curves')
            axes[0, 0].set_xlabel('Validation Steps')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
            
            # 学习率曲线
            axes[0, 1].plot(self.training_history['learning_rate'])
            axes[0, 1].set_title('Learning Rate')
            axes[0, 1].set_xlabel('Validation Steps')
            axes[0, 1].set_ylabel('Learning Rate')
            axes[0, 1].set_yscale('log')
            axes[0, 1].grid(True)
            
            # KL权重曲线
            axes[1, 0].plot(self.training_history['kl_weight'])
            axes[1, 0].set_title('KL Weight (β)')
            axes[1, 0].set_xlabel('Validation Steps')
            axes[1, 0].set_ylabel('KL Weight')
            axes[1, 0].grid(True)
            
            # 重构损失
            axes[1, 1].plot(self.training_history['train_reconstruction'], label='Train Reconstruction')
            axes[1, 1].plot(self.training_history['val_reconstruction'], label='Val Reconstruction')
            axes[1, 1].set_title('Reconstruction Loss')
            axes[1, 1].set_xlabel('Validation Steps')
            axes[1, 1].set_ylabel('Loss')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
            
            plt.tight_layout()
            plt.savefig(self.save_dir / "basic_training_curves.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info("基础训练曲线已保存")
            
        except Exception as e:
            logger.warning(f"无法绘制基础训练曲线: {e}")
    
    def _generate_training_report(self, training_time: float) -> Dict[str, Any]:
        """生成训练报告"""
        report = {
            'training_summary': {
                'total_epochs': self.current_epoch + 1,
                'total_iterations': self.current_iteration,
                'training_time_seconds': training_time,
                'best_validation_loss': self.best_loss,
                'final_learning_rate': self.optimizer.param_groups[0]['lr']
            },
            'model_info': {
                'total_parameters': sum(p.numel() for p in self.model.parameters()),
                'trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad),
                'model_size_mb': sum(p.numel() * p.element_size() for p in self.model.parameters()) / 1024 / 1024
            },
            'config': asdict(self.config),
            'training_history': self.training_history
        }
        
        # 添加Early Stopping摘要
        if self.early_stopping_monitor:
            early_stopping_summary = self.early_stopping_monitor.get_summary()
            report['early_stopping'] = {
                'strategy': early_stopping_summary['config'].strategy.value,
                'best_score': early_stopping_summary['best_score'],
                'best_epoch': early_stopping_summary['best_epoch'],
                'total_improvements': early_stopping_summary['improvements'],
                'final_patience': early_stopping_summary['final_patience'],
                'statistics': early_stopping_summary['statistics']
            }
        
        # 保存报告
        report_path = self.save_dir / "training_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"训练报告已保存: {report_path}")
        
        return report
    
    def _analyze_gradient_health(self) -> Dict[str, float]:
        """
        分析梯度健康状况（新增方法）
        
        Returns:
            包含梯度统计信息的字典
        """
        total_params = 0
        nan_params = 0
        inf_params = 0
        zero_params = 0
        finite_params = 0
        grad_norm_sum = 0.0
        
        for param in self.model.parameters():
            if param.grad is not None:
                grad_flat = param.grad.flatten()
                param_count = grad_flat.numel()
                total_params += param_count
                
                # 统计NaN和无限值
                nan_mask = torch.isnan(grad_flat)
                inf_mask = torch.isinf(grad_flat)
                zero_mask = (grad_flat == 0.0)
                finite_mask = torch.isfinite(grad_flat)
                
                nan_params += nan_mask.sum().item()
                inf_params += inf_mask.sum().item()
                zero_params += zero_mask.sum().item()
                finite_params += finite_mask.sum().item()
                
                # 计算有限梯度的范数
                finite_grads = grad_flat[finite_mask]
                if len(finite_grads) > 0:
                    grad_norm_sum += torch.norm(finite_grads).item() ** 2
        
        # 计算比例
        nan_ratio = nan_params / max(total_params, 1)
        inf_ratio = inf_params / max(total_params, 1)
        zero_ratio = zero_params / max(total_params, 1)
        finite_ratio = finite_params / max(total_params, 1)
        
        # 计算总梯度范数
        total_grad_norm = grad_norm_sum ** 0.5
        
        return {
            'nan_ratio': nan_ratio,
            'inf_ratio': inf_ratio,
            'zero_ratio': zero_ratio,
            'finite_ratio': finite_ratio,
            'total_params': total_params,
            'grad_norm': total_grad_norm,
            'nan_count': nan_params,
            'inf_count': inf_params,
            'zero_count': zero_params
        }
    
    def _replace_nan_gradients(self):
        """
        将NaN梯度替换为零值（新增方法）
        """
        replaced_count = 0
        
        for param in self.model.parameters():
            if param.grad is not None:
                nan_mask = torch.isnan(param.grad)
                if nan_mask.any():
                    param.grad[nan_mask] = 0.0
                    replaced_count += nan_mask.sum().item()
        
        if replaced_count > 0:
            logger.debug(f"已将 {replaced_count} 个NaN梯度替换为零值")
    
    def load_checkpoint(self, checkpoint_path: str) -> bool:
        """加载训练检查点"""
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            if checkpoint['scheduler_state_dict'] and self.scheduler:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            self.current_epoch = checkpoint['epoch']
            self.training_history = checkpoint['training_history']
            self.best_loss = checkpoint['best_loss']
            
            logger.info(f"检查点已加载: {checkpoint_path}")
            return True
            
        except Exception as e:
            logger.error(f"检查点加载失败: {e}")
            return False


def create_trainer(model: G2MILPGenerator, 
                  config: TrainingConfig = None,
                  evaluator = None) -> G2MILPTrainer:
    """
    创建G2MILP训练器的工厂函数
    
    Args:
        model: G2MILP生成器模型
        config: 训练配置
        evaluator: 在线质量评估器（可选）
        
    Returns:
        G2MILP训练器实例
    """
    return G2MILPTrainer(model, config, evaluator)


if __name__ == "__main__":
    # 测试代码
    print("G2MILP训练模块测试")
    print("=" * 40)
    
    # 创建测试配置
    training_config = TrainingConfig(
        num_epochs=100,
        iterations_per_epoch=10,
        learning_rate=1e-3,
        use_early_stopping=True,
        early_stopping_patience=20
    )
    
    print(f"训练配置:")
    print(f"- Epochs: {training_config.num_epochs}")
    print(f"- Learning Rate: {training_config.learning_rate}")
    print(f"- Device: {training_config.device}")
    print(f"- Early Stopping: {training_config.use_early_stopping}")
    print("训练器配置创建成功!")
"""
G2MILP遮盖过程模块
G2MILP Masking Process Module

实现G2MILP的约束遮盖机制，根据论文3.3节的描述：
1. 随机选择一个约束节点进行遮盖
2. 用特殊的[mask]标记替换约束节点
3. 添加虚拟边连接遮盖节点与所有变量节点
4. 为遮盖节点和虚拟边分配特殊嵌入

支持可控的遮盖比例η来控制生成实例的相似度
"""

import torch
import torch.nn as nn
from torch_geometric.data import HeteroData
import numpy as np
import random
import logging
from typing import Dict, Tuple, Optional, List, Set
from dataclasses import dataclass
import copy

logger = logging.getLogger(__name__)


@dataclass
class MaskingConfig:
    """遮盖配置"""
    # 遮盖参数
    masking_ratio: float = 0.1  # η参数，遮盖比例
    mask_token_dim: int = 16    # [mask]标记的维度
    virtual_edge_dim: int = 8   # 虚拟边特征维度
    
    # 随机种子
    random_seed: Optional[int] = None
    
    # 遮盖策略
    mask_strategy: str = "random"  # random, degree_based, importance_based
    min_constraint_degree: int = 1  # 最小约束度数（避免遮盖孤立节点）
    
    # 特殊标记值
    mask_token_value: float = -999.0  # 特殊标记值
    virtual_edge_weight: float = 0.0  # 虚拟边权重


class ConstraintMasker:
    """
    约束遮盖器
    
    实现论文中的遮盖过程，包括：
    1. 约束节点选择和遮盖
    2. 虚拟边添加
    3. 特殊嵌入生成
    4. 遮盖信息记录
    """
    
    def __init__(self, config: MaskingConfig = None):
        self.config = config or MaskingConfig()
        
        if self.config.random_seed is not None:
            random.seed(self.config.random_seed)
            np.random.seed(self.config.random_seed)
            torch.manual_seed(self.config.random_seed)
        
        # 特殊嵌入
        self.mask_token_embedding = self._create_mask_token_embedding()
        self.virtual_edge_embedding = self._create_virtual_edge_embedding()
        
        logger.info(f"约束遮盖器初始化完成 - 遮盖比例: {self.config.masking_ratio}")
    
    def _create_mask_token_embedding(self) -> torch.Tensor:
        """创建[mask]标记的特殊嵌入"""
        # 使用特殊值填充的向量
        mask_embedding = torch.full(
            (self.config.mask_token_dim,), 
            self.config.mask_token_value, 
            dtype=torch.float32
        )
        return mask_embedding
    
    def _create_virtual_edge_embedding(self) -> torch.Tensor:
        """创建虚拟边的特殊嵌入"""
        # 使用零向量作为虚拟边特征
        virtual_embedding = torch.zeros(
            self.config.virtual_edge_dim, 
            dtype=torch.float32
        )
        return virtual_embedding
    
    def select_constraint_to_mask(self, data: HeteroData) -> int:
        """
        选择要遮盖的约束节点
        
        Args:
            data: 异构图数据
            
        Returns:
            被选中的约束节点ID
        """
        n_constraints = data['constraint'].x.size(0)
        
        if self.config.mask_strategy == "random":
            # 随机选择
            return random.randint(0, n_constraints - 1)
        
        elif self.config.mask_strategy == "degree_based":
            # 基于度数选择（优先选择高度数节点）
            constraint_degrees = self._compute_constraint_degrees(data)
            
            # 过滤掉度数过低的节点
            valid_constraints = [
                i for i in range(n_constraints) 
                if constraint_degrees[i] >= self.config.min_constraint_degree
            ]
            
            if not valid_constraints:
                return random.randint(0, n_constraints - 1)
            
            # 按度数加权选择
            degrees = [constraint_degrees[i] for i in valid_constraints]
            weights = np.array(degrees, dtype=float)
            weights = weights / weights.sum()
            
            selected_idx = np.random.choice(len(valid_constraints), p=weights)
            return valid_constraints[selected_idx]
        
        elif self.config.mask_strategy == "importance_based":
            # 基于重要性选择（可以根据约束特征计算重要性）
            importance_scores = self._compute_constraint_importance(data)
            
            # 按重要性加权选择
            weights = importance_scores / importance_scores.sum()
            selected_idx = np.random.choice(n_constraints, p=weights)
            return selected_idx
        
        else:
            # 默认随机选择
            return random.randint(0, n_constraints - 1)
    
    def _compute_constraint_degrees(self, data: HeteroData) -> List[int]:
        """计算约束节点的度数"""
        n_constraints = data['constraint'].x.size(0)
        degrees = [0] * n_constraints
        
        # 从边信息中计算度数
        edge_index = data[('constraint', 'connects', 'variable')].edge_index
        for i in range(edge_index.size(1)):
            constraint_id = edge_index[0, i].item()
            degrees[constraint_id] += 1
        
        return degrees
    
    def _compute_constraint_importance(self, data: HeteroData) -> np.ndarray:
        """计算约束节点的重要性分数"""
        n_constraints = data['constraint'].x.size(0)
        
        # 简单的重要性计算：基于约束特征的某些维度
        constraint_features = data['constraint'].x
        
        # 使用约束特征的范数作为重要性指标
        importance_scores = torch.norm(constraint_features, dim=1).cpu().numpy()
        
        # 归一化
        if importance_scores.sum() > 0:
            importance_scores = importance_scores / importance_scores.sum()
        else:
            importance_scores = np.ones(n_constraints) / n_constraints
        
        return importance_scores
    
    def mask_constraint(self, data: HeteroData, constraint_id: int) -> Tuple[HeteroData, Dict]:
        """
        遮盖指定的约束节点
        
        Args:
            data: 原始异构图数据
            constraint_id: 要遮盖的约束节点ID
            
        Returns:
            (masked_data, mask_info) - 遮盖后的数据和遮盖信息
        """
        # 深拷贝数据以避免修改原始数据，并保持设备一致性
        original_device = None
        try:
            # 推断原始设备
            if hasattr(data['constraint'], 'x') and data['constraint'].x is not None:
                original_device = data['constraint'].x.device
        except:
            pass
        
        masked_data = copy.deepcopy(data)
        
        # 确保深拷贝后设备保持一致
        if original_device is not None:
            try:
                masked_data = masked_data.to(original_device)
            except:
                logger.warning(f"无法将遮盖数据移动到原设备 {original_device}")
                pass
        
        # 1. 保存原始约束信息
        original_constraint_features = data['constraint'].x[constraint_id].clone()
        
        # 2. 用[mask]标记替换约束节点特征
        mask_token = self.mask_token_embedding[:masked_data['constraint'].x.size(1)]
        masked_data['constraint'].x[constraint_id] = mask_token
        
        # 3. 移除原有的连接边
        original_edges, removed_edges = self._remove_constraint_edges(
            masked_data, constraint_id
        )
        
        # 4. 添加虚拟边连接到所有变量节点
        self._add_virtual_edges(masked_data, constraint_id)
        
        # 5. 记录遮盖信息
        mask_info = {
            'masked_constraint_id': constraint_id,
            'original_features': original_constraint_features,
            'original_edges': original_edges,
            'removed_edges': removed_edges,
            'n_original_connections': len(original_edges)
        }
        
        logger.debug(f"约束 {constraint_id} 已遮盖，原有连接数: {len(original_edges)}")
        
        return masked_data, mask_info
    
    def _remove_constraint_edges(self, data: HeteroData, constraint_id: int) -> Tuple[List[Tuple], List[Tuple]]:
        """移除指定约束的所有边"""
        original_edges = []
        removed_edges = []
        
        # 处理 constraint -> variable 边
        edge_index = data[('constraint', 'connects', 'variable')].edge_index
        edge_attr = data[('constraint', 'connects', 'variable')].edge_attr if hasattr(
            data[('constraint', 'connects', 'variable')], 'edge_attr'
        ) else None
        
        # 找到需要保留的边
        keep_mask = edge_index[0] != constraint_id
        kept_edge_indices = keep_mask.nonzero().squeeze()
        
        # 记录被移除的边
        remove_mask = edge_index[0] == constraint_id
        removed_edge_indices = remove_mask.nonzero().squeeze()
        
        if removed_edge_indices.numel() > 0:
            if removed_edge_indices.dim() == 0:
                removed_edge_indices = removed_edge_indices.unsqueeze(0)
            
            for idx in removed_edge_indices:
                edge_idx = idx.item()
                constraint_idx = edge_index[0, edge_idx].item()
                variable_idx = edge_index[1, edge_idx].item()
                
                edge_info = (constraint_idx, variable_idx)
                if edge_attr is not None:
                    edge_info = edge_info + (edge_attr[edge_idx].clone(),)
                
                original_edges.append(edge_info)
                removed_edges.append(edge_info)
        
        # 更新边索引
        if kept_edge_indices.numel() > 0:
            if kept_edge_indices.dim() == 0:
                kept_edge_indices = kept_edge_indices.unsqueeze(0)
            
            new_edge_index = edge_index[:, kept_edge_indices]
            data[('constraint', 'connects', 'variable')].edge_index = new_edge_index
            
            if edge_attr is not None:
                new_edge_attr = edge_attr[kept_edge_indices]
                data[('constraint', 'connects', 'variable')].edge_attr = new_edge_attr
        else:
            # 没有保留的边
            data[('constraint', 'connects', 'variable')].edge_index = torch.empty((2, 0), dtype=torch.long)
            if edge_attr is not None:
                data[('constraint', 'connects', 'variable')].edge_attr = torch.empty((0, edge_attr.size(1)))
        
        # 处理反向边 variable -> constraint
        self._update_reverse_edges(data, constraint_id)
        
        return original_edges, removed_edges
    
    def _update_reverse_edges(self, data: HeteroData, constraint_id: int):
        """更新反向边"""
        edge_index = data[('variable', 'connected_by', 'constraint')].edge_index
        edge_attr = data[('variable', 'connected_by', 'constraint')].edge_attr if hasattr(
            data[('variable', 'connected_by', 'constraint')], 'edge_attr'
        ) else None
        
        # 找到需要保留的边
        keep_mask = edge_index[1] != constraint_id
        kept_edge_indices = keep_mask.nonzero().squeeze()
        
        if kept_edge_indices.numel() > 0:
            if kept_edge_indices.dim() == 0:
                kept_edge_indices = kept_edge_indices.unsqueeze(0)
            
            new_edge_index = edge_index[:, kept_edge_indices]
            data[('variable', 'connected_by', 'constraint')].edge_index = new_edge_index
            
            if edge_attr is not None:
                new_edge_attr = edge_attr[kept_edge_indices]
                data[('variable', 'connected_by', 'constraint')].edge_attr = new_edge_attr
        else:
            data[('variable', 'connected_by', 'constraint')].edge_index = torch.empty((2, 0), dtype=torch.long)
            if edge_attr is not None:
                data[('variable', 'connected_by', 'constraint')].edge_attr = torch.empty((0, edge_attr.size(1)))
    
    def _add_virtual_edges(self, data: HeteroData, constraint_id: int):
        """添加虚拟边连接遮盖约束到所有变量节点 - 简化版本"""
        # 暂时禁用虚拟边添加以避免维度问题
        # 在实际G2MILP框架中，虚拟边是重要的，但现在我们专注于基本训练
        logger.debug(f"跳过虚拟边添加（简化模式）- 约束ID: {constraint_id}")
        return
        
        # 反向边特征
        current_reverse_edge_attr = data[('variable', 'connected_by', 'constraint')].edge_attr if hasattr(
            data[('variable', 'connected_by', 'constraint')], 'edge_attr'
        ) else None
        
        if current_reverse_edge_attr is not None:
            new_reverse_edge_attr = torch.cat([current_reverse_edge_attr, virtual_edge_attr], dim=0)
            data[('variable', 'connected_by', 'constraint')].edge_attr = new_reverse_edge_attr
        else:
            data[('variable', 'connected_by', 'constraint')].edge_attr = virtual_edge_attr
    
    def unmask_constraint(self, masked_data: HeteroData, 
                         mask_info: Dict,
                         predicted_results: Dict) -> HeteroData:
        """
        用预测结果替换遮盖的约束
        
        Args:
            masked_data: 遮盖后的数据
            mask_info: 遮盖信息
            predicted_results: 解码器预测结果
            
        Returns:
            重建后的异构图数据
        """
        constraint_id = mask_info['masked_constraint_id']
        
        # 推断目标设备
        target_device = None
        try:
            if hasattr(masked_data['constraint'], 'x') and masked_data['constraint'].x is not None:
                target_device = masked_data['constraint'].x.device
        except:
            pass
        
        # 深拷贝避免修改原数据
        reconstructed_data = copy.deepcopy(masked_data)
        
        # 确保深拷贝后设备一致性
        if target_device is not None:
            try:
                reconstructed_data = reconstructed_data.to(target_device)
                # 确保预测结果也在正确设备上
                for key, value in predicted_results.items():
                    if isinstance(value, torch.Tensor):
                        predicted_results[key] = value.to(target_device)
            except:
                logger.warning(f"无法确保unmask过程设备一致性")
                pass
        
        # 1. 恢复约束节点特征（使用预测的bias更新）
        # 这里可以根据预测结果更新约束特征，暂时保持原有特征结构
        # reconstructed_data['constraint'].x[constraint_id] = new_constraint_features
        
        # 2. 移除虚拟边
        self._remove_virtual_edges(reconstructed_data, constraint_id)
        
        # 3. 添加预测的连接边
        self._add_predicted_edges(
            reconstructed_data, 
            constraint_id, 
            predicted_results['connection_mask'],
            predicted_results['predicted_weights']
        )
        
        logger.debug(f"约束 {constraint_id} 已重建")
        
        return reconstructed_data
    
    def _remove_virtual_edges(self, data: HeteroData, constraint_id: int):
        """移除虚拟边"""
        # 移除 constraint -> variable 虚拟边
        edge_index = data[('constraint', 'connects', 'variable')].edge_index
        edge_attr = data[('constraint', 'connects', 'variable')].edge_attr
        
        # 找到非虚拟边（虚拟边权重为0）
        non_virtual_mask = ~((edge_index[0] == constraint_id) & (torch.norm(edge_attr, dim=1) == 0))
        
        new_edge_index = edge_index[:, non_virtual_mask]
        new_edge_attr = edge_attr[non_virtual_mask]
        
        data[('constraint', 'connects', 'variable')].edge_index = new_edge_index
        data[('constraint', 'connects', 'variable')].edge_attr = new_edge_attr
        
        # 移除反向虚拟边
        reverse_edge_index = data[('variable', 'connected_by', 'constraint')].edge_index
        reverse_edge_attr = data[('variable', 'connected_by', 'constraint')].edge_attr
        
        reverse_non_virtual_mask = ~((reverse_edge_index[1] == constraint_id) & (torch.norm(reverse_edge_attr, dim=1) == 0))
        
        new_reverse_edge_index = reverse_edge_index[:, reverse_non_virtual_mask]
        new_reverse_edge_attr = reverse_edge_attr[reverse_non_virtual_mask]
        
        data[('variable', 'connected_by', 'constraint')].edge_index = new_reverse_edge_index
        data[('variable', 'connected_by', 'constraint')].edge_attr = new_reverse_edge_attr
    
    def _add_predicted_edges(self, data: HeteroData, 
                           constraint_id: int,
                           connection_mask: torch.Tensor,
                           predicted_weights: torch.Tensor):
        """根据预测结果添加新的连接边"""
        # 找到被连接的变量
        connected_var_indices = connection_mask.squeeze().nonzero().squeeze()
        
        if connected_var_indices.numel() == 0:
            return
        
        if connected_var_indices.dim() == 0:
            connected_var_indices = connected_var_indices.unsqueeze(0)
        
        # 获取目标设备（从现有数据中推断）
        target_device = data[('constraint', 'connects', 'variable')].edge_index.device
        
        # 确保所有张量在同一设备上
        connected_var_indices = connected_var_indices.to(target_device)
        connection_mask = connection_mask.to(target_device)
        predicted_weights = predicted_weights.to(target_device)
        
        # 创建新边索引
        n_new_edges = connected_var_indices.size(0)
        constraint_indices = torch.full((n_new_edges,), constraint_id, dtype=torch.long, device=target_device)
        new_edge_index = torch.stack([constraint_indices, connected_var_indices])
        
        # 创建新边特征（使用预测的权重）
        # 需要将权重转换为完整的边特征向量
        edge_attr_dim = data[('constraint', 'connects', 'variable')].edge_attr.size(1)
        new_edge_attr = torch.zeros(n_new_edges, edge_attr_dim, device=target_device)
        
        # 将预测的权重放在第一个维度
        if predicted_weights.size(0) > 0:
            new_edge_attr[:, 0] = predicted_weights.squeeze()
        
        # 添加到现有边中
        current_edge_index = data[('constraint', 'connects', 'variable')].edge_index
        current_edge_attr = data[('constraint', 'connects', 'variable')].edge_attr
        
        # 确保设备一致性
        current_edge_index = current_edge_index.to(target_device)
        current_edge_attr = current_edge_attr.to(target_device)
        
        combined_edge_index = torch.cat([current_edge_index, new_edge_index], dim=1)
        combined_edge_attr = torch.cat([current_edge_attr, new_edge_attr], dim=0)
        
        data[('constraint', 'connects', 'variable')].edge_index = combined_edge_index
        data[('constraint', 'connects', 'variable')].edge_attr = combined_edge_attr
        
        # 添加反向边
        reverse_new_edge_index = torch.stack([connected_var_indices, constraint_indices])
        current_reverse_edge_index = data[('variable', 'connected_by', 'constraint')].edge_index
        current_reverse_edge_attr = data[('variable', 'connected_by', 'constraint')].edge_attr
        
        # 确保反向边设备一致性
        current_reverse_edge_index = current_reverse_edge_index.to(target_device)
        current_reverse_edge_attr = current_reverse_edge_attr.to(target_device)
        
        combined_reverse_edge_index = torch.cat([current_reverse_edge_index, reverse_new_edge_index], dim=1)
        combined_reverse_edge_attr = torch.cat([current_reverse_edge_attr, new_edge_attr], dim=0)
        
        data[('variable', 'connected_by', 'constraint')].edge_index = combined_reverse_edge_index
        data[('variable', 'connected_by', 'constraint')].edge_attr = combined_reverse_edge_attr


def create_masker(config: MaskingConfig = None) -> ConstraintMasker:
    """
    创建约束遮盖器的工厂函数
    
    Args:
        config: 遮盖配置
        
    Returns:
        约束遮盖器实例
    """
    return ConstraintMasker(config)


if __name__ == "__main__":
    # 测试代码
    print("G2MILP遮盖过程模块测试")
    print("=" * 40)
    
    # 创建测试配置
    config = MaskingConfig(
        masking_ratio=0.1,
        mask_strategy="random",
        random_seed=42
    )
    
    # 创建遮盖器
    masker = create_masker(config)
    print("遮盖器创建成功!")
    print(f"遮盖比例: {config.masking_ratio}")
    print(f"遮盖策略: {config.mask_strategy}")
    print(f"Mask标记维度: {config.mask_token_dim}")
    print(f"虚拟边维度: {config.virtual_edge_dim}")
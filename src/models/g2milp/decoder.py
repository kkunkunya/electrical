"""
G2MILP解码器模块
G2MILP Decoder Module

实现G2MILP的解码器部分，包含四个关键预测器：
1. Bias Predictor - 预测约束的偏置项(右端项)
2. Degree Predictor - 预测约束的度数
3. Logits Predictor - 预测变量与约束的连接概率
4. Weights Predictor - 预测边的权重(系数)

根据论文3.3节的解码器公式实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GraphConv, HeteroConv
from torch_geometric.data import HeteroData
import numpy as np
import logging
from typing import Dict, Tuple, Optional, Any, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass 
class DecoderConfig:
    """解码器配置"""
    # 基础网络参数
    hidden_dim: int = 128
    latent_dim: int = 64
    num_layers: int = 2
    dropout: float = 0.1
    
    # GNN参数  
    gnn_type: str = "GraphConv"  # GraphConv支持异构图且稳定
    use_edge_features: bool = True
    
    # 预测器参数
    predictor_hidden_dim: int = 64
    use_batch_norm: bool = True
    activation: str = "relu"
    
    # 数据范围参数（用于归一化）
    bias_min: float = 0.0
    bias_max: float = 1000.0
    degree_min: int = 1
    degree_max: int = 50
    coeff_min: float = -100.0
    coeff_max: float = 100.0


class BiasPredictor(nn.Module):
    """
    偏置预测器
    
    根据论文3.3节实现，预测约束的右端项值b_v
    使用归一化技巧：b*_v = (b_v - b_min) / (b_max - b_min)
    """
    
    def __init__(self, input_dim: int, config: DecoderConfig):
        super().__init__()
        self.config = config
        
        # MLP网络
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, config.predictor_hidden_dim),
            nn.LayerNorm(config.predictor_hidden_dim) if config.use_batch_norm else nn.Identity(),
            self._get_activation(),
            nn.Dropout(config.dropout),
            
            nn.Linear(config.predictor_hidden_dim, config.predictor_hidden_dim // 2),
            nn.LayerNorm(config.predictor_hidden_dim // 2) if config.use_batch_norm else nn.Identity(),
            self._get_activation(),
            nn.Dropout(config.dropout),
            
            nn.Linear(config.predictor_hidden_dim // 2, 1),
            nn.Sigmoid()  # 限制输出到[0,1]
        )
        
        self.bias_range = config.bias_max - config.bias_min
        
    def _get_activation(self):
        if self.config.activation == "relu":
            return nn.ReLU()
        elif self.config.activation == "leaky_relu":
            return nn.LeakyReLU()
        elif self.config.activation == "elu":
            return nn.ELU()
        else:
            return nn.ReLU()
    
    def forward(self, constraint_features: torch.Tensor, 
                constraint_latents: torch.Tensor) -> torch.Tensor:
        """
        预测归一化的偏置值
        
        Args:
            constraint_features: 约束节点特征 [n_constraints, feature_dim]
            constraint_latents: 约束节点潜变量 [n_constraints, latent_dim]
            
        Returns:
            归一化的偏置值 [n_constraints, 1]
        """
        # 拼接特征和潜变量
        # 调试：检查设备一致性
        if constraint_features.device != constraint_latents.device:
            print(f"警告：BiasPredictor设备不匹配! features: {constraint_features.device}, latents: {constraint_latents.device}")
            constraint_latents = constraint_latents.to(constraint_features.device)
        
        input_features = torch.cat([constraint_features, constraint_latents], dim=1)
        
        # MLP预测
        normalized_bias = self.mlp(input_features)
        
        return normalized_bias
    
    def denormalize_bias(self, normalized_bias: torch.Tensor) -> torch.Tensor:
        """将归一化的偏置值转换回原始范围"""
        return self.config.bias_min + normalized_bias * self.bias_range


class DegreePredictor(nn.Module):
    """
    度数预测器
    
    预测约束节点的度数（连接的变量数量）
    使用归一化：d*_v = (d_v - d_min) / (d_max - d_min)
    """
    
    def __init__(self, input_dim: int, config: DecoderConfig):
        super().__init__()
        self.config = config
        
        # MLP网络
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, config.predictor_hidden_dim),
            nn.LayerNorm(config.predictor_hidden_dim) if config.use_batch_norm else nn.Identity(),
            self._get_activation(),
            nn.Dropout(config.dropout),
            
            nn.Linear(config.predictor_hidden_dim, config.predictor_hidden_dim // 2),
            nn.LayerNorm(config.predictor_hidden_dim // 2) if config.use_batch_norm else nn.Identity(),
            self._get_activation(),
            nn.Dropout(config.dropout),
            
            nn.Linear(config.predictor_hidden_dim // 2, 1),
            nn.Sigmoid()  # 限制输出到[0,1]
        )
        
        self.degree_range = config.degree_max - config.degree_min
        
    def _get_activation(self):
        if self.config.activation == "relu":
            return nn.ReLU()
        elif self.config.activation == "leaky_relu":
            return nn.LeakyReLU()
        elif self.config.activation == "elu":
            return nn.ELU()
        else:
            return nn.ReLU()
    
    def forward(self, constraint_features: torch.Tensor,
                constraint_latents: torch.Tensor) -> torch.Tensor:
        """
        预测归一化的度数值
        
        Args:
            constraint_features: 约束节点特征
            constraint_latents: 约束节点潜变量
            
        Returns:
            归一化的度数值 [n_constraints, 1]
        """
        # 拼接特征和潜变量
        # 调试：检查设备一致性
        if constraint_features.device != constraint_latents.device:
            print(f"警告：BiasPredictor设备不匹配! features: {constraint_features.device}, latents: {constraint_latents.device}")
            constraint_latents = constraint_latents.to(constraint_features.device)
        
        input_features = torch.cat([constraint_features, constraint_latents], dim=1)
        
        # MLP预测
        normalized_degree = self.mlp(input_features)
        
        return normalized_degree
    
    def denormalize_degree(self, normalized_degree: torch.Tensor) -> torch.Tensor:
        """将归一化度数转换为整数度数"""
        real_degree = self.config.degree_min + normalized_degree * self.degree_range
        return torch.round(real_degree).long()


class LogitsPredictor(nn.Module):
    """
    连接概率预测器
    
    预测变量节点与被遮盖约束节点的连接概率
    输出logits，用于判断是否连接
    """
    
    def __init__(self, input_dim: int, config: DecoderConfig):
        super().__init__()
        self.config = config
        
        # MLP网络
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, config.predictor_hidden_dim),
            nn.LayerNorm(config.predictor_hidden_dim) if config.use_batch_norm else nn.Identity(),
            self._get_activation(),
            nn.Dropout(config.dropout),
            
            nn.Linear(config.predictor_hidden_dim, config.predictor_hidden_dim // 2),
            nn.LayerNorm(config.predictor_hidden_dim // 2) if config.use_batch_norm else nn.Identity(),
            self._get_activation(),
            nn.Dropout(config.dropout),
            
            nn.Linear(config.predictor_hidden_dim // 2, 1),
            nn.Sigmoid()  # 输出连接概率
        )
        
    def _get_activation(self):
        if self.config.activation == "relu":
            return nn.ReLU()
        elif self.config.activation == "leaky_relu":
            return nn.LeakyReLU()
        elif self.config.activation == "elu":
            return nn.ELU()
        else:
            return nn.ReLU()
    
    def forward(self, variable_features: torch.Tensor,
                variable_latents: torch.Tensor) -> torch.Tensor:
        """
        预测变量与约束的连接概率
        
        Args:
            variable_features: 变量节点特征 [n_variables, feature_dim]
            variable_latents: 变量节点潜变量 [n_variables, latent_dim]
            
        Returns:
            连接概率 [n_variables, 1]
        """
        # 拼接特征和潜变量
        # 调试：检查设备一致性
        if variable_features.device != variable_latents.device:
            print(f"警告：LogitsPredictor设备不匹配! features: {variable_features.device}, latents: {variable_latents.device}")
            variable_latents = variable_latents.to(variable_features.device)
        
        input_features = torch.cat([variable_features, variable_latents], dim=1)
        
        # MLP预测
        connection_probs = self.mlp(input_features)
        
        return connection_probs
    
    def select_top_k_connections(self, logits: torch.Tensor, k: int) -> torch.Tensor:
        """
        根据度数预测结果选择top-k连接
        
        Args:
            logits: 连接概率 [n_variables, 1]
            k: 要选择的连接数量
            
        Returns:
            二值连接向量 [n_variables, 1]
        """
        # 安全检查：确保k值合理
        n_variables = logits.size(0)
        k = max(1, min(int(k), n_variables))  # 确保k在合理范围内
        
        # 处理空tensor的情况
        if n_variables == 0:
            return torch.zeros_like(logits)
        
        # 获取top-k索引，使用安全的方式
        logits_flat = logits.squeeze(-1) if logits.dim() > 1 else logits
        
        # 防止k大于可用元素数量
        actual_k = min(k, logits_flat.numel())
        
        if actual_k <= 0:
            return torch.zeros_like(logits)
        
        # 使用torch.topk，添加错误处理
        try:
            _, top_k_indices = torch.topk(logits_flat, actual_k, largest=True)
        except RuntimeError as e:
            logger.warning(f"torch.topk失败: {e}, 使用全部连接")
            # 如果topk失败，返回前k个索引
            top_k_indices = torch.arange(min(actual_k, logits_flat.size(0)), 
                                       device=logits.device, dtype=torch.long)
        
        # 创建二值连接向量
        connections = torch.zeros_like(logits)
        if top_k_indices.numel() > 0:
            # 确保索引维度正确
            if logits.dim() > 1:
                connections[top_k_indices, :] = 1.0
            else:
                connections[top_k_indices] = 1.0
        
        return connections


class WeightsPredictor(nn.Module):
    """
    权重预测器
    
    预测被选中连接的边权重（约束系数）
    只对连接的变量-约束对进行权重预测
    """
    
    def __init__(self, input_dim: int, config: DecoderConfig):
        super().__init__()
        self.config = config
        
        # MLP网络
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, config.predictor_hidden_dim),
            nn.LayerNorm(config.predictor_hidden_dim) if config.use_batch_norm else nn.Identity(),
            self._get_activation(),
            nn.Dropout(config.dropout),
            
            nn.Linear(config.predictor_hidden_dim, config.predictor_hidden_dim // 2),
            nn.LayerNorm(config.predictor_hidden_dim // 2) if config.use_batch_norm else nn.Identity(),
            self._get_activation(),
            nn.Dropout(config.dropout),
            
            nn.Linear(config.predictor_hidden_dim // 2, 1),
            nn.Sigmoid()  # 输出归一化权重
        )
        
        self.coeff_range = config.coeff_max - config.coeff_min
        
    def _get_activation(self):
        if self.config.activation == "relu":
            return nn.ReLU()
        elif self.config.activation == "leaky_relu":
            return nn.LeakyReLU()
        elif self.config.activation == "elu":
            return nn.ELU()
        else:
            return nn.ReLU()
    
    def forward(self, variable_features: torch.Tensor,
                variable_latents: torch.Tensor,
                connection_mask: torch.Tensor) -> torch.Tensor:
        """
        预测连接边的权重
        
        Args:
            variable_features: 变量节点特征
            variable_latents: 变量节点潜变量
            connection_mask: 连接掩码 [n_variables, 1]
            
        Returns:
            归一化权重 [n_connected_variables, 1]
        """
        # 只对连接的变量进行权重预测
        connected_indices = connection_mask.squeeze().nonzero().squeeze()
        
        if connected_indices.numel() == 0:
            return torch.empty(0, 1, device=variable_features.device)
        
        # 如果只有一个连接，需要处理维度
        if connected_indices.dim() == 0:
            connected_indices = connected_indices.unsqueeze(0)
        
        # 获取连接变量的特征
        connected_var_features = variable_features[connected_indices]
        connected_var_latents = variable_latents[connected_indices]
        
        # 拼接特征和潜变量
        # 调试：检查设备一致性
        if connected_var_features.device != connected_var_latents.device:
            print(f"警告：WeightsPredictor设备不匹配! features: {connected_var_features.device}, latents: {connected_var_latents.device}")
            connected_var_latents = connected_var_latents.to(connected_var_features.device)
        
        input_features = torch.cat([connected_var_features, connected_var_latents], dim=1)
        
        # MLP预测
        normalized_weights = self.mlp(input_features)
        
        return normalized_weights
    
    def denormalize_weights(self, normalized_weights: torch.Tensor) -> torch.Tensor:
        """将归一化权重转换回原始范围"""
        return self.config.coeff_min + normalized_weights * self.coeff_range


class G2MILPDecoder(nn.Module):
    """
    G2MILP解码器主类
    
    集成四个预测器，实现约束重建功能
    """
    
    def __init__(self,
                 constraint_feature_dim: int = 16,
                 variable_feature_dim: int = 9,
                 config: DecoderConfig = None):
        super().__init__()
        
        self.config = config or DecoderConfig()
        self.constraint_feature_dim = constraint_feature_dim
        self.variable_feature_dim = variable_feature_dim
        
        # GNN用于处理遮盖后的图
        self.gnn = self._build_gnn()
        
        # 四个预测器
        total_constraint_dim = self.config.hidden_dim + self.config.latent_dim
        total_variable_dim = self.config.hidden_dim + self.config.latent_dim
        
        self.bias_predictor = BiasPredictor(total_constraint_dim, self.config)
        self.degree_predictor = DegreePredictor(total_constraint_dim, self.config)
        self.logits_predictor = LogitsPredictor(total_variable_dim, self.config)
        self.weights_predictor = WeightsPredictor(total_variable_dim, self.config)
        
        # 输入投影层
        self.constraint_proj = nn.Linear(constraint_feature_dim, self.config.hidden_dim)
        self.variable_proj = nn.Linear(variable_feature_dim, self.config.hidden_dim)
        
        logger.info(f"G2MILP解码器初始化完成 - 隐藏维度: {self.config.hidden_dim}")
    
    def _build_gnn(self):
        """构建GNN网络"""
        layers = nn.ModuleList()
        
        for i in range(self.config.num_layers):
            if self.config.gnn_type == "GCN":
                conv = HeteroConv({
                    ('constraint', 'connects', 'variable'): GCNConv(
                        self.config.hidden_dim, self.config.hidden_dim,
                        add_self_loops=False  # 异构图不支持自环
                    ),
                    ('variable', 'connected_by', 'constraint'): GCNConv(
                        self.config.hidden_dim, self.config.hidden_dim,
                        add_self_loops=False  # 异构图不支持自环
                    )
                }, aggr='sum')
            elif self.config.gnn_type == "GraphConv":
                conv = HeteroConv({
                    ('constraint', 'connects', 'variable'): GraphConv(
                        self.config.hidden_dim, self.config.hidden_dim,
                        aggr='add'
                    ),
                    ('variable', 'connected_by', 'constraint'): GraphConv(
                        self.config.hidden_dim, self.config.hidden_dim,
                        aggr='add'
                    )
                }, aggr='sum')
            elif self.config.gnn_type == "GAT":
                conv = HeteroConv({
                    ('constraint', 'connects', 'variable'): GATConv(
                        self.config.hidden_dim, self.config.hidden_dim // 4, heads=4
                    ),
                    ('variable', 'connected_by', 'constraint'): GATConv(
                        self.config.hidden_dim, self.config.hidden_dim // 4, heads=4
                    )
                }, aggr='sum')
            elif self.config.gnn_type == "SAGE":
                conv = HeteroConv({
                    ('constraint', 'connects', 'variable'): SAGEConv(
                        self.config.hidden_dim, self.config.hidden_dim,
                        aggr='mean'
                    ),
                    ('variable', 'connected_by', 'constraint'): SAGEConv(
                        self.config.hidden_dim, self.config.hidden_dim,
                        aggr='mean'
                    )
                }, aggr='sum')
            else:
                raise ValueError(f"不支持的GNN类型: {self.config.gnn_type}")
            
            layers.append(conv)
        
        return layers
    
    def forward(self, masked_data: HeteroData,
                latent_dict: Dict[str, torch.Tensor],
                masked_constraint_id: int) -> Dict[str, torch.Tensor]:
        """
        解码器前向传播
        
        Args:
            masked_data: 遮盖后的异构图数据
            latent_dict: 潜变量字典
            masked_constraint_id: 被遮盖的约束ID
            
        Returns:
            预测结果字典
        """
        # 1. 输入特征变换
        x_dict = {
            'constraint': self.constraint_proj(masked_data['constraint'].x),
            'variable': self.variable_proj(masked_data['variable'].x)
        }
        
        # 2. GNN特征提取
        edge_index_dict = {
            ('constraint', 'connects', 'variable'): masked_data[('constraint', 'connects', 'variable')].edge_index,
            ('variable', 'connected_by', 'constraint'): masked_data[('variable', 'connected_by', 'constraint')].edge_index
        }
        
        # GraphConv不支持多维边特征，不使用边特征
        for gnn_layer in self.gnn:
            x_dict = gnn_layer(x_dict, edge_index_dict)
            for node_type in x_dict:
                x_dict[node_type] = F.relu(x_dict[node_type])
                x_dict[node_type] = F.dropout(x_dict[node_type], p=self.config.dropout, training=self.training)
        
        # 3. 获取被遮盖约束的特征和潜变量
        masked_constraint_features = x_dict['constraint'][masked_constraint_id:masked_constraint_id+1]
        masked_constraint_latent = latent_dict['constraint'][masked_constraint_id:masked_constraint_id+1]
        
        # 4. 四个预测器依次预测
        # Bias预测
        predicted_bias = self.bias_predictor(masked_constraint_features, masked_constraint_latent)
        
        # Degree预测
        predicted_degree_norm = self.degree_predictor(masked_constraint_features, masked_constraint_latent)
        predicted_degree = self.degree_predictor.denormalize_degree(predicted_degree_norm)
        
        # Logits预测（所有变量的连接概率）
        all_variable_features = x_dict['variable']
        all_variable_latents = latent_dict['variable']
        connection_logits = self.logits_predictor(all_variable_features, all_variable_latents)
        
        # 选择top-k连接，添加智能约束处理
        try:
            predicted_k = max(1, int(predicted_degree.item()))  # 确保k至少为1
            n_variables = connection_logits.size(0)
            
            # 智能度数约束：防止预测过多连接
            max_reasonable_k = min(n_variables // 4, 50)  # 最多连接1/4的变量，且不超过50
            min_reasonable_k = max(1, min(3, n_variables))  # 至少连接1-3个变量
            
            # 应用约束
            constrained_k = max(min_reasonable_k, min(predicted_k, max_reasonable_k))
            
            if constrained_k != predicted_k:
                logger.debug(f"度数约束: 预测{predicted_k} -> 应用{constrained_k} (变量数{n_variables})")
            
            connection_mask = self.logits_predictor.select_top_k_connections(connection_logits, constrained_k)
            
        except Exception as e:
            logger.warning(f"top-k连接选择失败: {e}, 使用保守连接数")
            # 如果失败，使用保守的连接数
            n_variables = connection_logits.size(0)
            conservative_k = max(1, min(3, n_variables))  # 保守策略：最多3个连接
            connection_mask = self.logits_predictor.select_top_k_connections(connection_logits, conservative_k)
        
        # Weights预测
        predicted_weights = self.weights_predictor(
            all_variable_features, all_variable_latents, connection_mask
        )
        
        return {
            'predicted_bias': predicted_bias,
            'predicted_degree': predicted_degree,
            'connection_logits': connection_logits,
            'connection_mask': connection_mask,
            'predicted_weights': predicted_weights,
            'bias_normalized': predicted_bias,
            'degree_normalized': predicted_degree_norm
        }


def create_decoder(constraint_feature_dim: int = 16,
                  variable_feature_dim: int = 9,
                  config: DecoderConfig = None) -> G2MILPDecoder:
    """
    创建G2MILP解码器的工厂函数
    
    Args:
        constraint_feature_dim: 约束节点特征维度
        variable_feature_dim: 变量节点特征维度
        config: 解码器配置
        
    Returns:
        G2MILP解码器实例
    """
    if config is None:
        config = DecoderConfig()
    
    return G2MILPDecoder(
        constraint_feature_dim=constraint_feature_dim,
        variable_feature_dim=variable_feature_dim,
        config=config
    )


if __name__ == "__main__":
    # 测试代码
    print("G2MILP解码器模块测试")
    print("=" * 40)
    
    # 创建测试配置
    config = DecoderConfig(
        hidden_dim=64,
        latent_dim=32,
        num_layers=2
    )
    
    # 创建解码器
    decoder = create_decoder(config=config)
    print(f"解码器参数数量: {sum(p.numel() for p in decoder.parameters())}")
    print("解码器创建成功!")
    
    # 测试各个预测器
    print("\n预测器测试:")
    print(f"- Bias预测器参数: {sum(p.numel() for p in decoder.bias_predictor.parameters())}")
    print(f"- Degree预测器参数: {sum(p.numel() for p in decoder.degree_predictor.parameters())}")
    print(f"- Logits预测器参数: {sum(p.numel() for p in decoder.logits_predictor.parameters())}")
    print(f"- Weights预测器参数: {sum(p.numel() for p in decoder.weights_predictor.parameters())}")
"""
G2MILP编码器模块
G2MILP Encoder Module

实现G2MILP的编码器部分，基于图神经网络(GNN)将二分图编码为潜向量空间。
根据论文3.3节的Encoder公式：q_φ(Z|G) = Π q_φ(z_u|G)，其中z_u～N(μ_φ(h_u^G), exp(Σ_φ(h_u^G)))
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GraphConv, HeteroConv
from torch_geometric.data import HeteroData
import numpy as np
import logging
from typing import Dict, Tuple, Optional, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class EncoderConfig:
    """编码器配置"""
    # 网络架构参数
    gnn_type: str = "GraphConv"  # GCN, GAT, SAGE, GraphConv (GraphConv支持异构图)
    hidden_dim: int = 128
    latent_dim: int = 64
    num_layers: int = 3
    dropout: float = 0.1
    
    # GCN特定参数
    gcn_improved: bool = False
    gcn_cached: bool = True
    gcn_add_self_loops: bool = True
    
    # GAT特定参数
    gat_heads: int = 4
    gat_concat: bool = True
    gat_negative_slope: float = 0.2
    
    # SAGE特定参数
    sage_aggr: str = "mean"  # mean, max, sum
    
    # 训练参数
    use_batch_norm: bool = True
    activation: str = "relu"  # relu, leaky_relu, elu


class BipartiteGNNLayer(nn.Module):
    """
    二分图GNN层
    
    专为约束-变量二分图设计的异构图神经网络层
    """
    
    def __init__(self, 
                 constraint_in_dim: int,
                 variable_in_dim: int, 
                 out_dim: int,
                 gnn_type: str = "GCN",
                 config: EncoderConfig = None):
        super().__init__()
        
        self.config = config or EncoderConfig()
        self.gnn_type = gnn_type
        
        # 异构图卷积层
        if gnn_type == "GCN":
            self.conv = HeteroConv({
                ('constraint', 'connects', 'variable'): GCNConv(
                    constraint_in_dim, out_dim, 
                    improved=self.config.gcn_improved,
                    cached=self.config.gcn_cached,
                    add_self_loops=False  # 异构图不支持自环
                ),
                ('variable', 'connected_by', 'constraint'): GCNConv(
                    variable_in_dim, out_dim,
                    improved=self.config.gcn_improved,
                    cached=self.config.gcn_cached,
                    add_self_loops=False  # 异构图不支持自环
                )
            }, aggr='sum')
            
        elif gnn_type == "GraphConv":
            self.conv = HeteroConv({
                ('constraint', 'connects', 'variable'): GraphConv(
                    constraint_in_dim, out_dim,
                    aggr='add'
                ),
                ('variable', 'connected_by', 'constraint'): GraphConv(
                    variable_in_dim, out_dim,
                    aggr='add'
                )
            }, aggr='sum')
            
        elif gnn_type == "GAT":
            self.conv = HeteroConv({
                ('constraint', 'connects', 'variable'): GATConv(
                    constraint_in_dim, out_dim // self.config.gat_heads,
                    heads=self.config.gat_heads,
                    concat=self.config.gat_concat,
                    negative_slope=self.config.gat_negative_slope,
                    dropout=self.config.dropout
                ),
                ('variable', 'connected_by', 'constraint'): GATConv(
                    variable_in_dim, out_dim // self.config.gat_heads,
                    heads=self.config.gat_heads,
                    concat=self.config.gat_concat,
                    negative_slope=self.config.gat_negative_slope,
                    dropout=self.config.dropout
                )
            }, aggr='sum')
            
        elif gnn_type == "SAGE":
            self.conv = HeteroConv({
                ('constraint', 'connects', 'variable'): SAGEConv(
                    constraint_in_dim, out_dim,
                    aggr=self.config.sage_aggr
                ),
                ('variable', 'connected_by', 'constraint'): SAGEConv(
                    variable_in_dim, out_dim,
                    aggr=self.config.sage_aggr
                )
            }, aggr='sum')
        else:
            raise ValueError(f"不支持的GNN类型: {gnn_type}")
        
        # 批归一化 - 针对单实例训练禁用
        if self.config.use_batch_norm:
            # 使用LayerNorm替代BatchNorm以避免batch_size=1的问题
            self.batch_norms = nn.ModuleDict({
                'constraint': nn.LayerNorm(out_dim),
                'variable': nn.LayerNorm(out_dim)
            })
        
        # 激活函数
        if self.config.activation == "relu":
            self.activation = F.relu
        elif self.config.activation == "leaky_relu":
            self.activation = F.leaky_relu
        elif self.config.activation == "elu":
            self.activation = F.elu
        else:
            self.activation = F.relu
    
    def forward(self, x_dict: Dict[str, torch.Tensor], 
                edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
                edge_attr_dict: Optional[Dict[Tuple[str, str, str], torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            x_dict: 节点特征字典 {'constraint': tensor, 'variable': tensor}
            edge_index_dict: 边索引字典
            edge_attr_dict: 边特征字典（可选）
            
        Returns:
            更新后的节点特征字典
        """
        # 图卷积
        if edge_attr_dict is not None:
            x_dict = self.conv(x_dict, edge_index_dict, edge_attr_dict)
        else:
            x_dict = self.conv(x_dict, edge_index_dict)
        
        # 批归一化
        if self.config.use_batch_norm and hasattr(self, 'batch_norms'):
            for node_type in x_dict:
                if node_type in self.batch_norms:
                    x_dict[node_type] = self.batch_norms[node_type](x_dict[node_type])
        
        # 激活函数
        for node_type in x_dict:
            x_dict[node_type] = self.activation(x_dict[node_type])
        
        return x_dict


class G2MILPEncoder(nn.Module):
    """
    G2MILP编码器
    
    基于图神经网络的变分编码器，将二分图编码为潜向量分布
    实现论文公式：q_φ(Z|G) = Π q_φ(z_u|G)，z_u ~ N(μ_φ(h_u^G), exp(Σ_φ(h_u^G)))
    """
    
    def __init__(self, 
                 constraint_feature_dim: int = 16,
                 variable_feature_dim: int = 9,
                 edge_feature_dim: int = 8,
                 config: EncoderConfig = None):
        super().__init__()
        
        self.config = config or EncoderConfig()
        self.constraint_feature_dim = constraint_feature_dim
        self.variable_feature_dim = variable_feature_dim
        self.edge_feature_dim = edge_feature_dim
        
        # 输入特征变换
        self.constraint_input_proj = nn.Linear(constraint_feature_dim, self.config.hidden_dim)
        self.variable_input_proj = nn.Linear(variable_feature_dim, self.config.hidden_dim)
        
        # 边特征变换（如果使用）
        if edge_feature_dim > 0:
            self.edge_proj = nn.Linear(edge_feature_dim, self.config.hidden_dim)
        
        # GNN层
        self.gnn_layers = nn.ModuleList()
        for i in range(self.config.num_layers):
            if i == 0:
                # 第一层
                layer = BipartiteGNNLayer(
                    constraint_in_dim=self.config.hidden_dim,
                    variable_in_dim=self.config.hidden_dim,
                    out_dim=self.config.hidden_dim,
                    gnn_type=self.config.gnn_type,
                    config=self.config
                )
            else:
                # 后续层
                layer = BipartiteGNNLayer(
                    constraint_in_dim=self.config.hidden_dim,
                    variable_in_dim=self.config.hidden_dim,
                    out_dim=self.config.hidden_dim,
                    gnn_type=self.config.gnn_type,
                    config=self.config
                )
            self.gnn_layers.append(layer)
        
        # 潜变量分布参数网络
        self.mu_networks = nn.ModuleDict({
            'constraint': nn.Linear(self.config.hidden_dim, self.config.latent_dim),
            'variable': nn.Linear(self.config.hidden_dim, self.config.latent_dim)
        })
        
        self.logvar_networks = nn.ModuleDict({
            'constraint': nn.Linear(self.config.hidden_dim, self.config.latent_dim),
            'variable': nn.Linear(self.config.hidden_dim, self.config.latent_dim)
        })
        
        # Dropout
        self.dropout = nn.Dropout(self.config.dropout)
        
        logger.info(f"G2MILP编码器初始化完成 - 潜变量维度: {self.config.latent_dim}, "
                   f"隐藏维度: {self.config.hidden_dim}, 层数: {self.config.num_layers}")
    
    def forward(self, data: HeteroData) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        编码器前向传播
        
        Args:
            data: 异构图数据
            
        Returns:
            (node_representations, mu_dict, logvar_dict)
        """
        # 1. 输入特征变换
        x_dict = {
            'constraint': self.constraint_input_proj(data['constraint'].x),
            'variable': self.variable_input_proj(data['variable'].x)
        }
        
        # 2. 提取边信息
        edge_index_dict = {
            ('constraint', 'connects', 'variable'): data[('constraint', 'connects', 'variable')].edge_index,
            ('variable', 'connected_by', 'constraint'): data[('variable', 'connected_by', 'constraint')].edge_index
        }
        
        # GraphConv不支持多维边特征，暂时不使用边特征
        edge_attr_dict = None
        
        # 3. GNN特征提取
        for layer in self.gnn_layers:
            x_dict = layer(x_dict, edge_index_dict, edge_attr_dict)
            
            # Dropout
            for node_type in x_dict:
                x_dict[node_type] = self.dropout(x_dict[node_type])
        
        # 4. 计算潜变量分布参数
        mu_dict = {}
        logvar_dict = {}
        
        for node_type in ['constraint', 'variable']:
            if node_type in x_dict:
                mu_dict[node_type] = self.mu_networks[node_type](x_dict[node_type])
                logvar_dict[node_type] = self.logvar_networks[node_type](x_dict[node_type])
        
        return x_dict, mu_dict, logvar_dict
    
    def sample_latent_variables(self, mu_dict: Dict[str, torch.Tensor], 
                              logvar_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        从潜变量分布中采样（数值稳定版）
        
        实现重参数化技巧：z = μ + σ * ε，其中ε ~ N(0,I)
        添加数值稳定性检查，防止梯度爆炸
        """
        z_dict = {}
        
        for node_type in mu_dict:
            mu = mu_dict[node_type]
            logvar = logvar_dict[node_type]
            
            # 数值稳定性检查和裁剪
            mu = torch.clamp(mu, min=-10.0, max=10.0)
            logvar = torch.clamp(logvar, min=-10.0, max=10.0)  # 防止exp(logvar)过大
            
            # 检查是否有NaN或无限值
            if not torch.isfinite(mu).all():
                logger.warning(f"检测到{node_type}节点的mu中有非有限值，使用零值替换")
                mu = torch.zeros_like(mu)
            if not torch.isfinite(logvar).all():
                logger.warning(f"检测到{node_type}节点的logvar中有非有限值，使用零值替换")
                logvar = torch.zeros_like(logvar)
            
            # 重参数化采样（数值稳定版）
            std = torch.exp(0.5 * logvar)
            # 进一步裁剪std，防止过大
            std = torch.clamp(std, min=1e-6, max=10.0)
            
            eps = torch.randn_like(std)
            z = mu + eps * std
            
            # 最终输出裁剪
            z_dict[node_type] = torch.clamp(z, min=-20.0, max=20.0)
        
        return z_dict
    
    def encode(self, data: HeteroData, 
               sample: bool = True) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        完整编码过程
        
        Args:
            data: 输入的异构图数据
            sample: 是否从潜变量分布中采样
            
        Returns:
            (z_dict, mu_dict, logvar_dict)
        """
        # 前向传播
        node_representations, mu_dict, logvar_dict = self.forward(data)
        
        # 采样潜变量
        if sample:
            z_dict = self.sample_latent_variables(mu_dict, logvar_dict)
        else:
            # 使用均值作为潜变量
            z_dict = mu_dict
        
        return z_dict, mu_dict, logvar_dict
    
    def compute_kl_divergence(self, mu_dict: Dict[str, torch.Tensor], 
                            logvar_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        计算KL散度损失（数值稳定版）
        
        KL[q(z|x) || p(z)] = 0.5 * Σ(1 + log(σ²) - μ² - σ²)
        添加数值稳定性保护，防止NaN和无限值
        """
        kl_loss = 0.0
        device = next(iter(mu_dict.values())).device
        
        for node_type in mu_dict:
            mu = mu_dict[node_type]
            logvar = logvar_dict[node_type]
            
            # 数值稳定性预处理
            mu = torch.clamp(mu, min=-10.0, max=10.0)
            logvar = torch.clamp(logvar, min=-10.0, max=10.0)
            
            # 检查输入有效性
            if not torch.isfinite(mu).all() or not torch.isfinite(logvar).all():
                logger.warning(f"KL散度计算中检测到{node_type}节点的非有限值，跳过该节点")
                continue
            
            try:
                # KL散度公式（数值稳定版）
                mu_sq = mu.pow(2)
                logvar_exp = torch.exp(logvar)
                
                # 进一步防护
                mu_sq = torch.clamp(mu_sq, max=100.0)
                logvar_exp = torch.clamp(logvar_exp, max=100.0)
                
                kl = -0.5 * torch.sum(1 + logvar - mu_sq - logvar_exp, dim=1)
                
                # 检查KL散度的有效性
                if torch.isfinite(kl).all():
                    kl_mean = torch.mean(kl)
                    # 裁剪KL散度，防止过大
                    kl_mean = torch.clamp(kl_mean, min=0.0, max=50.0)
                    kl_loss += kl_mean
                else:
                    logger.warning(f"KL散度计算产生非有限值，使用默认值替代")
                    kl_loss += torch.tensor(0.1, device=device)
                    
            except Exception as e:
                logger.warning(f"KL散度计算异常: {e}，使用默认值")
                kl_loss += torch.tensor(0.1, device=device)
        
        # 最终结果稳定性检查
        if isinstance(kl_loss, (int, float)):
            kl_loss = torch.tensor(float(kl_loss), device=device)
        
        return torch.clamp(kl_loss, min=0.0, max=100.0)


def create_encoder(constraint_feature_dim: int = 16,
                  variable_feature_dim: int = 9,
                  edge_feature_dim: int = 8,
                  config: EncoderConfig = None) -> G2MILPEncoder:
    """
    创建G2MILP编码器的工厂函数
    
    Args:
        constraint_feature_dim: 约束节点特征维度
        variable_feature_dim: 变量节点特征维度  
        edge_feature_dim: 边特征维度
        config: 编码器配置
        
    Returns:
        G2MILP编码器实例
    """
    if config is None:
        config = EncoderConfig()
    
    return G2MILPEncoder(
        constraint_feature_dim=constraint_feature_dim,
        variable_feature_dim=variable_feature_dim,
        edge_feature_dim=edge_feature_dim,
        config=config
    )


if __name__ == "__main__":
    # 测试代码
    print("G2MILP编码器模块测试")
    print("=" * 40)
    
    # 创建测试配置
    config = EncoderConfig(
        gnn_type="GCN",
        hidden_dim=64,
        latent_dim=32,
        num_layers=2
    )
    
    # 创建编码器
    encoder = create_encoder(config=config)
    print(f"编码器参数数量: {sum(p.numel() for p in encoder.parameters())}")
    print("编码器创建成功!")
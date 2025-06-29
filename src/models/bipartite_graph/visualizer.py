"""
二分图可视化工具
提供二分图的多种可视化方案

可视化功能:
1. NetworkX布局可视化
2. 特征向量热图
3. 度数分布图  
4. 系数分布分析
5. 图结构统计图表
6. 交互式可视化（如果支持）
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import pandas as pd

from .data_structures import BipartiteGraph

logger = logging.getLogger(__name__)

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    logger.warning("NetworkX不可用，图布局可视化受限")

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    logger.warning("Plotly不可用，交互式可视化不支持")


class BipartiteGraphVisualizer:
    """
    二分图可视化工具
    支持多种可视化方案和自定义配置
    """
    
    def __init__(self, 
                 figure_size: Tuple[int, int] = (12, 8),
                 dpi: int = 300,
                 style: str = 'seaborn-v0_8',
                 color_palette: str = 'Set2'):
        """
        初始化可视化器
        
        Args:
            figure_size: 图形大小
            dpi: 图像分辨率
            style: matplotlib样式
            color_palette: 颜色方案
        """
        self.figure_size = figure_size
        self.dpi = dpi
        
        # 设置matplotlib样式
        try:
            plt.style.use(style)
        except:
            logger.warning(f"样式 {style} 不可用，使用默认样式")
        
        # 设置seaborn样式
        try:
            sns.set_palette(color_palette)
        except:
            logger.warning(f"颜色方案 {color_palette} 不可用，使用默认方案")
        
        # 颜色配置
        self.colors = {
            'variable_node': '#2E86C1',    # 蓝色 - 变量节点
            'constraint_node': '#E74C3C',  # 红色 - 约束节点
            'edge_positive': '#28B463',    # 绿色 - 正系数边
            'edge_negative': '#F39C12',    # 橙色 - 负系数边
            'edge_zero': '#BDC3C7'         # 灰色 - 零系数边
        }
        
        logger.info("二分图可视化器初始化完成")
        logger.info(f"  图形大小: {figure_size}, DPI: {dpi}")
    
    def plot_graph_layout(self, 
                         graph: BipartiteGraph,
                         layout: str = 'bipartite',
                         node_size_scale: float = 1.0,
                         edge_width_scale: float = 1.0,
                         show_labels: bool = True,
                         save_path: Optional[str] = None) -> plt.Figure:
        """
        绘制图布局
        
        Args:
            graph: 二分图对象
            layout: 布局类型 ('bipartite', 'spring', 'circular')
            node_size_scale: 节点大小缩放因子
            edge_width_scale: 边宽度缩放因子
            show_labels: 是否显示节点标签
            save_path: 保存路径
            
        Returns:
            matplotlib图形对象
        """
        if not NETWORKX_AVAILABLE:
            logger.error("图布局可视化需要NetworkX")
            return None
        
        try:
            from .serializer import BipartiteGraphSerializer
            
            # 转换为NetworkX图
            serializer = BipartiteGraphSerializer()
            G = serializer.to_networkx(graph)
            
            if G is None:
                logger.error("转换为NetworkX图失败")
                return None
            
            # 创建图形
            fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)
            
            # 分离变量节点和约束节点
            variable_nodes = [n for n, d in G.nodes(data=True) if d.get('bipartite') == 0]
            constraint_nodes = [n for n, d in G.nodes(data=True) if d.get('bipartite') == 1]
            
            # 选择布局算法
            if layout == 'bipartite':
                pos = nx.bipartite_layout(G, variable_nodes)
            elif layout == 'spring':
                pos = nx.spring_layout(G, k=1, iterations=50)
            elif layout == 'circular':
                pos = nx.circular_layout(G)
            else:
                pos = nx.bipartite_layout(G, variable_nodes)
            
            # 计算节点大小（基于度数）
            var_degrees = [G.degree(n) for n in variable_nodes]
            cons_degrees = [G.degree(n) for n in constraint_nodes]
            
            var_sizes = [max(50, d * 10 * node_size_scale) for d in var_degrees]
            cons_sizes = [max(50, d * 10 * node_size_scale) for d in cons_degrees]
            
            # 绘制变量节点
            nx.draw_networkx_nodes(G, pos, 
                                 nodelist=variable_nodes,
                                 node_color=self.colors['variable_node'],
                                 node_size=var_sizes,
                                 alpha=0.8,
                                 label='变量节点')
            
            # 绘制约束节点
            nx.draw_networkx_nodes(G, pos,
                                 nodelist=constraint_nodes,
                                 node_color=self.colors['constraint_node'],
                                 node_size=cons_sizes,
                                 alpha=0.8,
                                 label='约束节点')
            
            # 分类边（根据权重符号）
            edges_positive = [(u, v) for u, v, d in G.edges(data=True) if d.get('weight', 0) > 0]
            edges_negative = [(u, v) for u, v, d in G.edges(data=True) if d.get('weight', 0) < 0]
            edges_zero = [(u, v) for u, v, d in G.edges(data=True) if d.get('weight', 0) == 0]
            
            # 计算边宽度
            edge_weights = [abs(d.get('weight', 1)) for u, v, d in G.edges(data=True)]
            max_weight = max(edge_weights) if edge_weights else 1
            
            # 绘制正权重边
            if edges_positive:
                pos_widths = [abs(G[u][v].get('weight', 1)) / max_weight * 3 * edge_width_scale 
                             for u, v in edges_positive]
                nx.draw_networkx_edges(G, pos,
                                     edgelist=edges_positive,
                                     width=pos_widths,
                                     edge_color=self.colors['edge_positive'],
                                     alpha=0.6)
            
            # 绘制负权重边
            if edges_negative:
                neg_widths = [abs(G[u][v].get('weight', 1)) / max_weight * 3 * edge_width_scale 
                             for u, v in edges_negative]
                nx.draw_networkx_edges(G, pos,
                                     edgelist=edges_negative,
                                     width=neg_widths,
                                     edge_color=self.colors['edge_negative'],
                                     alpha=0.6,
                                     style='dashed')
            
            # 绘制零权重边
            if edges_zero:
                nx.draw_networkx_edges(G, pos,
                                     edgelist=edges_zero,
                                     width=0.5,
                                     edge_color=self.colors['edge_zero'],
                                     alpha=0.3)
            
            # 添加标签（仅对小图）
            if show_labels and len(G.nodes()) <= 50:
                nx.draw_networkx_labels(G, pos, font_size=8, alpha=0.8)
            
            # 设置标题和图例
            ax.set_title(f'二分图布局 - {graph.graph_id}\\n'
                        f'变量节点: {len(variable_nodes)}, 约束节点: {len(constraint_nodes)}',
                        fontsize=14, fontweight='bold')
            
            ax.legend(loc='upper right')
            ax.axis('off')
            
            plt.tight_layout()
            
            # 保存图形
            if save_path:
                plt.savefig(save_path, bbox_inches='tight', dpi=self.dpi)
                logger.info(f"图布局已保存: {save_path}")
            
            return fig
            
        except Exception as e:
            logger.error(f"图布局绘制失败: {e}")
            return None
    
    def plot_feature_heatmap(self, 
                           graph: BipartiteGraph,
                           feature_type: str = 'variable',
                           max_nodes: int = 100,
                           save_path: Optional[str] = None) -> plt.Figure:
        """
        绘制特征向量热图
        
        Args:
            graph: 二分图对象
            feature_type: 特征类型 ('variable', 'constraint')
            max_nodes: 最大显示节点数
            save_path: 保存路径
            
        Returns:
            matplotlib图形对象
        """
        try:
            fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)
            
            if feature_type == 'variable':
                nodes = list(graph.variable_nodes.values())[:max_nodes]
                features = np.array([node.get_feature_vector() for node in nodes])
                node_ids = [node.node_id for node in nodes]
                feature_names = [
                    '变量类型', '下界', '上界', '目标系数', 
                    '有下界', '有上界', '度数', '约束类型', '系数统计'
                ]
                title = f'变量节点特征热图 - {graph.graph_id}'
                
            elif feature_type == 'constraint':
                nodes = list(graph.constraint_nodes.values())[:max_nodes]
                features = np.array([node.get_constraint_features() for node in nodes])
                node_ids = [node.node_id for node in nodes]
                feature_names = [
                    '约束类型', '约束方向', '度数', '非零数', 
                    '右侧值', '平均系数', '最大系数', '系数标准差', '是否紧约束'
                ]
                title = f'约束节点特征热图 - {graph.graph_id}'
            else:
                logger.error(f"不支持的特征类型: {feature_type}")
                return None
            
            # 标准化特征（按列）
            features_normalized = features.copy()
            for i in range(features.shape[1]):
                col = features[:, i]
                if np.std(col) > 1e-12:
                    features_normalized[:, i] = (col - np.mean(col)) / np.std(col)
            
            # 绘制热图
            sns.heatmap(features_normalized.T, 
                       xticklabels=[f'{id[:10]}...' if len(id) > 10 else id for id in node_ids],
                       yticklabels=feature_names,
                       cmap='RdYlBu_r',
                       center=0,
                       annot=False,
                       cbar_kws={'label': '标准化特征值'})
            
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_xlabel('节点ID', fontsize=12)
            ax.set_ylabel('特征维度', fontsize=12)
            
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            # 保存图形
            if save_path:
                plt.savefig(save_path, bbox_inches='tight', dpi=self.dpi)
                logger.info(f"特征热图已保存: {save_path}")
            
            return fig
            
        except Exception as e:
            logger.error(f"特征热图绘制失败: {e}")
            return None
    
    def plot_degree_distribution(self, 
                               graph: BipartiteGraph,
                               save_path: Optional[str] = None) -> plt.Figure:
        """
        绘制度数分布图
        
        Args:
            graph: 二分图对象
            save_path: 保存路径
            
        Returns:
            matplotlib图形对象
        """
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=self.dpi)
            
            # 变量节点度数分布
            var_degrees = [node.degree for node in graph.variable_nodes.values()]
            ax1.hist(var_degrees, bins=min(20, len(set(var_degrees))), 
                    alpha=0.7, color=self.colors['variable_node'], edgecolor='black')
            ax1.set_title('变量节点度数分布', fontsize=12, fontweight='bold')
            ax1.set_xlabel('度数')
            ax1.set_ylabel('节点数量')
            ax1.grid(True, alpha=0.3)
            
            # 添加统计信息
            ax1.axvline(np.mean(var_degrees), color='red', linestyle='--', 
                       label=f'平均值: {np.mean(var_degrees):.2f}')
            ax1.axvline(np.median(var_degrees), color='orange', linestyle='--',
                       label=f'中位数: {np.median(var_degrees):.2f}')
            ax1.legend()
            
            # 约束节点度数分布
            cons_degrees = [node.degree for node in graph.constraint_nodes.values()]
            ax2.hist(cons_degrees, bins=min(20, len(set(cons_degrees))), 
                    alpha=0.7, color=self.colors['constraint_node'], edgecolor='black')
            ax2.set_title('约束节点度数分布', fontsize=12, fontweight='bold')
            ax2.set_xlabel('度数')
            ax2.set_ylabel('节点数量')
            ax2.grid(True, alpha=0.3)
            
            # 添加统计信息
            ax2.axvline(np.mean(cons_degrees), color='red', linestyle='--',
                       label=f'平均值: {np.mean(cons_degrees):.2f}')
            ax2.axvline(np.median(cons_degrees), color='orange', linestyle='--',
                       label=f'中位数: {np.median(cons_degrees):.2f}')
            ax2.legend()
            
            plt.suptitle(f'度数分布分析 - {graph.graph_id}', fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            # 保存图形
            if save_path:
                plt.savefig(save_path, bbox_inches='tight', dpi=self.dpi)
                logger.info(f"度数分布图已保存: {save_path}")
            
            return fig
            
        except Exception as e:
            logger.error(f"度数分布图绘制失败: {e}")
            return None
    
    def plot_coefficient_analysis(self, 
                                graph: BipartiteGraph,
                                save_path: Optional[str] = None) -> plt.Figure:
        """
        绘制系数分布分析图
        
        Args:
            graph: 二分图对象
            save_path: 保存路径
            
        Returns:
            matplotlib图形对象
        """
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12), dpi=self.dpi)
            
            # 提取系数数据
            coefficients = [edge.coefficient for edge in graph.edges.values()]
            abs_coefficients = [abs(coeff) for coeff in coefficients]
            
            # 1. 系数分布直方图
            ax1.hist(coefficients, bins=50, alpha=0.7, color=self.colors['edge_positive'], 
                    edgecolor='black')
            ax1.set_title('系数分布', fontsize=12, fontweight='bold')
            ax1.set_xlabel('系数值')
            ax1.set_ylabel('频次')
            ax1.axvline(np.mean(coefficients), color='red', linestyle='--', 
                       label=f'均值: {np.mean(coefficients):.3f}')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 2. 绝对值系数分布（对数尺度）
            ax2.hist(abs_coefficients, bins=50, alpha=0.7, color=self.colors['edge_negative'],
                    edgecolor='black')
            ax2.set_title('绝对值系数分布', fontsize=12, fontweight='bold')
            ax2.set_xlabel('|系数值|')
            ax2.set_ylabel('频次')
            ax2.set_yscale('log')
            ax2.grid(True, alpha=0.3)
            
            # 3. 系数符号分布
            positive_coeffs = sum(1 for c in coefficients if c > 0)
            negative_coeffs = sum(1 for c in coefficients if c < 0)
            zero_coeffs = sum(1 for c in coefficients if c == 0)
            
            labels = ['正系数', '负系数', '零系数']
            sizes = [positive_coeffs, negative_coeffs, zero_coeffs]
            colors = [self.colors['edge_positive'], self.colors['edge_negative'], self.colors['edge_zero']]
            
            ax3.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            ax3.set_title('系数符号分布', fontsize=12, fontweight='bold')
            
            # 4. 系数范围分析
            coeff_ranges = {
                '(0, 1e-6]': sum(1 for c in abs_coefficients if 0 < c <= 1e-6),
                '(1e-6, 1e-3]': sum(1 for c in abs_coefficients if 1e-6 < c <= 1e-3),
                '(1e-3, 1]': sum(1 for c in abs_coefficients if 1e-3 < c <= 1),
                '(1, 1e3]': sum(1 for c in abs_coefficients if 1 < c <= 1e3),
                '(1e3, inf)': sum(1 for c in abs_coefficients if c > 1e3)
            }
            
            ranges = list(coeff_ranges.keys())
            counts = list(coeff_ranges.values())
            
            bars = ax4.bar(ranges, counts, color=self.colors['variable_node'], alpha=0.7)
            ax4.set_title('系数幅度分布', fontsize=12, fontweight='bold')
            ax4.set_xlabel('系数范围')
            ax4.set_ylabel('边数量')
            ax4.tick_params(axis='x', rotation=45)
            
            # 添加数值标签
            for bar, count in zip(bars, counts):
                if count > 0:
                    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                           str(count), ha='center', va='bottom')
            
            plt.suptitle(f'系数分析 - {graph.graph_id}', fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            # 保存图形
            if save_path:
                plt.savefig(save_path, bbox_inches='tight', dpi=self.dpi)
                logger.info(f"系数分析图已保存: {save_path}")
            
            return fig
            
        except Exception as e:
            logger.error(f"系数分析图绘制失败: {e}")
            return None
    
    def plot_statistics_dashboard(self, 
                                graph: BipartiteGraph,
                                save_path: Optional[str] = None) -> plt.Figure:
        """
        绘制图统计信息仪表板
        
        Args:
            graph: 二分图对象
            save_path: 保存路径
            
        Returns:
            matplotlib图形对象
        """
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12), dpi=self.dpi)
            
            stats = graph.statistics
            
            # 1. 基本统计信息
            basic_stats = [
                stats.n_variable_nodes,
                stats.n_constraint_nodes,
                stats.n_edges,
                stats.n_nonzero_edges
            ]
            basic_labels = ['变量节点', '约束节点', '总边数', '非零边数']
            
            bars1 = ax1.bar(basic_labels, basic_stats, 
                           color=[self.colors['variable_node'], self.colors['constraint_node'],
                                 self.colors['edge_positive'], self.colors['edge_negative']])
            ax1.set_title('基本统计', fontsize=12, fontweight='bold')
            ax1.set_ylabel('数量')
            
            # 添加数值标签
            for bar, stat in zip(bars1, basic_stats):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(basic_stats)*0.01,
                        str(stat), ha='center', va='bottom', fontweight='bold')
            
            # 2. 变量类型分布
            var_type_counts = {
                '连续': stats.n_continuous_vars,
                '二进制': stats.n_binary_vars,
                '整数': stats.n_integer_vars
            }
            
            non_zero_counts = {k: v for k, v in var_type_counts.items() if v > 0}
            if non_zero_counts:
                ax2.pie(non_zero_counts.values(), labels=non_zero_counts.keys(), 
                       autopct='%1.1f%%', startangle=90)
                ax2.set_title('变量类型分布', fontsize=12, fontweight='bold')
            else:
                ax2.text(0.5, 0.5, '无变量数据', ha='center', va='center', transform=ax2.transAxes)
                ax2.set_title('变量类型分布', fontsize=12, fontweight='bold')
            
            # 3. 约束类型分布
            cons_type_counts = {
                '等式': stats.n_equality_constraints,
                '不等式': stats.n_inequality_constraints,
                'SOC': stats.n_soc_constraints,
                '二次': stats.n_quadratic_constraints
            }
            
            non_zero_cons = {k: v for k, v in cons_type_counts.items() if v > 0}
            if non_zero_cons:
                ax3.pie(non_zero_cons.values(), labels=non_zero_cons.keys(),
                       autopct='%1.1f%%', startangle=90)
                ax3.set_title('约束类型分布', fontsize=12, fontweight='bold')
            else:
                ax3.text(0.5, 0.5, '无约束数据', ha='center', va='center', transform=ax3.transAxes)
                ax3.set_title('约束类型分布', fontsize=12, fontweight='bold')
            
            # 4. 关键指标
            key_metrics = {
                '图密度': f'{stats.density:.4f}',
                '平均变量度数': f'{stats.avg_variable_degree:.2f}',
                '平均约束度数': f'{stats.avg_constraint_degree:.2f}',
                '最大变量度数': f'{stats.max_variable_degree}',
                '最大约束度数': f'{stats.max_constraint_degree}'
            }
            
            # 创建指标表格
            ax4.axis('off')
            table_data = [[k, v] for k, v in key_metrics.items()]
            table = ax4.table(cellText=table_data,
                             colLabels=['指标', '数值'],
                             cellLoc='center',
                             loc='center',
                             colWidths=[0.6, 0.4])
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 2)
            ax4.set_title('关键指标', fontsize=12, fontweight='bold')
            
            plt.suptitle(f'图统计仪表板 - {graph.graph_id}', fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            # 保存图形
            if save_path:
                plt.savefig(save_path, bbox_inches='tight', dpi=self.dpi)
                logger.info(f"统计仪表板已保存: {save_path}")
            
            return fig
            
        except Exception as e:
            logger.error(f"统计仪表板绘制失败: {e}")
            return None
    
    def create_interactive_plot(self, 
                              graph: BipartiteGraph,
                              save_path: Optional[str] = None) -> Optional[Any]:
        """
        创建交互式可视化（使用Plotly）
        
        Args:
            graph: 二分图对象
            save_path: 保存路径（HTML格式）
            
        Returns:
            Plotly图形对象或None
        """
        if not PLOTLY_AVAILABLE:
            logger.error("交互式可视化需要Plotly")
            return None
        
        try:
            # 创建子图
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('图统计', '度数分布', '系数分布', '节点信息'),
                specs=[[{"type": "bar"}, {"type": "histogram"}],
                       [{"type": "histogram"}, {"type": "table"}]]
            )
            
            # 1. 基本统计柱状图
            stats = graph.statistics
            fig.add_trace(
                go.Bar(
                    x=['变量节点', '约束节点', '边数'],
                    y=[stats.n_variable_nodes, stats.n_constraint_nodes, stats.n_edges],
                    name='基本统计',
                    marker_color=['blue', 'red', 'green']
                ),
                row=1, col=1
            )
            
            # 2. 变量节点度数分布
            var_degrees = [node.degree for node in graph.variable_nodes.values()]
            fig.add_trace(
                go.Histogram(
                    x=var_degrees,
                    name='变量度数分布',
                    marker_color='blue',
                    opacity=0.7
                ),
                row=1, col=2
            )
            
            # 3. 系数分布
            coefficients = [edge.coefficient for edge in graph.edges.values()]
            fig.add_trace(
                go.Histogram(
                    x=coefficients,
                    name='系数分布',
                    marker_color='green',
                    opacity=0.7
                ),
                row=2, col=1
            )
            
            # 4. 关键指标表格
            metrics_data = [
                ['图密度', f'{stats.density:.4f}'],
                ['平均变量度数', f'{stats.avg_variable_degree:.2f}'],
                ['平均约束度数', f'{stats.avg_constraint_degree:.2f}'],
                ['构建耗时(秒)', f'{stats.build_duration:.3f}' if stats.build_duration else 'N/A']
            ]
            
            fig.add_trace(
                go.Table(
                    header=dict(values=['指标', '数值'],
                               fill_color='lightblue',
                               align='center'),
                    cells=dict(values=list(zip(*metrics_data)),
                              fill_color='lightgray',
                              align='center')
                ),
                row=2, col=2
            )
            
            # 更新布局
            fig.update_layout(
                title_text=f"二分图交互式分析 - {graph.graph_id}",
                title_x=0.5,
                showlegend=False,
                height=800
            )
            
            # 保存HTML文件
            if save_path:
                fig.write_html(save_path)
                logger.info(f"交互式图表已保存: {save_path}")
            
            return fig
            
        except Exception as e:
            logger.error(f"交互式可视化创建失败: {e}")
            return None
    
    def generate_visualization_report(self, 
                                    graph: BipartiteGraph,
                                    output_dir: Union[str, Path],
                                    formats: List[str] = ['png', 'pdf']) -> bool:
        """
        生成完整的可视化报告
        
        Args:
            graph: 二分图对象
            output_dir: 输出目录
            formats: 保存格式列表
            
        Returns:
            是否生成成功
        """
        try:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            base_name = f"{graph.graph_id}_visualization"
            
            # 1. 图布局
            if len(graph.variable_nodes) + len(graph.constraint_nodes) <= 200:
                layout_fig = self.plot_graph_layout(graph)
                if layout_fig:
                    for fmt in formats:
                        layout_fig.savefig(output_dir / f"{base_name}_layout.{fmt}", 
                                         bbox_inches='tight', dpi=self.dpi)
                    plt.close(layout_fig)
            
            # 2. 特征热图
            if len(graph.variable_nodes) <= 100:
                var_heatmap = self.plot_feature_heatmap(graph, 'variable')
                if var_heatmap:
                    for fmt in formats:
                        var_heatmap.savefig(output_dir / f"{base_name}_variable_features.{fmt}",
                                          bbox_inches='tight', dpi=self.dpi)
                    plt.close(var_heatmap)
            
            if len(graph.constraint_nodes) <= 100:
                cons_heatmap = self.plot_feature_heatmap(graph, 'constraint')
                if cons_heatmap:
                    for fmt in formats:
                        cons_heatmap.savefig(output_dir / f"{base_name}_constraint_features.{fmt}",
                                           bbox_inches='tight', dpi=self.dpi)
                    plt.close(cons_heatmap)
            
            # 3. 度数分布
            degree_fig = self.plot_degree_distribution(graph)
            if degree_fig:
                for fmt in formats:
                    degree_fig.savefig(output_dir / f"{base_name}_degree_distribution.{fmt}",
                                     bbox_inches='tight', dpi=self.dpi)
                plt.close(degree_fig)
            
            # 4. 系数分析
            coeff_fig = self.plot_coefficient_analysis(graph)
            if coeff_fig:
                for fmt in formats:
                    coeff_fig.savefig(output_dir / f"{base_name}_coefficient_analysis.{fmt}",
                                    bbox_inches='tight', dpi=self.dpi)
                plt.close(coeff_fig)
            
            # 5. 统计仪表板
            stats_fig = self.plot_statistics_dashboard(graph)
            if stats_fig:
                for fmt in formats:
                    stats_fig.savefig(output_dir / f"{base_name}_statistics_dashboard.{fmt}",
                                    bbox_inches='tight', dpi=self.dpi)
                plt.close(stats_fig)
            
            # 6. 交互式图表
            if PLOTLY_AVAILABLE:
                interactive_fig = self.create_interactive_plot(graph)
                if interactive_fig:
                    interactive_fig.write_html(output_dir / f"{base_name}_interactive.html")
            
            logger.info(f"可视化报告已生成: {output_dir}")
            return True
            
        except Exception as e:
            logger.error(f"可视化报告生成失败: {e}")
            return False


# 便捷函数
def visualize_bipartite_graph(graph: BipartiteGraph,
                            plot_type: str = 'layout',
                            save_path: Optional[str] = None,
                            **kwargs) -> Optional[plt.Figure]:
    """
    可视化二分图（便捷函数）
    
    Args:
        graph: 二分图对象
        plot_type: 图表类型 ('layout', 'features', 'degrees', 'coefficients', 'dashboard')
        save_path: 保存路径
        **kwargs: 其他参数
        
    Returns:
        matplotlib图形对象或None
    """
    visualizer = BipartiteGraphVisualizer()
    
    if plot_type == 'layout':
        return visualizer.plot_graph_layout(graph, save_path=save_path, **kwargs)
    elif plot_type == 'features':
        return visualizer.plot_feature_heatmap(graph, save_path=save_path, **kwargs)
    elif plot_type == 'degrees':
        return visualizer.plot_degree_distribution(graph, save_path=save_path, **kwargs)
    elif plot_type == 'coefficients':
        return visualizer.plot_coefficient_analysis(graph, save_path=save_path, **kwargs)
    elif plot_type == 'dashboard':
        return visualizer.plot_statistics_dashboard(graph, save_path=save_path, **kwargs)
    else:
        logger.error(f"不支持的图表类型: {plot_type}")
        return None


if __name__ == "__main__":
    """测试可视化工具"""
    logger.info("二分图可视化工具测试")
    print("✅ 可视化工具模块加载成功!")
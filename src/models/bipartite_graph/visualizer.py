
"""
Bipartite Graph Visualization Tool
Provides various visualization options for bipartite graphs.

Visualization Functions:
1. NetworkX layout visualization
2. Feature vector heatmap
3. Degree distribution plot
4. Coefficient distribution analysis
5. Graph structure statistics dashboard
6. Interactive visualization (if supported)
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
    logger.warning("NetworkX is not available, graph layout visualization is limited.")

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    logger.warning("Plotly is not available, interactive visualization is not supported.")


class BipartiteGraphVisualizer:
    """
    Bipartite Graph Visualization Tool
    Supports multiple visualization schemes and custom configurations.
    """
    
    def __init__(self, 
                 figure_size: Tuple[int, int] = (12, 8),
                 dpi: int = 300,
                 style: str = 'seaborn-v0_8',
                 color_palette: str = 'Set2'):
        """
        Initializes the visualizer.
        
        Args:
            figure_size: Figure size.
            dpi: Image resolution.
            style: matplotlib style.
            color_palette: Color scheme.
        """
        self.figure_size = figure_size
        self.dpi = dpi
        
        # Configure matplotlib for English fonts
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['font.family'] = 'sans-serif'
        
        try:
            plt.style.use(style)
        except:
            logger.warning(f"Style '{style}' not available, using default.")
        
        try:
            sns.set_palette(color_palette)
        except:
            logger.warning(f"Color palette '{color_palette}' not available, using default.")
        
        self.colors = {
            'variable_node': '#2E86C1',    # Blue - Variable Node
            'constraint_node': '#E74C3C',  # Red - Constraint Node
            'edge_positive': '#28B463',    # Green - Positive Edge
            'edge_negative': '#F39C12',    # Orange - Negative Edge
            'edge_zero': '#BDC3C7'         # Gray - Zero Edge
        }
        
        logger.info("BipartiteGraphVisualizer initialized.")
        logger.info(f"  Figure Size: {figure_size}, DPI: {dpi}")
    
    def plot_graph_layout(self, 
                         graph: BipartiteGraph,
                         layout: str = 'bipartite',
                         node_size_scale: float = 1.0,
                         edge_width_scale: float = 1.0,
                         show_labels: bool = True,
                         save_path: Optional[str] = None) -> plt.Figure:
        """
        Plots the graph layout.
        
        Args:
            graph: BipartiteGraph object.
            layout: Layout type ('bipartite', 'spring', 'circular').
            node_size_scale: Node size scaling factor.
            edge_width_scale: Edge width scaling factor.
            show_labels: Whether to show node labels.
            save_path: Path to save the figure.
            
        Returns:
            matplotlib Figure object.
        """
        if not NETWORKX_AVAILABLE:
            logger.error("Graph layout visualization requires NetworkX.")
            return None
        
        try:
            from .serializer import BipartiteGraphSerializer
            
            serializer = BipartiteGraphSerializer()
            G = serializer.to_networkx(graph)
            
            if G is None:
                logger.error("Failed to convert to NetworkX graph.")
                return None
            
            fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)
            
            variable_nodes = [n for n, d in G.nodes(data=True) if d.get('bipartite') == 0]
            constraint_nodes = [n for n, d in G.nodes(data=True) if d.get('bipartite') == 1]
            
            if layout == 'bipartite':
                pos = nx.bipartite_layout(G, variable_nodes)
            elif layout == 'spring':
                pos = nx.spring_layout(G, k=1, iterations=50)
            elif layout == 'circular':
                pos = nx.circular_layout(G)
            else:
                pos = nx.bipartite_layout(G, variable_nodes)
            
            var_degrees = [G.degree(n) for n in variable_nodes]
            cons_degrees = [G.degree(n) for n in constraint_nodes]
            
            var_sizes = [max(50, d * 10 * node_size_scale) for d in var_degrees]
            cons_sizes = [max(50, d * 10 * node_size_scale) for d in cons_degrees]
            
            nx.draw_networkx_nodes(G, pos, 
                                 nodelist=variable_nodes,
                                 node_color=self.colors['variable_node'],
                                 node_size=var_sizes,
                                 alpha=0.8,
                                 label='Variable Nodes')
            
            nx.draw_networkx_nodes(G, pos,
                                 nodelist=constraint_nodes,
                                 node_color=self.colors['constraint_node'],
                                 node_size=cons_sizes,
                                 alpha=0.8,
                                 label='Constraint Nodes')
            
            edges_positive = [(u, v) for u, v, d in G.edges(data=True) if d.get('weight', 0) > 0]
            edges_negative = [(u, v) for u, v, d in G.edges(data=True) if d.get('weight', 0) < 0]
            edges_zero = [(u, v) for u, v, d in G.edges(data=True) if d.get('weight', 0) == 0]
            
            edge_weights = [abs(d.get('weight', 1)) for u, v, d in G.edges(data=True)]
            max_weight = max(edge_weights) if edge_weights else 1
            
            if edges_positive:
                pos_widths = [abs(G[u][v].get('weight', 1)) / max_weight * 3 * edge_width_scale for u, v in edges_positive]
                nx.draw_networkx_edges(G, pos, edgelist=edges_positive, width=pos_widths, edge_color=self.colors['edge_positive'], alpha=0.6)
            
            if edges_negative:
                neg_widths = [abs(G[u][v].get('weight', 1)) / max_weight * 3 * edge_width_scale for u, v in edges_negative]
                nx.draw_networkx_edges(G, pos, edgelist=edges_negative, width=neg_widths, edge_color=self.colors['edge_negative'], alpha=0.6, style='dashed')
            
            if edges_zero:
                nx.draw_networkx_edges(G, pos, edgelist=edges_zero, width=0.5, edge_color=self.colors['edge_zero'], alpha=0.3)
            
            if show_labels and len(G.nodes()) <= 50:
                nx.draw_networkx_labels(G, pos, font_size=8, alpha=0.8)
            
            ax.set_title(f'Bipartite Graph Layout - {graph.graph_id}\n'
                        f'Variable Nodes: {len(variable_nodes)}, Constraint Nodes: {len(constraint_nodes)}',
                        fontsize=14, fontweight='bold')
            
            ax.legend(loc='upper right')
            ax.axis('off')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, bbox_inches='tight', dpi=self.dpi)
                logger.info(f"Graph layout saved to: {save_path}")
            
            return fig
            
        except Exception as e:
            logger.error(f"Failed to plot graph layout: {e}")
            return None
    
    def plot_feature_heatmap(self, 
                           graph: BipartiteGraph,
                           feature_type: str = 'variable',
                           max_nodes: int = 100,
                           save_path: Optional[str] = None) -> plt.Figure:
        """
        Plots a heatmap of feature vectors.
        """
        try:
            fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)
            
            if feature_type == 'variable':
                nodes = list(graph.variable_nodes.values())[:max_nodes]
                features = np.array([node.get_feature_vector() for node in nodes])
                node_ids = [node.node_id for node in nodes]
                feature_names = ['Var Type', 'Lower Bound', 'Upper Bound', 'Obj Coeff', 'Has LB', 'Has UB', 'Degree', 'Constr Type', 'Coeff Stats']
                title = f'Variable Node Feature Heatmap - {graph.graph_id}'
                
            elif feature_type == 'constraint':
                nodes = list(graph.constraint_nodes.values())[:max_nodes]
                features = np.array([node.get_constraint_features() for node in nodes])
                node_ids = [node.node_id for node in nodes]
                feature_names = ['Constr Type', 'Direction', 'Degree', 'Non-zeros', 'RHS', 'Avg Coeff', 'Max Coeff', 'Std Coeff', 'Is Tight']
                title = f'Constraint Node Feature Heatmap - {graph.graph_id}'
            else:
                logger.error(f"Unsupported feature type: {feature_type}")
                return None
            
            features_normalized = features.copy()
            for i in range(features.shape[1]):
                col = features[:, i]
                if np.std(col) > 1e-12:
                    features_normalized[:, i] = (col - np.mean(col)) / np.std(col)
            
            sns.heatmap(features_normalized.T, 
                       xticklabels=[f'{id[:10]}...' if len(id) > 10 else id for id in node_ids],
                       yticklabels=feature_names,
                       cmap='RdYlBu_r',
                       center=0,
                       annot=False,
                       cbar_kws={'label': 'Normalized Feature Value'})
            
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_xlabel('Node ID', fontsize=12)
            ax.set_ylabel('Feature Dimension', fontsize=12)
            
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, bbox_inches='tight', dpi=self.dpi)
                logger.info(f"Feature heatmap saved to: {save_path}")
            
            return fig
            
        except Exception as e:
            logger.error(f"Failed to plot feature heatmap: {e}")
            return None

    def plot_degree_distribution(self, 
                               graph: BipartiteGraph,
                               save_path: Optional[str] = None) -> plt.Figure:
        """
        Plots the degree distribution.
        """
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=self.dpi)
            
            var_degrees = [node.degree for node in graph.variable_nodes.values()]
            ax1.hist(var_degrees, bins=min(20, len(set(var_degrees))), alpha=0.7, color=self.colors['variable_node'], edgecolor='black')
            ax1.set_title('Variable Node Degree Distribution', fontsize=12, fontweight='bold')
            ax1.set_xlabel('Degree')
            ax1.set_ylabel('Node Count')
            ax1.grid(True, alpha=0.3)
            ax1.axvline(np.mean(var_degrees), color='red', linestyle='--', label=f'Mean: {np.mean(var_degrees):.2f}')
            ax1.axvline(np.median(var_degrees), color='orange', linestyle='--', label=f'Median: {np.median(var_degrees):.2f}')
            ax1.legend()
            
            cons_degrees = [node.degree for node in graph.constraint_nodes.values()]
            ax2.hist(cons_degrees, bins=min(20, len(set(cons_degrees))), alpha=0.7, color=self.colors['constraint_node'], edgecolor='black')
            ax2.set_title('Constraint Node Degree Distribution', fontsize=12, fontweight='bold')
            ax2.set_xlabel('Degree')
            ax2.set_ylabel('Node Count')
            ax2.grid(True, alpha=0.3)
            ax2.axvline(np.mean(cons_degrees), color='red', linestyle='--', label=f'Mean: {np.mean(cons_degrees):.2f}')
            ax2.axvline(np.median(cons_degrees), color='orange', linestyle='--', label=f'Median: {np.median(cons_degrees):.2f}')
            ax2.legend()
            
            plt.suptitle(f'Degree Distribution Analysis - {graph.graph_id}', fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, bbox_inches='tight', dpi=self.dpi)
                logger.info(f"Degree distribution plot saved to: {save_path}")
            
            return fig
            
        except Exception as e:
            logger.error(f"Failed to plot degree distribution: {e}")
            return None

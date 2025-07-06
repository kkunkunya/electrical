"""
G2MILP二分图数据表示方法使用演示
G2MILP Bipartite Graph Data Representation Usage Demo

本演示脚本展示了如何：
1. 使用现有的MILP生成器创建优化问题实例
2. 将MILP实例转换为G2MILP框架的二分图表示
3. 分析二分图的结构和特征
4. 导出为不同的图神经网络框架格式
5. 进行批量处理和统计分析

这是一个完整的端到端演示，展示了G2MILP在电力系统优化问题中的应用。
"""

import sys
import logging
from pathlib import Path
from datetime import datetime
import numpy as np
import json

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 导入核心模块
try:
    from src.datasets.loader import load_system_data
    from src.models.biased_milp_generator import (
        BiasedMILPGenerator, 
        PerturbationConfig,
        create_scenario_perturbation_configs
    )
    from src.models.g2milp_bipartite import create_g2milp_generator
except ImportError as e:
    print(f"模块导入失败: {e}")
    print("请确保在项目根目录运行此脚本")
    sys.exit(1)

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class G2MILPDemo:
    """G2MILP演示类"""
    
    def __init__(self, data_dir: str = "data", output_dir: str = "output/g2milp_demo"):
        """
        初始化演示环境
        
        Args:
            data_dir: 数据目录
            output_dir: 输出目录
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"G2MILP演示初始化 - 数据目录: {self.data_dir}, 输出目录: {self.output_dir}")
    
    def step_1_load_data(self):
        """步骤1: 加载电力系统数据"""
        logger.info("="*60)
        logger.info("步骤 1: 加载电力系统数据")
        logger.info("="*60)
        
        if not self.data_dir.exists():
            raise FileNotFoundError(f"数据目录不存在: {self.data_dir}")
        
        # 加载系统数据
        self.system_data = load_system_data(str(self.data_dir))
        
        logger.info("✅ 电力系统数据加载成功")
        logger.info(f"  📊 发电机数量: {len(self.system_data.generators)}")
        logger.info(f"  📊 负荷节点数: {len(self.system_data.loads)}")
        logger.info(f"  📊 支路数量: {len(self.system_data.branches)}")
        
        return self.system_data
    
    def step_2_create_milp_instance(self):
        """步骤2: 创建MILP实例"""
        logger.info("="*60)
        logger.info("步骤 2: 创建MILP实例")
        logger.info("="*60)
        
        # 创建MILP生成器
        self.milp_generator = BiasedMILPGenerator(
            base_system_data=self.system_data,
            output_dir=str(self.output_dir / "milp_instances")
        )
        
        # 配置数据扰动参数
        perturbation_config = PerturbationConfig(
            load_perturbation_type="gaussian",
            load_noise_std=0.1,                    # 负荷10%标准差扰动
            generator_perturbation_type="gaussian",
            generator_noise_std=0.05,              # 发电机5%标准差扰动
            pv_noise_std=0.15,                     # 光伏15%标准差扰动
            perturbation_intensity=1.0,            # 100%扰动强度
            random_seed=42                         # 可重现性
        )
        
        # 生成MILP实例
        self.milp_instance = self.milp_generator.generate_single_instance(
            perturbation_config=perturbation_config,
            instance_id="g2milp_demo_instance",
            n_periods=21,                          # 21个时间段（3:00-23:00）
            start_hour=3,
            save_to_file=True
        )
        
        logger.info("✅ MILP实例创建成功")
        logger.info(f"  📊 实例ID: {self.milp_instance.instance_id}")
        logger.info(f"  📊 变量数量: {self.milp_instance.statistics.n_variables}")
        logger.info(f"  📊 约束数量: {self.milp_instance.statistics.n_constraints}")
        logger.info(f"  📊 二进制变量: {self.milp_instance.statistics.n_binary_vars}")
        logger.info(f"  📊 连续变量: {self.milp_instance.statistics.n_continuous_vars}")
        
        return self.milp_instance
    
    def step_3_generate_bipartite_graph(self):
        """步骤3: 生成G2MILP二分图表示"""
        logger.info("="*60)
        logger.info("步骤 3: 生成G2MILP二分图表示")
        logger.info("="*60)
        
        # 生成二分图表示
        success = self.milp_instance.generate_bipartite_graph(
            include_power_system_semantics=True
        )
        
        if not success:
            raise RuntimeError("二分图生成失败")
        
        bg = self.milp_instance.bipartite_graph
        
        logger.info("✅ G2MILP二分图生成成功")
        logger.info(f"  📊 约束节点数: {bg.n_constraint_nodes}")
        logger.info(f"  📊 变量节点数: {bg.n_variable_nodes}")
        logger.info(f"  📊 边数量: {bg.n_edges}")
        logger.info(f"  📊 二分图密度: {bg.graph_statistics.get('bipartite_density', 0):.6f}")
        logger.info(f"  📊 平均约束度数: {bg.graph_statistics.get('avg_constraint_degree', 0):.2f}")
        logger.info(f"  📊 平均变量度数: {bg.graph_statistics.get('avg_variable_degree', 0):.2f}")
        
        # 保存二分图
        graph_path = self.output_dir / f"{self.milp_instance.instance_id}_bipartite.pkl"
        self.milp_instance.save_bipartite_graph(str(graph_path))
        logger.info(f"  💾 二分图已保存: {graph_path}")
        
        return bg
    
    def step_4_analyze_features(self):
        """步骤4: 分析二分图特征"""
        logger.info("="*60)
        logger.info("步骤 4: 分析二分图特征")
        logger.info("="*60)
        
        bg = self.milp_instance.bipartite_graph
        
        # 分析变量节点9维特征向量
        logger.info("📊 变量节点9维特征向量分析:")
        variable_features = bg.variable_feature_matrix
        feature_names = [
            "变量类型", "目标函数系数", "下界", "上界", "变量度数",
            "系数均值", "系数标准差", "系数最大值", "索引归一化"
        ]
        
        feature_analysis = {}
        for i, name in enumerate(feature_names):
            values = variable_features[:, i]
            stats = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values))
            }
            feature_analysis[name] = stats
            logger.info(f"  {name}: 均值={stats['mean']:.3f}, 标准差={stats['std']:.3f}, "
                       f"范围=[{stats['min']:.3f}, {stats['max']:.3f}]")
        
        # 分析约束节点特征
        logger.info("📊 约束节点特征分析:")
        constraint_features = bg.constraint_feature_matrix
        
        # 约束类型分布
        constraint_types = constraint_features[:, 0].astype(int)
        type_counts = np.bincount(constraint_types)
        logger.info(f"  约束类型分布: {dict(enumerate(type_counts))}")
        
        # 行密度统计
        row_densities = constraint_features[:, 4]
        logger.info(f"  行密度: 均值={np.mean(row_densities):.4f}, "
                   f"标准差={np.std(row_densities):.4f}")
        
        # 约束度数统计
        constraint_degrees = constraint_features[:, 11]
        logger.info(f"  约束度数: 均值={np.mean(constraint_degrees):.2f}, "
                   f"最大={np.max(constraint_degrees):.0f}, "
                   f"最小={np.min(constraint_degrees):.0f}")
        
        # 分析边特征
        logger.info("📊 边特征分析:")
        if bg.n_edges > 0:
            edge_features = bg.edge_feature_matrix
            
            # 系数分布
            coefficients = edge_features[:, 0]  # 原始系数
            abs_coefficients = edge_features[:, 1]  # 绝对值
            
            logger.info(f"  系数分布: 均值={np.mean(coefficients):.3f}, "
                       f"标准差={np.std(coefficients):.3f}")
            logger.info(f"  系数绝对值: 均值={np.mean(abs_coefficients):.3f}, "
                       f"最大={np.max(abs_coefficients):.3f}, "
                       f"最小={np.min(abs_coefficients):.6f}")
        
        # 保存特征分析结果
        analysis_results = {
            'instance_id': self.milp_instance.instance_id,
            'generation_time': datetime.now().isoformat(),
            'graph_basic_stats': {
                'n_constraint_nodes': bg.n_constraint_nodes,
                'n_variable_nodes': bg.n_variable_nodes,
                'n_edges': bg.n_edges,
                'bipartite_density': bg.graph_statistics.get('bipartite_density', 0)
            },
            'variable_features_analysis': feature_analysis,
            'constraint_features_analysis': {
                'type_distribution': {str(k): int(v) for k, v in enumerate(type_counts)},
                'row_density_stats': {
                    'mean': float(np.mean(row_densities)),
                    'std': float(np.std(row_densities)),
                    'min': float(np.min(row_densities)),
                    'max': float(np.max(row_densities))
                },
                'degree_stats': {
                    'mean': float(np.mean(constraint_degrees)),
                    'std': float(np.std(constraint_degrees)),
                    'min': float(np.min(constraint_degrees)),
                    'max': float(np.max(constraint_degrees))
                }
            }
        }
        
        if bg.n_edges > 0:
            analysis_results['edge_features_analysis'] = {
                'coefficient_stats': {
                    'mean': float(np.mean(coefficients)),
                    'std': float(np.std(coefficients)),
                    'min': float(np.min(coefficients)),
                    'max': float(np.max(coefficients))
                },
                'abs_coefficient_stats': {
                    'mean': float(np.mean(abs_coefficients)),
                    'std': float(np.std(abs_coefficients)),
                    'min': float(np.min(abs_coefficients)),
                    'max': float(np.max(abs_coefficients))
                }
            }
        
        # 保存分析结果
        analysis_path = self.output_dir / "feature_analysis.json"
        with open(analysis_path, 'w', encoding='utf-8') as f:
            json.dump(analysis_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ 特征分析完成，结果已保存: {analysis_path}")
        
        return analysis_results
    
    def step_5_format_export(self):
        """步骤5: 格式转换和导出"""
        logger.info("="*60)
        logger.info("步骤 5: 格式转换和导出")
        logger.info("="*60)
        
        bg = self.milp_instance.bipartite_graph
        
        # 1. 导出原始特征矩阵
        features_path = self.output_dir / "g2milp_features.npz"
        np.savez_compressed(
            features_path,
            constraint_features=bg.constraint_feature_matrix,
            variable_features=bg.variable_feature_matrix,
            edge_features=bg.edge_feature_matrix,
            edges=np.array(bg.edges),
            constraint_matrix=bg.constraint_matrix.toarray() if hasattr(bg.constraint_matrix, 'toarray') else bg.constraint_matrix,
            objective_coeffs=bg.objective_coeffs,
            rhs_values=bg.rhs_values
        )
        logger.info(f"✅ 特征矩阵已导出: {features_path}")
        
        # 2. 尝试导出PyTorch Geometric格式
        try:
            pyg_data = self.milp_instance.export_pytorch_geometric()
            if pyg_data is not None:
                logger.info("✅ PyTorch Geometric格式导出成功")
                logger.info(f"  📊 节点类型: {pyg_data.node_types}")
                logger.info(f"  📊 边类型: {pyg_data.edge_types}")
                logger.info(f"  📊 约束特征形状: {pyg_data['constraint'].x.shape}")
                logger.info(f"  📊 变量特征形状: {pyg_data['variable'].x.shape}")
                
                # 保存PyTorch Geometric数据
                import torch
                pyg_path = self.output_dir / "g2milp_pyg_data.pt"
                torch.save(pyg_data, pyg_path)
                logger.info(f"  💾 PyTorch Geometric数据已保存: {pyg_path}")
            else:
                logger.warning("⚠️ PyTorch Geometric导出失败")
        except Exception as e:
            logger.warning(f"⚠️ PyTorch Geometric不可用: {e}")
        
        # 3. 尝试导出DGL格式
        try:
            dgl_graph = self.milp_instance.export_dgl_graph()
            if dgl_graph is not None:
                logger.info("✅ DGL格式导出成功")
                logger.info(f"  📊 节点类型: {dgl_graph.ntypes}")
                logger.info(f"  📊 边类型: {dgl_graph.etypes}")
                logger.info(f"  📊 约束节点数: {dgl_graph.num_nodes('constraint')}")
                logger.info(f"  📊 变量节点数: {dgl_graph.num_nodes('variable')}")
                
                # 保存DGL数据
                import dgl
                dgl_path = self.output_dir / "g2milp_dgl_graph.pkl"
                dgl.save_graphs(str(dgl_path), [dgl_graph])
                logger.info(f"  💾 DGL图数据已保存: {dgl_path}")
            else:
                logger.warning("⚠️ DGL导出失败")
        except Exception as e:
            logger.warning(f"⚠️ DGL不可用: {e}")
        
        logger.info("✅ 格式转换和导出完成")
    
    def step_6_batch_processing_demo(self):
        """步骤6: 批量处理演示"""
        logger.info("="*60)
        logger.info("步骤 6: 批量处理演示")
        logger.info("="*60)
        
        # 创建多种扰动场景
        scenario_configs = create_scenario_perturbation_configs()
        
        # 选择几个有代表性的场景进行演示
        selected_scenarios = {
            "负荷高峰": scenario_configs["load_peak"],
            "光伏不稳定": scenario_configs["pv_unstable"],
            "交通拥堵": scenario_configs["traffic_jam"]
        }
        
        logger.info(f"生成 {len(selected_scenarios)} 个场景的MILP实例...")
        
        # 生成场景实例
        scenario_instances = self.milp_generator.generate_scenario_instances(
            scenario_configs=selected_scenarios,
            n_periods=21,
            start_hour=3,
            save_to_file=True
        )
        
        logger.info(f"✅ 场景实例生成完成，共 {len(scenario_instances)} 个")
        
        # 批量生成二分图
        instances_list = list(scenario_instances.values())
        updated_instances = self.milp_generator.generate_bipartite_graphs_for_instances(
            instances=instances_list,
            include_power_system_semantics=True,
            save_graphs=True,
            graph_output_dir=str(self.output_dir / "scenario_graphs")
        )
        
        # 分析二分图统计
        analysis = self.milp_generator.analyze_bipartite_graph_statistics(updated_instances)
        
        logger.info("✅ 批量二分图统计分析:")
        logger.info(f"  📊 实例总数: {analysis['total_instances']}")
        logger.info(f"  📊 有效图数: {analysis['valid_bipartite_graphs']}")
        logger.info(f"  📊 覆盖率: {analysis['coverage_rate']:.2%}")
        logger.info(f"  📊 平均约束节点: {analysis['constraint_nodes_stats']['mean']:.1f}")
        logger.info(f"  📊 平均变量节点: {analysis['variable_nodes_stats']['mean']:.1f}")
        logger.info(f"  📊 平均边数: {analysis['edges_stats']['mean']:.1f}")
        logger.info(f"  📊 平均图密度: {analysis['density_stats']['mean']:.6f}")
        
        # 保存批量处理结果
        batch_results = {
            'scenarios': list(selected_scenarios.keys()),
            'analysis': analysis,
            'individual_stats': {}
        }
        
        for scenario_name, instance in scenario_instances.items():
            if instance.bipartite_graph is not None:
                batch_results['individual_stats'][scenario_name] = {
                    'n_constraint_nodes': instance.bipartite_graph.n_constraint_nodes,
                    'n_variable_nodes': instance.bipartite_graph.n_variable_nodes,
                    'n_edges': instance.bipartite_graph.n_edges,
                    'bipartite_density': instance.bipartite_graph.graph_statistics.get('bipartite_density', 0)
                }
        
        batch_path = self.output_dir / "batch_processing_results.json"
        with open(batch_path, 'w', encoding='utf-8') as f:
            json.dump(batch_results, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"✅ 批量处理结果已保存: {batch_path}")
        
        return batch_results
    
    def run_complete_demo(self):
        """运行完整演示"""
        logger.info("🚀 开始G2MILP二分图数据表示方法完整演示")
        logger.info("="*80)
        
        try:
            # 执行所有步骤
            self.step_1_load_data()
            self.step_2_create_milp_instance()
            self.step_3_generate_bipartite_graph()
            feature_analysis = self.step_4_analyze_features()
            self.step_5_format_export()
            batch_results = self.step_6_batch_processing_demo()
            
            # 生成总结报告
            summary_report = {
                'demo_completion_time': datetime.now().isoformat(),
                'output_directory': str(self.output_dir),
                'main_instance': {
                    'instance_id': self.milp_instance.instance_id,
                    'milp_stats': {
                        'n_variables': self.milp_instance.statistics.n_variables,
                        'n_constraints': self.milp_instance.statistics.n_constraints,
                        'n_binary_vars': self.milp_instance.statistics.n_binary_vars,
                        'n_continuous_vars': self.milp_instance.statistics.n_continuous_vars
                    },
                    'bipartite_graph_stats': {
                        'n_constraint_nodes': self.milp_instance.bipartite_graph.n_constraint_nodes,
                        'n_variable_nodes': self.milp_instance.bipartite_graph.n_variable_nodes,
                        'n_edges': self.milp_instance.bipartite_graph.n_edges,
                        'bipartite_density': self.milp_instance.bipartite_graph.graph_statistics.get('bipartite_density', 0)
                    }
                },
                'batch_processing': batch_results,
                'files_generated': {
                    'milp_instances': 'milp_instances/',
                    'bipartite_graphs': 'scenario_graphs/',
                    'feature_analysis': 'feature_analysis.json',
                    'numpy_features': 'g2milp_features.npz',
                    'batch_results': 'batch_processing_results.json'
                }
            }
            
            summary_path = self.output_dir / "demo_summary_report.json"
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary_report, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info("="*80)
            logger.info("🎉 G2MILP演示完成！")
            logger.info("="*80)
            logger.info("📊 演示总结:")
            logger.info(f"  • 主实例: {self.milp_instance.instance_id}")
            logger.info(f"  • MILP变量数: {self.milp_instance.statistics.n_variables}")
            logger.info(f"  • MILP约束数: {self.milp_instance.statistics.n_constraints}")
            logger.info(f"  • 二分图约束节点: {self.milp_instance.bipartite_graph.n_constraint_nodes}")
            logger.info(f"  • 二分图变量节点: {self.milp_instance.bipartite_graph.n_variable_nodes}")
            logger.info(f"  • 二分图边数: {self.milp_instance.bipartite_graph.n_edges}")
            logger.info(f"  • 批量处理场景数: {len(batch_results['scenarios'])}")
            logger.info("="*80)
            logger.info(f"📁 所有结果保存在: {self.output_dir}")
            logger.info(f"📄 总结报告: {summary_path}")
            logger.info("="*80)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 演示执行失败: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """主函数"""
    print("🎯 G2MILP二分图数据表示方法演示")
    print("="*80)
    print("本演示将展示如何:")
    print("• 将电力系统MILP优化问题转换为G2MILP二分图表示")
    print("• 分析约束节点、变量节点和边的特征")
    print("• 导出为PyTorch Geometric和DGL格式")
    print("• 进行批量处理和统计分析")
    print("="*80)
    
    # 检查数据目录
    data_dir = Path("data")
    if not data_dir.exists():
        print(f"❌ 数据目录不存在: {data_dir}")
        print("请确保data/目录包含必要的电力系统数据文件")
        return
    
    try:
        # 创建并运行演示
        demo = G2MILPDemo(
            data_dir=str(data_dir),
            output_dir="output/g2milp_demo"
        )
        
        success = demo.run_complete_demo()
        
        if success:
            print("\n✅ 演示成功完成！")
            print(f"📁 查看结果: {demo.output_dir}")
        else:
            print("\n❌ 演示执行失败")
    
    except KeyboardInterrupt:
        print("\n👋 演示被用户中断")
    except Exception as e:
        print(f"\n❌ 演示执行出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
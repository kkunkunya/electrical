"""
G2MILP转换器主入口模块
整合所有二分图相关功能的统一接口

主要功能：
1. 统一的转换接口
2. 批量处理支持
3. 配置管理
4. 结果分析和报告
5. 与现有MILP生成器的集成
"""

import cvxpy as cp
import numpy as np
import logging
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from datetime import datetime
import json

from .bipartite_graph import (
    BipartiteGraph, G2MILPConverter, ConversionConfig, ConversionResult,
    BipartiteGraphSerializer, BipartiteGraphVisualizer, BipartiteGraphValidator,
    save_bipartite_graph, load_bipartite_graph, validate_bipartite_graph,
    visualize_bipartite_graph
)

logger = logging.getLogger(__name__)


class G2MILPSystem:
    """
    G2MILP系统主类
    提供完整的CVXPY到二分图转换和分析功能
    """
    
    def __init__(self, 
                 output_dir: str = "output/g2milp",
                 config: ConversionConfig = None):
        """
        初始化G2MILP系统
        
        Args:
            output_dir: 输出目录
            config: 转换配置
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 使用默认配置或提供的配置
        if config is None:
            config = ConversionConfig(
                use_sparse_matrix=True,
                normalize_coefficients=True,
                save_intermediate_results=True,
                output_directory=str(self.output_dir),
                validate_graph=True,
                compute_statistics=True
            )
        else:
            config.output_directory = str(self.output_dir)
        
        self.config = config
        
        # 初始化组件
        self.converter = G2MILPConverter(config)
        self.serializer = BipartiteGraphSerializer()
        self.visualizer = BipartiteGraphVisualizer()
        self.validator = BipartiteGraphValidator()
        
        # 系统状态
        self.conversion_results: List[ConversionResult] = []
        self.system_statistics = {
            'total_conversions': 0,
            'successful_conversions': 0,
            'failed_conversions': 0,
            'total_graphs_created': 0,
            'last_conversion_time': None
        }
        
        logger.info(f"G2MILP系统初始化完成")
        logger.info(f"输出目录: {self.output_dir}")
        logger.info(f"配置: 稀疏矩阵={config.use_sparse_matrix}, 验证={config.validate_graph}")
    
    def convert_cvxpy_problem(self, 
                            problem: cp.Problem,
                            problem_name: str = None,
                            save_graph: bool = True,
                            generate_visualizations: bool = False,
                            validate_result: bool = True) -> Dict[str, Any]:
        """
        转换单个CVXPY问题
        
        Args:
            problem: CVXPY问题对象
            problem_name: 问题名称
            save_graph: 是否保存图到文件
            generate_visualizations: 是否生成可视化
            validate_result: 是否验证结果
            
        Returns:
            转换结果字典
        """
        logger.info(f"开始转换CVXPY问题: {problem_name}")
        
        start_time = datetime.now()
        
        try:
            # 1. 执行转换
            result = self.converter.convert_problem(problem, problem_name)
            
            # 更新系统统计
            self.system_statistics['total_conversions'] += 1
            self.system_statistics['last_conversion_time'] = start_time.isoformat()
            
            if result.errors:
                self.system_statistics['failed_conversions'] += 1
                logger.error(f"转换失败: {result.errors}")
                return {
                    'success': False,
                    'errors': result.errors,
                    'conversion_result': result
                }
            else:
                self.system_statistics['successful_conversions'] += 1
                self.system_statistics['total_graphs_created'] += 1
            
            # 2. 验证结果（如果需要）
            validation_report = None
            if validate_result:
                validation_report = self.validator.validate_graph(result.bipartite_graph)
            
            # 3. 保存图（如果需要）
            saved_paths = {}
            if save_graph:
                graph_name = f"{result.conversion_id}_graph"
                
                # 保存Pickle格式
                pickle_path = self.output_dir / f"{graph_name}.pkl"
                if self.serializer.save_pickle(result.bipartite_graph, pickle_path):
                    saved_paths['pickle'] = str(pickle_path)
                
                # 保存JSON格式（轻量级）
                json_path = self.output_dir / f"{graph_name}.json"
                if self.serializer.save_json(result.bipartite_graph, json_path, include_features=False):
                    saved_paths['json'] = str(json_path)
            
            # 4. 生成可视化（如果需要）
            visualization_paths = {}
            if generate_visualizations:
                viz_dir = self.output_dir / f"{result.conversion_id}_visualizations"
                viz_dir.mkdir(exist_ok=True)
                
                try:
                    # 统计仪表板（总是生成）
                    stats_fig = self.visualizer.plot_statistics_dashboard(result.bipartite_graph)
                    if stats_fig:
                        stats_path = viz_dir / "statistics_dashboard.png"
                        stats_fig.savefig(stats_path, bbox_inches='tight', dpi=300)
                        visualization_paths['statistics'] = str(stats_path)
                        import matplotlib.pyplot as plt
                        plt.close(stats_fig)
                    
                    # 度数分布
                    degree_fig = self.visualizer.plot_degree_distribution(result.bipartite_graph)
                    if degree_fig:
                        degree_path = viz_dir / "degree_distribution.png"
                        degree_fig.savefig(degree_path, bbox_inches='tight', dpi=300)
                        visualization_paths['degrees'] = str(degree_path)
                        plt.close(degree_fig)
                    
                    # 图布局（仅对小图）
                    total_nodes = len(result.bipartite_graph.variable_nodes) + len(result.bipartite_graph.constraint_nodes)
                    if total_nodes <= 200:
                        layout_fig = self.visualizer.plot_graph_layout(result.bipartite_graph)
                        if layout_fig:
                            layout_path = viz_dir / "graph_layout.png"
                            layout_fig.savefig(layout_path, bbox_inches='tight', dpi=300)
                            visualization_paths['layout'] = str(layout_path)
                            plt.close(layout_fig)
                    
                except Exception as e:
                    logger.warning(f"可视化生成部分失败: {e}")
            
            # 5. 保存转换结果
            self.conversion_results.append(result)
            
            # 6. 构建返回结果
            result_dict = {
                'success': True,
                'conversion_id': result.conversion_id,
                'graph_id': result.bipartite_graph.graph_id,
                'conversion_result': result,
                'graph_summary': {
                    'n_variable_nodes': len(result.bipartite_graph.variable_nodes),
                    'n_constraint_nodes': len(result.bipartite_graph.constraint_nodes),
                    'n_edges': len(result.bipartite_graph.edges),
                    'density': result.bipartite_graph.statistics.density
                },
                'timing': {
                    'total_duration': result.total_duration,
                    'extraction_duration': result.extraction_duration,
                    'building_duration': result.building_duration
                }
            }
            
            if validation_report:
                result_dict['validation_report'] = validation_report
            
            if saved_paths:
                result_dict['saved_files'] = saved_paths
            
            if visualization_paths:
                result_dict['visualizations'] = visualization_paths
            
            logger.info(f"✅ CVXPY问题转换成功: {problem_name}")
            logger.info(f"   图ID: {result.bipartite_graph.graph_id}")
            logger.info(f"   节点: {len(result.bipartite_graph.variable_nodes)} 变量, {len(result.bipartite_graph.constraint_nodes)} 约束")
            logger.info(f"   边数: {len(result.bipartite_graph.edges)}")
            
            return result_dict
            
        except Exception as e:
            self.system_statistics['total_conversions'] += 1
            self.system_statistics['failed_conversions'] += 1
            
            logger.error(f"CVXPY问题转换失败: {e}")
            return {
                'success': False,
                'error': str(e),
                'problem_name': problem_name
            }
    
    def convert_from_milp_instances(self,
                                  milp_instances: List[Any],
                                  save_graphs: bool = True,
                                  generate_visualizations: bool = False,
                                  validate_results: bool = True) -> Dict[str, Any]:
        """
        从MILP实例批量转换
        
        Args:
            milp_instances: MILP实例列表
            save_graphs: 是否保存图到文件
            generate_visualizations: 是否生成可视化
            validate_results: 是否验证结果
            
        Returns:
            批量转换结果
        """
        logger.info(f"开始批量转换 {len(milp_instances)} 个MILP实例")
        
        batch_start_time = datetime.now()
        successful_conversions = []
        failed_conversions = []
        
        for i, instance in enumerate(milp_instances):
            try:
                # 获取实例信息
                instance_id = getattr(instance, 'instance_id', f'milp_instance_{i:03d}')
                problem_name = f"milp_{instance_id}"
                
                logger.info(f"处理第 {i+1}/{len(milp_instances)} 个实例: {instance_id}")
                
                # 获取CVXPY问题
                if hasattr(instance, 'cvxpy_problem') and instance.cvxpy_problem is not None:
                    cvxpy_problem = instance.cvxpy_problem
                    
                    # 执行转换
                    conversion_result = self.convert_cvxpy_problem(
                        problem=cvxpy_problem,
                        problem_name=problem_name,
                        save_graph=save_graphs,
                        generate_visualizations=generate_visualizations,
                        validate_result=validate_results
                    )
                    
                    if conversion_result['success']:
                        # 添加MILP实例的元信息
                        conversion_result['milp_instance_info'] = {
                            'instance_id': instance_id,
                            'has_perturbation_config': hasattr(instance, 'perturbation_config'),
                            'has_statistics': hasattr(instance, 'statistics'),
                            'creation_time': getattr(instance, 'creation_time', None)
                        }
                        
                        # 将扰动配置信息添加到图中
                        if hasattr(instance, 'perturbation_config'):
                            conversion_result['conversion_result'].bipartite_graph.metadata['perturbation_config'] = instance.perturbation_config
                        
                        successful_conversions.append(conversion_result)
                    else:
                        failed_conversions.append({
                            'instance_id': instance_id,
                            'error': conversion_result.get('error', conversion_result.get('errors'))
                        })
                else:
                    failed_conversions.append({
                        'instance_id': instance_id,
                        'error': 'MILP实例缺少CVXPY问题对象'
                    })
                    
            except Exception as e:
                logger.error(f"处理第 {i+1} 个实例失败: {e}")
                failed_conversions.append({
                    'instance_index': i,
                    'error': str(e)
                })
        
        batch_duration = (datetime.now() - batch_start_time).total_seconds()
        
        # 生成批量转换报告
        batch_result = {
            'batch_info': {
                'total_instances': len(milp_instances),
                'successful_conversions': len(successful_conversions),
                'failed_conversions': len(failed_conversions),
                'success_rate': len(successful_conversions) / len(milp_instances),
                'batch_duration': batch_duration,
                'average_conversion_time': batch_duration / len(milp_instances) if milp_instances else 0
            },
            'successful_conversions': successful_conversions,
            'failed_conversions': failed_conversions,
            'system_statistics': self.system_statistics.copy()
        }
        
        # 保存批量结果报告
        batch_report_path = self.output_dir / f"batch_conversion_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        try:
            with open(batch_report_path, 'w', encoding='utf-8') as f:
                # 创建可序列化的报告副本
                serializable_report = self._make_report_serializable(batch_result)
                json.dump(serializable_report, f, indent=2, ensure_ascii=False, default=str)
            
            batch_result['report_path'] = str(batch_report_path)
            logger.info(f"批量转换报告已保存: {batch_report_path}")
        except Exception as e:
            logger.warning(f"保存批量报告失败: {e}")
        
        logger.info(f"✅ 批量转换完成!")
        logger.info(f"   成功: {len(successful_conversions)}/{len(milp_instances)}")
        logger.info(f"   耗时: {batch_duration:.2f} 秒")
        logger.info(f"   平均: {batch_duration/len(milp_instances):.3f} 秒/实例")
        
        return batch_result
    
    def analyze_conversion_results(self, 
                                 results: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        分析转换结果
        
        Args:
            results: 转换结果列表，如果为None则使用系统内部结果
            
        Returns:
            分析报告
        """
        if results is None:
            results = [{'conversion_result': r} for r in self.conversion_results]
        
        if not results:
            return {'message': '暂无转换结果可分析'}
        
        analysis = {
            'summary': {
                'total_results': len(results),
                'analysis_time': datetime.now().isoformat()
            },
            'graph_size_distribution': {},
            'performance_analysis': {},
            'feature_analysis': {},
            'error_analysis': {}
        }
        
        try:
            # 提取有效结果
            valid_results = [r for r in results if r.get('success', False)]
            
            if not valid_results:
                analysis['summary']['valid_results'] = 0
                return analysis
            
            analysis['summary']['valid_results'] = len(valid_results)
            
            # 图规模分析
            graph_sizes = []
            densities = []
            edge_counts = []
            
            for result in valid_results:
                if 'graph_summary' in result:
                    summary = result['graph_summary']
                    total_nodes = summary['n_variable_nodes'] + summary['n_constraint_nodes']
                    graph_sizes.append(total_nodes)
                    densities.append(summary['density'])
                    edge_counts.append(summary['n_edges'])
            
            if graph_sizes:
                analysis['graph_size_distribution'] = {
                    'total_nodes': {
                        'mean': float(np.mean(graph_sizes)),
                        'std': float(np.std(graph_sizes)),
                        'min': int(np.min(graph_sizes)),
                        'max': int(np.max(graph_sizes)),
                        'median': float(np.median(graph_sizes))
                    },
                    'density': {
                        'mean': float(np.mean(densities)),
                        'std': float(np.std(densities)),
                        'min': float(np.min(densities)),
                        'max': float(np.max(densities))
                    },
                    'edge_count': {
                        'mean': float(np.mean(edge_counts)),
                        'std': float(np.std(edge_counts)),
                        'min': int(np.min(edge_counts)),
                        'max': int(np.max(edge_counts))
                    }
                }
            
            # 性能分析
            durations = []
            extraction_times = []
            building_times = []
            
            for result in valid_results:
                if 'timing' in result:
                    timing = result['timing']
                    durations.append(timing['total_duration'])
                    extraction_times.append(timing['extraction_duration'])
                    building_times.append(timing['building_duration'])
            
            if durations:
                analysis['performance_analysis'] = {
                    'total_duration': {
                        'mean': float(np.mean(durations)),
                        'std': float(np.std(durations)),
                        'min': float(np.min(durations)),
                        'max': float(np.max(durations))
                    },
                    'extraction_time': {
                        'mean': float(np.mean(extraction_times)),
                        'std': float(np.std(extraction_times))
                    },
                    'building_time': {
                        'mean': float(np.mean(building_times)),
                        'std': float(np.std(building_times))
                    },
                    'throughput': {
                        'graphs_per_second': len(valid_results) / sum(durations) if sum(durations) > 0 else 0
                    }
                }
            
            # 验证分析
            validation_scores = []
            validation_statuses = []
            
            for result in valid_results:
                if 'validation_report' in result:
                    report = result['validation_report']
                    validation_scores.append(report.overall_score)
                    validation_statuses.append(report.overall_status)
            
            if validation_scores:
                status_counts = {}
                for status in validation_statuses:
                    status_counts[status] = status_counts.get(status, 0) + 1
                
                analysis['validation_analysis'] = {
                    'score_distribution': {
                        'mean': float(np.mean(validation_scores)),
                        'std': float(np.std(validation_scores)),
                        'min': float(np.min(validation_scores)),
                        'max': float(np.max(validation_scores))
                    },
                    'status_distribution': status_counts
                }
            
            # 错误分析
            failed_results = [r for r in results if not r.get('success', False)]
            if failed_results:
                error_types = {}
                for result in failed_results:
                    error = result.get('error', result.get('errors', ['未知错误']))
                    if isinstance(error, list):
                        error = error[0] if error else '未知错误'
                    
                    error_type = type(error).__name__ if hasattr(error, '__class__') else 'String'
                    error_types[error_type] = error_types.get(error_type, 0) + 1
                
                analysis['error_analysis'] = {
                    'total_failures': len(failed_results),
                    'failure_rate': len(failed_results) / len(results),
                    'error_type_distribution': error_types
                }
            
        except Exception as e:
            analysis['analysis_error'] = str(e)
            logger.error(f"结果分析失败: {e}")
        
        return analysis
    
    def _make_report_serializable(self, obj: Any) -> Any:
        """将报告对象转换为JSON可序列化格式"""
        if isinstance(obj, dict):
            return {key: self._make_report_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_report_serializable(item) for item in obj]
        elif hasattr(obj, 'to_dict'):
            return obj.to_dict()
        elif hasattr(obj, '__dict__'):
            return {key: self._make_report_serializable(value) 
                   for key, value in obj.__dict__.items() 
                   if not key.startswith('_')}
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, (int, float, str, bool)) or obj is None:
            return obj
        else:
            return str(obj)
    
    def get_system_report(self) -> Dict[str, Any]:
        """获取系统运行报告"""
        return {
            'system_info': {
                'output_directory': str(self.output_dir),
                'configuration': {
                    'use_sparse_matrix': self.config.use_sparse_matrix,
                    'normalize_coefficients': self.config.normalize_coefficients,
                    'validate_graph': self.config.validate_graph,
                    'save_intermediate_results': self.config.save_intermediate_results
                }
            },
            'statistics': self.system_statistics.copy(),
            'converter_performance': self.converter.get_performance_report(),
            'validator_summary': self.validator.get_validation_summary(),
            'recent_conversions': len(self.conversion_results)
        }
    
    def cleanup_temp_files(self, keep_days: int = 7) -> int:
        """
        清理临时文件
        
        Args:
            keep_days: 保留天数
            
        Returns:
            删除的文件数量
        """
        cutoff_date = datetime.now().timestamp() - (keep_days * 24 * 3600)
        deleted_count = 0
        
        try:
            for file_path in self.output_dir.rglob('*'):
                if file_path.is_file() and file_path.stat().st_mtime < cutoff_date:
                    if file_path.suffix in ['.pkl', '.json', '.png', '.html']:
                        file_path.unlink()
                        deleted_count += 1
                        
            logger.info(f"清理完成，删除 {deleted_count} 个文件")
            
        except Exception as e:
            logger.error(f"清理文件失败: {e}")
        
        return deleted_count


# 便捷函数
def convert_milp_to_bipartite_graph(milp_instance: Any,
                                  output_dir: str = "output/g2milp",
                                  save_graph: bool = True,
                                  generate_visualizations: bool = False) -> Dict[str, Any]:
    """
    转换单个MILP实例到二分图（便捷函数）
    
    Args:
        milp_instance: MILP实例对象
        output_dir: 输出目录
        save_graph: 是否保存图
        generate_visualizations: 是否生成可视化
        
    Returns:
        转换结果
    """
    system = G2MILPSystem(output_dir=output_dir)
    
    if hasattr(milp_instance, 'cvxpy_problem'):
        problem_name = getattr(milp_instance, 'instance_id', 'milp_instance')
        return system.convert_cvxpy_problem(
            problem=milp_instance.cvxpy_problem,
            problem_name=problem_name,
            save_graph=save_graph,
            generate_visualizations=generate_visualizations
        )
    else:
        return {
            'success': False,
            'error': 'MILP实例缺少CVXPY问题对象'
        }


def batch_convert_milp_instances(milp_instances: List[Any],
                                output_dir: str = "output/g2milp_batch",
                                config: ConversionConfig = None) -> Dict[str, Any]:
    """
    批量转换MILP实例（便捷函数）
    
    Args:
        milp_instances: MILP实例列表
        output_dir: 输出目录
        config: 转换配置
        
    Returns:
        批量转换结果
    """
    system = G2MILPSystem(output_dir=output_dir, config=config)
    return system.convert_from_milp_instances(
        milp_instances=milp_instances,
        save_graphs=True,
        generate_visualizations=False,
        validate_results=True
    )


if __name__ == "__main__":
    """测试G2MILP系统"""
    import sys
    from pathlib import Path
    
    # 添加项目路径
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))
    
    # 设置日志
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # 创建测试问题
    logger.info("创建测试G2MILP系统...")
    
    try:
        # 创建简单的CVXPY问题
        x = cp.Variable(3, name='x')
        y = cp.Variable(2, boolean=True, name='y')
        
        constraints = [
            x[0] + x[1] + x[2] <= 10,
            2*x[0] - x[1] == 5,
            x >= 0,
            y[0] + y[1] <= 1
        ]
        
        objective = cp.Minimize(3*x[0] + 2*x[1] + x[2] + 5*y[0] + 3*y[1])
        problem = cp.Problem(objective, constraints)
        
        # 初始化G2MILP系统
        system = G2MILPSystem(output_dir="test_output/g2milp")
        
        # 测试单个问题转换
        result = system.convert_cvxpy_problem(
            problem=problem,
            problem_name="测试问题",
            save_graph=True,
            generate_visualizations=True,
            validate_result=True
        )
        
        if result['success']:
            print("✅ G2MILP系统测试成功!")
            print(f"转换ID: {result['conversion_id']}")
            print(f"图摘要: {result['graph_summary']}")
            if 'validation_report' in result:
                print(f"验证状态: {result['validation_report'].overall_status}")
            print(f"保存的文件: {result.get('saved_files', {})}")
            
            # 测试分析功能
            analysis = system.analyze_conversion_results()
            print(f"\n分析结果:")
            print(f"  有效结果: {analysis['summary']['valid_results']}")
            if 'graph_size_distribution' in analysis:
                print(f"  平均图大小: {analysis['graph_size_distribution']['total_nodes']['mean']:.1f}")
            
            # 获取系统报告
            system_report = system.get_system_report()
            print(f"\n系统统计:")
            print(f"  总转换次数: {system_report['statistics']['total_conversions']}")
            print(f"  成功次数: {system_report['statistics']['successful_conversions']}")
            
        else:
            print(f"❌ 转换失败: {result.get('error', result.get('errors'))}")
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
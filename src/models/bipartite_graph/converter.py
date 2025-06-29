"""
G2MILP主转换器
整合CVXPY提取器和二分图构建器，提供完整的转换流程

主要功能:
1. 端到端的CVXPY问题到二分图转换
2. 批量处理多个MILP实例
3. 集成验证和错误处理
4. 性能监控和优化
5. 支持不同的转换配置
"""

import cvxpy as cp
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import json
import pickle

from .data_structures import BipartiteGraph, GraphStatistics
from .extractor import CVXPYToMILPExtractor, MILPStandardForm
from .builder import BipartiteGraphBuilder

logger = logging.getLogger(__name__)


@dataclass
class ConversionConfig:
    """转换配置"""
    # 提取器配置
    use_sparse_matrix: bool = True          # 使用稀疏矩阵
    extraction_tolerance: float = 1e-12     # 提取容差
    
    # 构建器配置  
    normalize_coefficients: bool = True      # 归一化系数
    sparse_threshold: float = 0.1           # 稀疏阈值
    batch_size: int = 1000                  # 批处理大小
    
    # 输出配置
    save_intermediate_results: bool = False  # 保存中间结果
    output_directory: Optional[str] = None   # 输出目录
    
    # 性能配置
    memory_limit_gb: float = 8.0            # 内存限制(GB)
    enable_parallel: bool = False           # 启用并行处理
    max_workers: int = 4                    # 最大工作线程数
    
    # 验证配置
    validate_graph: bool = True             # 验证图结构
    compute_statistics: bool = True         # 计算统计信息


@dataclass
class ConversionResult:
    """转换结果"""
    # 转换产物
    bipartite_graph: BipartiteGraph
    standard_form: MILPStandardForm
    
    # 转换统计
    conversion_id: str
    source_problem_name: str
    start_time: datetime
    end_time: datetime
    total_duration: float
    
    # 阶段耗时
    extraction_duration: float
    building_duration: float
    validation_duration: float = 0.0
    
    # 问题统计
    problem_statistics: Dict[str, Any] = field(default_factory=dict)
    
    # 转换配置
    config: ConversionConfig = None
    
    # 错误信息
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def get_summary(self) -> Dict[str, Any]:
        """获取转换摘要"""
        return {
            'conversion_id': self.conversion_id,
            'source_problem': self.source_problem_name,
            'success': len(self.errors) == 0,
            'total_duration': self.total_duration,
            'graph_statistics': {
                'n_variable_nodes': len(self.bipartite_graph.variable_nodes),
                'n_constraint_nodes': len(self.bipartite_graph.constraint_nodes),
                'n_edges': len(self.bipartite_graph.edges),
                'density': self.bipartite_graph.statistics.density
            },
            'problem_statistics': self.problem_statistics,
            'errors': self.errors,
            'warnings': self.warnings
        }


class G2MILPConverter:
    """
    G2MILP主转换器
    提供CVXPY问题到二分图的完整转换流程
    """
    
    def __init__(self, config: ConversionConfig = None):
        """
        初始化转换器
        
        Args:
            config: 转换配置对象
        """
        self.config = config or ConversionConfig()
        
        # 创建组件
        self.extractor = None
        self.builder = BipartiteGraphBuilder(
            normalize_coefficients=self.config.normalize_coefficients,
            sparse_threshold=self.config.sparse_threshold,
            batch_size=self.config.batch_size
        )
        
        # 转换历史
        self.conversion_history: List[ConversionResult] = []
        
        # 性能监控
        self.performance_stats: Dict[str, Any] = {
            'total_conversions': 0,
            'successful_conversions': 0,
            'failed_conversions': 0,
            'average_duration': 0.0,
            'memory_usage': []
        }
        
        # 设置输出目录
        if self.config.output_directory:
            self.output_dir = Path(self.config.output_directory)
            self.output_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.output_dir = None
        
        logger.info("G2MILP转换器初始化完成")
        logger.info(f"  配置: 稀疏矩阵={self.config.use_sparse_matrix}, 归一化={self.config.normalize_coefficients}")
        logger.info(f"  输出目录: {self.output_dir}")
    
    def convert_problem(self, 
                       problem: cp.Problem,
                       problem_name: str = None,
                       graph_id: str = None) -> ConversionResult:
        """
        转换单个CVXPY问题
        
        Args:
            problem: CVXPY问题对象
            problem_name: 问题名称
            graph_id: 图标识符
            
        Returns:
            转换结果对象
        """
        # 生成唯一标识
        conversion_id = f"conversion_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]}"
        if problem_name is None:
            problem_name = f"problem_{conversion_id}"
        if graph_id is None:
            graph_id = f"graph_{conversion_id}"
        
        logger.info("=" * 70)
        logger.info(f"开始G2MILP转换: {conversion_id}")
        logger.info(f"问题名称: {problem_name}")
        logger.info("=" * 70)
        
        start_time = datetime.now()
        result = ConversionResult(
            bipartite_graph=None,
            standard_form=None,
            conversion_id=conversion_id,
            source_problem_name=problem_name,
            start_time=start_time,
            end_time=None,
            total_duration=0.0,
            extraction_duration=0.0,
            building_duration=0.0,
            config=self.config
        )
        
        try:
            # 1. 提取MILP标准形式
            logger.info("步骤1: 提取MILP标准形式...")
            extract_start = datetime.now()
            
            self.extractor = CVXPYToMILPExtractor(problem, problem_name)
            standard_form = self.extractor.extract(
                use_sparse=self.config.use_sparse_matrix,
                tolerance=self.config.extraction_tolerance
            )
            
            result.standard_form = standard_form
            result.extraction_duration = (datetime.now() - extract_start).total_seconds()
            
            # 保存中间结果（如果需要）
            if self.config.save_intermediate_results and self.output_dir:
                self._save_standard_form(standard_form, conversion_id)
            
            # 2. 构建二分图
            logger.info("步骤2: 构建二分图...")
            build_start = datetime.now()
            
            bipartite_graph = self.builder.build_graph(standard_form, graph_id)
            result.bipartite_graph = bipartite_graph
            result.building_duration = (datetime.now() - build_start).total_seconds()
            
            # 3. 验证和统计（如果需要）
            if self.config.validate_graph or self.config.compute_statistics:
                logger.info("步骤3: 验证和统计...")
                validate_start = datetime.now()
                
                if self.config.validate_graph:
                    self._validate_conversion_result(result)
                
                if self.config.compute_statistics:
                    result.problem_statistics = self._compute_problem_statistics(result)
                
                result.validation_duration = (datetime.now() - validate_start).total_seconds()
            
            # 4. 完成转换
            result.end_time = datetime.now()
            result.total_duration = (result.end_time - start_time).total_seconds()
            
            # 更新性能统计
            self._update_performance_stats(result)
            
            # 保存结果（如果需要）
            if self.output_dir:
                self._save_conversion_result(result)
            
            # 添加到历史记录
            self.conversion_history.append(result)
            
            logger.info("=" * 70)
            logger.info("✅ G2MILP转换完成!")
            logger.info("=" * 70)
            logger.info(f"⏱️  总耗时: {result.total_duration:.3f} 秒")
            logger.info(f"   - 提取: {result.extraction_duration:.3f} 秒")
            logger.info(f"   - 构建: {result.building_duration:.3f} 秒")
            logger.info(f"   - 验证: {result.validation_duration:.3f} 秒")
            logger.info(f"📊 图统计:")
            logger.info(f"   - 变量节点: {len(bipartite_graph.variable_nodes)}")
            logger.info(f"   - 约束节点: {len(bipartite_graph.constraint_nodes)}")
            logger.info(f"   - 边数量: {len(bipartite_graph.edges)}")
            logger.info(f"   - 图密度: {bipartite_graph.statistics.density:.4f}")
            logger.info("=" * 70)
            
            return result
            
        except Exception as e:
            # 处理转换错误
            result.errors.append(str(e))
            result.end_time = datetime.now()
            result.total_duration = (result.end_time - start_time).total_seconds()
            
            logger.error(f"G2MILP转换失败: {e}")
            
            # 更新失败统计
            self.performance_stats['failed_conversions'] += 1
            self.conversion_history.append(result)
            
            raise
    
    def convert_batch(self, 
                     problems: List[cp.Problem],
                     problem_names: List[str] = None,
                     graph_ids: List[str] = None) -> List[ConversionResult]:
        """
        批量转换CVXPY问题
        
        Args:
            problems: CVXPY问题列表
            problem_names: 问题名称列表
            graph_ids: 图标识符列表
            
        Returns:
            转换结果列表
        """
        logger.info(f"开始批量G2MILP转换: {len(problems)} 个问题")
        
        # 准备名称和ID
        if problem_names is None:
            problem_names = [f"batch_problem_{i:03d}" for i in range(len(problems))]
        if graph_ids is None:
            graph_ids = [f"batch_graph_{i:03d}" for i in range(len(problems))]
        
        results = []
        
        for i, problem in enumerate(problems):
            try:
                logger.info(f"转换第 {i+1}/{len(problems)} 个问题...")
                
                result = self.convert_problem(
                    problem=problem,
                    problem_name=problem_names[i],
                    graph_id=graph_ids[i]
                )
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"第 {i+1} 个问题转换失败: {e}")
                continue
        
        logger.info(f"批量转换完成，成功转换 {len(results)} 个问题")
        return results
    
    def convert_from_milp_instances(self,
                                  milp_instances: List[Any],
                                  instance_id_field: str = 'instance_id') -> List[ConversionResult]:
        """
        从MILP实例对象批量转换
        
        Args:
            milp_instances: MILP实例列表（如BiasedMILPGenerator的输出）
            instance_id_field: 实例ID字段名
            
        Returns:
            转换结果列表
        """
        logger.info(f"从MILP实例批量转换: {len(milp_instances)} 个实例")
        
        results = []
        
        for i, instance in enumerate(milp_instances):
            try:
                # 获取CVXPY问题对象
                if hasattr(instance, 'cvxpy_problem') and instance.cvxpy_problem is not None:
                    problem = instance.cvxpy_problem
                    problem_name = getattr(instance, instance_id_field, f"milp_instance_{i:03d}")
                    graph_id = f"graph_from_{problem_name}"
                    
                    logger.info(f"转换MILP实例 {i+1}/{len(milp_instances)}: {problem_name}")
                    
                    result = self.convert_problem(
                        problem=problem,
                        problem_name=problem_name,
                        graph_id=graph_id
                    )
                    
                    # 添加MILP实例的元信息
                    if hasattr(instance, 'metadata'):
                        result.bipartite_graph.metadata['milp_instance'] = instance.metadata
                    if hasattr(instance, 'perturbation_config'):
                        result.bipartite_graph.metadata['perturbation_config'] = instance.perturbation_config
                    
                    results.append(result)
                    
                else:
                    logger.warning(f"第 {i+1} 个实例没有有效的CVXPY问题对象")
                    
            except Exception as e:
                logger.error(f"转换第 {i+1} 个MILP实例失败: {e}")
                continue
        
        logger.info(f"MILP实例批量转换完成，成功转换 {len(results)} 个实例")
        return results
    
    def _validate_conversion_result(self, result: ConversionResult):
        """验证转换结果"""
        # 基本完整性检查
        if result.standard_form is None:
            result.errors.append("标准形式对象为空")
            return
        
        if result.bipartite_graph is None:
            result.errors.append("二分图对象为空")
            return
        
        # 维度一致性检查
        if result.standard_form.n_variables != len(result.bipartite_graph.variable_nodes):
            result.errors.append("变量节点数量与标准形式不一致")
        
        if result.standard_form.n_constraints != len(result.bipartite_graph.constraint_nodes):
            result.errors.append("约束节点数量与标准形式不一致")
        
        # 边数量合理性检查
        expected_max_edges = result.standard_form.n_variables * result.standard_form.n_constraints
        actual_edges = len(result.bipartite_graph.edges)
        
        if actual_edges > expected_max_edges:
            result.errors.append(f"边数量异常: {actual_edges} > {expected_max_edges}")
        
        # 特征向量有效性检查
        invalid_features = 0
        for var_node in result.bipartite_graph.variable_nodes.values():
            try:
                features = var_node.get_feature_vector()
                if len(features) != 9:
                    invalid_features += 1
                elif np.any(np.isnan(features)) or np.any(np.isinf(features)):
                    invalid_features += 1
            except Exception:
                invalid_features += 1
        
        if invalid_features > 0:
            result.warnings.append(f"{invalid_features} 个变量节点的特征向量无效")
        
        logger.info(f"转换验证完成: {len(result.errors)} 个错误, {len(result.warnings)} 个警告")
    
    def _compute_problem_statistics(self, result: ConversionResult) -> Dict[str, Any]:
        """计算问题统计信息"""
        stats = {}
        
        if result.standard_form:
            stats['standard_form'] = {
                'n_variables': result.standard_form.n_variables,
                'n_constraints': result.standard_form.n_constraints,
                'objective_sense': result.standard_form.objective_sense,
                'nnz_objective': np.count_nonzero(result.standard_form.objective_coefficients),
                'variable_types': {
                    'continuous': sum(1 for vt in result.standard_form.variable_types if vt.value == 'continuous'),
                    'binary': sum(1 for vt in result.standard_form.variable_types if vt.value == 'binary'),
                    'integer': sum(1 for vt in result.standard_form.variable_types if vt.value == 'integer')
                }
            }
        
        if result.bipartite_graph:
            graph_stats = result.bipartite_graph.statistics
            stats['bipartite_graph'] = {
                'density': graph_stats.density,
                'avg_variable_degree': graph_stats.avg_variable_degree,
                'avg_constraint_degree': graph_stats.avg_constraint_degree,
                'max_variable_degree': graph_stats.max_variable_degree,
                'max_constraint_degree': graph_stats.max_constraint_degree,
                'coefficient_stats': graph_stats.coefficient_stats
            }
        
        return stats
    
    def _update_performance_stats(self, result: ConversionResult):
        """更新性能统计"""
        self.performance_stats['total_conversions'] += 1
        
        if len(result.errors) == 0:
            self.performance_stats['successful_conversions'] += 1
        else:
            self.performance_stats['failed_conversions'] += 1
        
        # 更新平均耗时
        total_successful = self.performance_stats['successful_conversions']
        if total_successful > 0:
            current_avg = self.performance_stats['average_duration']
            new_avg = ((current_avg * (total_successful - 1)) + result.total_duration) / total_successful
            self.performance_stats['average_duration'] = new_avg
    
    def _save_standard_form(self, standard_form: MILPStandardForm, conversion_id: str):
        """保存MILP标准形式"""
        if self.output_dir is None:
            return
        
        filepath = self.output_dir / f"{conversion_id}_standard_form.pkl"
        
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(standard_form, f)
            logger.debug(f"标准形式已保存: {filepath}")
        except Exception as e:
            logger.warning(f"保存标准形式失败: {e}")
    
    def _save_conversion_result(self, result: ConversionResult):
        """保存转换结果"""
        if self.output_dir is None:
            return
        
        # 保存二分图
        graph_filepath = self.output_dir / f"{result.conversion_id}_bipartite_graph.pkl"
        try:
            with open(graph_filepath, 'wb') as f:
                pickle.dump(result.bipartite_graph, f)
        except Exception as e:
            logger.warning(f"保存二分图失败: {e}")
        
        # 保存转换摘要
        summary_filepath = self.output_dir / f"{result.conversion_id}_summary.json"
        try:
            with open(summary_filepath, 'w', encoding='utf-8') as f:
                json.dump(result.get_summary(), f, indent=2, ensure_ascii=False, default=str)
        except Exception as e:
            logger.warning(f"保存转换摘要失败: {e}")
    
    def get_performance_report(self) -> Dict[str, Any]:
        """获取性能报告"""
        return {
            'performance_statistics': self.performance_stats.copy(),
            'conversion_history_count': len(self.conversion_history),
            'recent_conversions': [
                result.get_summary() for result in self.conversion_history[-10:]
            ]
        }
    
    def export_batch_results(self, 
                           results: List[ConversionResult],
                           export_path: str = None) -> str:
        """
        导出批量转换结果
        
        Args:
            results: 转换结果列表
            export_path: 导出路径
            
        Returns:
            导出文件路径
        """
        if export_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            export_path = f"g2milp_batch_results_{timestamp}.json"
        
        export_data = {
            'export_info': {
                'timestamp': datetime.now().isoformat(),
                'total_results': len(results),
                'successful_results': len([r for r in results if len(r.errors) == 0])
            },
            'results': [result.get_summary() for result in results],
            'performance_report': self.get_performance_report()
        }
        
        try:
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"批量结果已导出: {export_path}")
            return export_path
            
        except Exception as e:
            logger.error(f"导出批量结果失败: {e}")
            return ""


def convert_cvxpy_to_bipartite_graph(problem: cp.Problem,
                                   problem_name: str = None,
                                   config: ConversionConfig = None) -> BipartiteGraph:
    """
    便捷函数：将CVXPY问题转换为二分图
    
    Args:
        problem: CVXPY问题对象
        problem_name: 问题名称
        config: 转换配置
        
    Returns:
        二分图对象
    """
    converter = G2MILPConverter(config)
    result = converter.convert_problem(problem, problem_name)
    
    if result.errors:
        raise RuntimeError(f"转换失败: {result.errors}")
    
    return result.bipartite_graph


if __name__ == "__main__":
    """测试G2MILP转换器"""
    import sys
    from pathlib import Path
    
    # 添加项目路径
    project_root = Path(__file__).parent.parent.parent.parent
    sys.path.insert(0, str(project_root))
    
    # 创建测试问题
    logger.info("创建测试CVXPY问题...")
    
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
    
    try:
        # 创建转换配置
        config = ConversionConfig(
            use_sparse_matrix=True,
            normalize_coefficients=True,
            validate_graph=True,
            compute_statistics=True
        )
        
        # 测试单个问题转换
        converter = G2MILPConverter(config)
        result = converter.convert_problem(problem, "测试问题")
        
        print("✅ G2MILP转换测试成功!")
        print(f"转换ID: {result.conversion_id}")
        print(f"总耗时: {result.total_duration:.3f} 秒")
        print("\n" + result.bipartite_graph.summary())
        
        # 测试批量转换
        problems = [problem] * 3
        batch_results = converter.convert_batch(problems)
        
        print(f"\n✅ 批量转换测试成功: {len(batch_results)} 个结果")
        
        # 获取性能报告
        performance_report = converter.get_performance_report()
        print(f"\n性能报告:")
        print(f"  总转换次数: {performance_report['performance_statistics']['total_conversions']}")
        print(f"  成功次数: {performance_report['performance_statistics']['successful_conversions']}")
        print(f"  平均耗时: {performance_report['performance_statistics']['average_duration']:.3f} 秒")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
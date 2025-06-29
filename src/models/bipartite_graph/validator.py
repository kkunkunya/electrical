"""
二分图验证工具
提供二分图结构、数据完整性和正确性验证

验证功能:
1. 图结构完整性验证
2. 特征向量有效性检查
3. 数学约束一致性验证
4. 数据类型和范围检查
5. 性能指标计算
6. 错误诊断和修复建议
"""

import numpy as np
import logging
from typing import Dict, List, Set, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
import warnings

from .data_structures import (
    BipartiteGraph, VariableNode, ConstraintNode, BipartiteEdge,
    VariableType, ConstraintType
)

logger = logging.getLogger(__name__)


@dataclass
class ValidationRule:
    """验证规则"""
    rule_id: str
    rule_name: str
    rule_description: str
    severity: str  # 'error', 'warning', 'info'
    check_function: callable
    fix_function: Optional[callable] = None


@dataclass
class ValidationIssue:
    """验证问题"""
    issue_id: str
    rule_id: str
    severity: str
    component_type: str  # 'graph', 'variable_node', 'constraint_node', 'edge'
    component_id: Optional[str]
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    suggested_fix: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass 
class ValidationReport:
    """验证报告"""
    graph_id: str
    validation_time: datetime
    total_checks: int
    passed_checks: int
    failed_checks: int
    warnings: int
    errors: int
    
    # 问题列表
    issues: List[ValidationIssue] = field(default_factory=list)
    
    # 统计信息
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    
    # 整体评估
    overall_score: float = 0.0
    overall_status: str = "UNKNOWN"  # 'PASSED', 'WARNING', 'FAILED'
    
    def get_summary(self) -> Dict[str, Any]:
        """获取验证摘要"""
        return {
            'graph_id': self.graph_id,
            'validation_time': self.validation_time.isoformat(),
            'overall_status': self.overall_status,
            'overall_score': self.overall_score,
            'total_checks': self.total_checks,
            'passed_checks': self.passed_checks,
            'issues_summary': {
                'errors': self.errors,
                'warnings': self.warnings,
                'total': len(self.issues)
            },
            'performance_metrics': self.performance_metrics
        }


class BipartiteGraphValidator:
    """
    二分图验证器
    提供全面的图结构和数据验证功能
    """
    
    def __init__(self, strict_mode: bool = False):
        """
        初始化验证器
        
        Args:
            strict_mode: 是否使用严格模式（更严格的验证标准）
        """
        self.strict_mode = strict_mode
        
        # 注册验证规则
        self.validation_rules: Dict[str, ValidationRule] = {}
        self._register_default_rules()
        
        # 验证历史
        self.validation_history: List[ValidationReport] = []
        
        logger.info(f"二分图验证器初始化完成，严格模式: {strict_mode}")
        logger.info(f"注册验证规则: {len(self.validation_rules)} 个")
    
    def validate_graph(self, graph: BipartiteGraph) -> ValidationReport:
        """
        验证二分图
        
        Args:
            graph: 二分图对象
            
        Returns:
            验证报告
        """
        logger.info("=" * 60)
        logger.info(f"开始验证二分图: {graph.graph_id}")
        logger.info("=" * 60)
        
        start_time = datetime.now()
        
        # 创建验证报告
        report = ValidationReport(
            graph_id=graph.graph_id,
            validation_time=start_time,
            total_checks=len(self.validation_rules),
            passed_checks=0,
            failed_checks=0,
            warnings=0,
            errors=0
        )
        
        try:
            # 执行所有验证规则
            for rule_id, rule in self.validation_rules.items():
                try:
                    logger.debug(f"执行验证规则: {rule.rule_name}")
                    
                    # 调用验证函数
                    issues = rule.check_function(graph)
                    
                    if issues:
                        # 有问题，记录到报告中
                        for issue in issues:
                            issue.rule_id = rule_id
                            report.issues.append(issue)
                            
                            if issue.severity == 'error':
                                report.errors += 1
                            elif issue.severity == 'warning':
                                report.warnings += 1
                        
                        report.failed_checks += 1
                    else:
                        # 没问题，通过检查
                        report.passed_checks += 1
                        
                except Exception as e:
                    # 验证规则执行失败
                    error_issue = ValidationIssue(
                        issue_id=f"rule_error_{rule_id}",
                        rule_id=rule_id,
                        severity='error',
                        component_type='graph',
                        component_id=graph.graph_id,
                        message=f"验证规则 {rule.rule_name} 执行失败: {e}",
                        details={'exception': str(e)}
                    )
                    report.issues.append(error_issue)
                    report.errors += 1
                    report.failed_checks += 1
                    
                    logger.error(f"验证规则 {rule_id} 执行失败: {e}")
            
            # 计算性能指标
            report.performance_metrics = self._compute_performance_metrics(graph)
            
            # 计算整体评分和状态
            self._compute_overall_assessment(report)
            
            # 记录验证历史
            self.validation_history.append(report)
            
            validation_duration = (datetime.now() - start_time).total_seconds()
            
            logger.info("=" * 60)
            logger.info("✅ 二分图验证完成!")
            logger.info("=" * 60)
            logger.info(f"⏱️  验证耗时: {validation_duration:.3f} 秒")
            logger.info(f"📊 总检查项: {report.total_checks}")
            logger.info(f"✅ 通过检查: {report.passed_checks}")
            logger.info(f"❌ 失败检查: {report.failed_checks}")
            logger.info(f"⚠️  警告数量: {report.warnings}")
            logger.info(f"🚨 错误数量: {report.errors}")
            logger.info(f"🏆 整体状态: {report.overall_status}")
            logger.info(f"📈 整体评分: {report.overall_score:.2f}")
            logger.info("=" * 60)
            
            return report
            
        except Exception as e:
            logger.error(f"验证过程出现异常: {e}")
            
            # 创建异常报告
            report.errors += 1
            report.issues.append(ValidationIssue(
                issue_id="validation_exception",
                rule_id="system",
                severity='error',
                component_type='graph',
                component_id=graph.graph_id,
                message=f"验证过程异常: {e}",
                details={'exception': str(e)}
            ))
            
            report.overall_status = "FAILED"
            return report
    
    def _register_default_rules(self):
        """注册默认验证规则"""
        
        # 1. 图结构完整性规则
        self.validation_rules['graph_structure'] = ValidationRule(
            rule_id='graph_structure',
            rule_name='图结构完整性',
            rule_description='检查图的基本结构是否完整和一致',
            severity='error',
            check_function=self._check_graph_structure
        )
        
        # 2. 节点有效性规则
        self.validation_rules['node_validity'] = ValidationRule(
            rule_id='node_validity',
            rule_name='节点有效性',
            rule_description='检查所有节点的数据有效性',
            severity='error',
            check_function=self._check_node_validity
        )
        
        # 3. 边有效性规则
        self.validation_rules['edge_validity'] = ValidationRule(
            rule_id='edge_validity',
            rule_name='边有效性',
            rule_description='检查所有边的数据有效性',
            severity='error',
            check_function=self._check_edge_validity
        )
        
        # 4. 特征向量规则
        self.validation_rules['feature_vectors'] = ValidationRule(
            rule_id='feature_vectors',
            rule_name='特征向量有效性',
            rule_description='检查节点特征向量的数学有效性',
            severity='warning',
            check_function=self._check_feature_vectors
        )
        
        # 5. 数据一致性规则
        self.validation_rules['data_consistency'] = ValidationRule(
            rule_id='data_consistency',
            rule_name='数据一致性',
            rule_description='检查图中数据的内部一致性',
            severity='warning',
            check_function=self._check_data_consistency
        )
        
        # 6. 邻接信息规则
        self.validation_rules['adjacency_consistency'] = ValidationRule(
            rule_id='adjacency_consistency',
            rule_name='邻接信息一致性',
            rule_description='检查邻接信息与边的一致性',
            severity='error',
            check_function=self._check_adjacency_consistency
        )
        
        # 7. 统计信息规则
        self.validation_rules['statistics_accuracy'] = ValidationRule(
            rule_id='statistics_accuracy',
            rule_name='统计信息准确性',
            rule_description='检查图统计信息的准确性',
            severity='warning',
            check_function=self._check_statistics_accuracy
        )
        
        # 8. 性能指标规则
        self.validation_rules['performance_metrics'] = ValidationRule(
            rule_id='performance_metrics',
            rule_name='性能指标',
            rule_description='评估图的性能相关指标',
            severity='info',
            check_function=self._check_performance_metrics
        )
    
    def _check_graph_structure(self, graph: BipartiteGraph) -> List[ValidationIssue]:
        """检查图结构完整性"""
        issues = []
        
        # 检查基本属性
        if not graph.graph_id:
            issues.append(ValidationIssue(
                issue_id="missing_graph_id",
                rule_id="graph_structure",
                severity='error',
                component_type='graph',
                component_id=None,
                message="图缺少有效的graph_id"
            ))
        
        if not graph.source_problem_id:
            issues.append(ValidationIssue(
                issue_id="missing_source_problem_id",
                rule_id="graph_structure",
                severity='warning',
                component_type='graph',
                component_id=graph.graph_id,
                message="图缺少source_problem_id"
            ))
        
        # 检查节点集合
        if not graph.variable_nodes:
            issues.append(ValidationIssue(
                issue_id="empty_variable_nodes",
                rule_id="graph_structure",
                severity='error',
                component_type='graph',
                component_id=graph.graph_id,
                message="图没有变量节点"
            ))
        
        if not graph.constraint_nodes:
            issues.append(ValidationIssue(
                issue_id="empty_constraint_nodes",
                rule_id="graph_structure",
                severity='error',
                component_type='graph',
                component_id=graph.graph_id,
                message="图没有约束节点"
            ))
        
        # 检查边集合
        if not graph.edges and graph.variable_nodes and graph.constraint_nodes:
            issues.append(ValidationIssue(
                issue_id="no_edges",
                rule_id="graph_structure",
                severity='warning',
                component_type='graph',
                component_id=graph.graph_id,
                message="图有节点但没有边"
            ))
        
        return issues
    
    def _check_node_validity(self, graph: BipartiteGraph) -> List[ValidationIssue]:
        """检查节点有效性"""
        issues = []
        
        # 检查变量节点
        for var_id, var_node in graph.variable_nodes.items():
            # 检查基本属性
            if not var_node.node_id:
                issues.append(ValidationIssue(
                    issue_id=f"var_missing_id_{var_id}",
                    rule_id="node_validity",
                    severity='error',
                    component_type='variable_node',
                    component_id=var_id,
                    message="变量节点缺少node_id"
                ))
            
            if var_node.node_id != var_id:
                issues.append(ValidationIssue(
                    issue_id=f"var_id_mismatch_{var_id}",
                    rule_id="node_validity",
                    severity='error',
                    component_type='variable_node',
                    component_id=var_id,
                    message=f"变量节点ID不匹配: {var_node.node_id} != {var_id}"
                ))
            
            # 检查数值有效性
            if not np.isfinite(var_node.obj_coeff):
                issues.append(ValidationIssue(
                    issue_id=f"var_invalid_obj_coeff_{var_id}",
                    rule_id="node_validity",
                    severity='warning',
                    component_type='variable_node',
                    component_id=var_id,
                    message=f"变量节点目标系数无效: {var_node.obj_coeff}"
                ))
            
            # 检查边界约束
            if var_node.has_lower_bound and var_node.has_upper_bound:
                if var_node.lower_bound > var_node.upper_bound:
                    issues.append(ValidationIssue(
                        issue_id=f"var_invalid_bounds_{var_id}",
                        rule_id="node_validity",
                        severity='error',
                        component_type='variable_node',
                        component_id=var_id,
                        message=f"变量下界大于上界: {var_node.lower_bound} > {var_node.upper_bound}"
                    ))
            
            # 检查二进制变量的边界
            if var_node.var_type == VariableType.BINARY:
                if var_node.has_lower_bound and var_node.lower_bound < 0:
                    issues.append(ValidationIssue(
                        issue_id=f"binary_var_invalid_lower_{var_id}",
                        rule_id="node_validity",
                        severity='warning',
                        component_type='variable_node',
                        component_id=var_id,
                        message=f"二进制变量下界应≥0: {var_node.lower_bound}"
                    ))
                
                if var_node.has_upper_bound and var_node.upper_bound > 1:
                    issues.append(ValidationIssue(
                        issue_id=f"binary_var_invalid_upper_{var_id}",
                        rule_id="node_validity",
                        severity='warning',
                        component_type='variable_node',
                        component_id=var_id,
                        message=f"二进制变量上界应≤1: {var_node.upper_bound}"
                    ))
        
        # 检查约束节点
        for cons_id, cons_node in graph.constraint_nodes.items():
            # 检查基本属性
            if not cons_node.node_id:
                issues.append(ValidationIssue(
                    issue_id=f"cons_missing_id_{cons_id}",
                    rule_id="node_validity",
                    severity='error',
                    component_type='constraint_node',
                    component_id=cons_id,
                    message="约束节点缺少node_id"
                ))
            
            if cons_node.node_id != cons_id:
                issues.append(ValidationIssue(
                    issue_id=f"cons_id_mismatch_{cons_id}",
                    rule_id="node_validity",
                    severity='error',
                    component_type='constraint_node',
                    component_id=cons_id,
                    message=f"约束节点ID不匹配: {cons_node.node_id} != {cons_id}"
                ))
            
            # 检查约束方向
            if cons_node.sense not in ['==', '<=', '>=', 'soc']:
                issues.append(ValidationIssue(
                    issue_id=f"cons_invalid_sense_{cons_id}",
                    rule_id="node_validity",
                    severity='warning',
                    component_type='constraint_node',
                    component_id=cons_id,
                    message=f"约束方向无效: {cons_node.sense}"
                ))
            
            # 检查RHS值
            if not np.isfinite(cons_node.rhs_value):
                issues.append(ValidationIssue(
                    issue_id=f"cons_invalid_rhs_{cons_id}",
                    rule_id="node_validity",
                    severity='warning',
                    component_type='constraint_node',
                    component_id=cons_id,
                    message=f"约束RHS值无效: {cons_node.rhs_value}"
                ))
        
        return issues
    
    def _check_edge_validity(self, graph: BipartiteGraph) -> List[ValidationIssue]:
        """检查边有效性"""
        issues = []
        
        for edge_id, edge in graph.edges.items():
            # 检查边ID
            if not edge.edge_id:
                issues.append(ValidationIssue(
                    issue_id=f"edge_missing_id_{edge_id}",
                    rule_id="edge_validity",
                    severity='error',
                    component_type='edge',
                    component_id=edge_id,
                    message="边缺少edge_id"
                ))
            
            # 检查节点引用
            if edge.variable_node_id not in graph.variable_nodes:
                issues.append(ValidationIssue(
                    issue_id=f"edge_invalid_var_ref_{edge_id}",
                    rule_id="edge_validity",
                    severity='error',
                    component_type='edge',
                    component_id=edge_id,
                    message=f"边引用不存在的变量节点: {edge.variable_node_id}"
                ))
            
            if edge.constraint_node_id not in graph.constraint_nodes:
                issues.append(ValidationIssue(
                    issue_id=f"edge_invalid_cons_ref_{edge_id}",
                    rule_id="edge_validity",
                    severity='error',
                    component_type='edge',
                    component_id=edge_id,
                    message=f"边引用不存在的约束节点: {edge.constraint_node_id}"
                ))
            
            # 检查系数有效性
            if not np.isfinite(edge.coefficient):
                issues.append(ValidationIssue(
                    issue_id=f"edge_invalid_coeff_{edge_id}",
                    rule_id="edge_validity",
                    severity='warning',
                    component_type='edge',
                    component_id=edge_id,
                    message=f"边系数无效: {edge.coefficient}"
                ))
            
            # 检查计算属性一致性
            if abs(edge.abs_coefficient - abs(edge.coefficient)) > 1e-12:
                issues.append(ValidationIssue(
                    issue_id=f"edge_abs_coeff_inconsistent_{edge_id}",
                    rule_id="edge_validity",
                    severity='warning',
                    component_type='edge',
                    component_id=edge_id,
                    message="边的绝对值系数与计算值不一致"
                ))
            
            # 检查非零标记
            expected_nonzero = abs(edge.coefficient) > 1e-12
            if edge.is_nonzero != expected_nonzero:
                issues.append(ValidationIssue(
                    issue_id=f"edge_nonzero_flag_wrong_{edge_id}",
                    rule_id="edge_validity",
                    severity='warning',
                    component_type='edge',
                    component_id=edge_id,
                    message="边的非零标记错误"
                ))
        
        return issues
    
    def _check_feature_vectors(self, graph: BipartiteGraph) -> List[ValidationIssue]:
        """检查特征向量有效性"""
        issues = []
        
        # 检查变量节点特征向量
        for var_id, var_node in graph.variable_nodes.items():
            try:
                features = var_node.get_feature_vector()
                
                # 检查维度
                if len(features) != 9:
                    issues.append(ValidationIssue(
                        issue_id=f"var_feature_dim_wrong_{var_id}",
                        rule_id="feature_vectors",
                        severity='error',
                        component_type='variable_node',
                        component_id=var_id,
                        message=f"变量特征向量维度错误: {len(features)} != 9"
                    ))
                
                # 检查数值有效性
                if np.any(np.isnan(features)):
                    issues.append(ValidationIssue(
                        issue_id=f"var_feature_nan_{var_id}",
                        rule_id="feature_vectors",
                        severity='warning',
                        component_type='variable_node',
                        component_id=var_id,
                        message="变量特征向量包含NaN值"
                    ))
                
                if np.any(np.isinf(features)):
                    issues.append(ValidationIssue(
                        issue_id=f"var_feature_inf_{var_id}",
                        rule_id="feature_vectors",
                        severity='warning',
                        component_type='variable_node',
                        component_id=var_id,
                        message="变量特征向量包含无穷值"
                    ))
                
                # 检查二进制变量的布尔特征
                if len(features) >= 5:
                    has_lower = features[4]
                    has_upper = features[5]
                    if has_lower not in [0.0, 1.0] or has_upper not in [0.0, 1.0]:
                        issues.append(ValidationIssue(
                            issue_id=f"var_feature_bool_invalid_{var_id}",
                            rule_id="feature_vectors",
                            severity='warning',
                            component_type='variable_node',
                            component_id=var_id,
                            message="变量特征向量中的布尔值不标准"
                        ))
                
            except Exception as e:
                issues.append(ValidationIssue(
                    issue_id=f"var_feature_exception_{var_id}",
                    rule_id="feature_vectors",
                    severity='error',
                    component_type='variable_node',
                    component_id=var_id,
                    message=f"变量特征向量计算失败: {e}"
                ))
        
        # 检查约束节点特征向量
        for cons_id, cons_node in graph.constraint_nodes.items():
            try:
                features = cons_node.get_constraint_features()
                
                # 检查数值有效性
                if np.any(np.isnan(features)):
                    issues.append(ValidationIssue(
                        issue_id=f"cons_feature_nan_{cons_id}",
                        rule_id="feature_vectors",
                        severity='warning',
                        component_type='constraint_node',
                        component_id=cons_id,
                        message="约束特征向量包含NaN值"
                    ))
                
                if np.any(np.isinf(features)):
                    issues.append(ValidationIssue(
                        issue_id=f"cons_feature_inf_{cons_id}",
                        rule_id="feature_vectors",
                        severity='warning',
                        component_type='constraint_node',
                        component_id=cons_id,
                        message="约束特征向量包含无穷值"
                    ))
                
            except Exception as e:
                issues.append(ValidationIssue(
                    issue_id=f"cons_feature_exception_{cons_id}",
                    rule_id="feature_vectors",
                    severity='error',
                    component_type='constraint_node',
                    component_id=cons_id,
                    message=f"约束特征向量计算失败: {e}"
                ))
        
        return issues
    
    def _check_data_consistency(self, graph: BipartiteGraph) -> List[ValidationIssue]:
        """检查数据一致性"""
        issues = []
        
        # 检查度数一致性
        for var_id, var_node in graph.variable_nodes.items():
            expected_degree = len(graph.variable_to_constraints.get(var_id, set()))
            if var_node.degree != expected_degree:
                issues.append(ValidationIssue(
                    issue_id=f"var_degree_inconsistent_{var_id}",
                    rule_id="data_consistency",
                    severity='warning',
                    component_type='variable_node',
                    component_id=var_id,
                    message=f"变量度数不一致: {var_node.degree} != {expected_degree}"
                ))
        
        for cons_id, cons_node in graph.constraint_nodes.items():
            expected_degree = len(graph.constraint_to_variables.get(cons_id, set()))
            if cons_node.degree != expected_degree:
                issues.append(ValidationIssue(
                    issue_id=f"cons_degree_inconsistent_{cons_id}",
                    rule_id="data_consistency",
                    severity='warning',
                    component_type='constraint_node',
                    component_id=cons_id,
                    message=f"约束度数不一致: {cons_node.degree} != {expected_degree}"
                ))
        
        # 检查约束节点系数一致性
        for cons_id, cons_node in graph.constraint_nodes.items():
            # 从边中收集系数
            edge_coeffs = {}
            for edge in graph.edges.values():
                if edge.constraint_node_id == cons_id:
                    edge_coeffs[edge.variable_node_id] = edge.coefficient
            
            # 与约束节点存储的系数比较
            for var_id, coeff in cons_node.lhs_coefficients.items():
                if var_id not in edge_coeffs:
                    issues.append(ValidationIssue(
                        issue_id=f"cons_coeff_missing_edge_{cons_id}_{var_id}",
                        rule_id="data_consistency",
                        severity='warning',
                        component_type='constraint_node',
                        component_id=cons_id,
                        message=f"约束节点有系数但缺少对应的边: 变量 {var_id}"
                    ))
                elif abs(edge_coeffs[var_id] - coeff) > 1e-12:
                    issues.append(ValidationIssue(
                        issue_id=f"cons_coeff_mismatch_{cons_id}_{var_id}",
                        rule_id="data_consistency",
                        severity='warning',
                        component_type='constraint_node',
                        component_id=cons_id,
                        message=f"约束系数与边系数不匹配: {coeff} != {edge_coeffs[var_id]}"
                    ))
        
        return issues
    
    def _check_adjacency_consistency(self, graph: BipartiteGraph) -> List[ValidationIssue]:
        """检查邻接信息一致性"""
        issues = []
        
        # 从边重建邻接信息
        edges_var_to_cons = {}
        edges_cons_to_var = {}
        
        for edge in graph.edges.values():
            var_id = edge.variable_node_id
            cons_id = edge.constraint_node_id
            
            if var_id not in edges_var_to_cons:
                edges_var_to_cons[var_id] = set()
            edges_var_to_cons[var_id].add(cons_id)
            
            if cons_id not in edges_cons_to_var:
                edges_cons_to_var[cons_id] = set()
            edges_cons_to_var[cons_id].add(var_id)
        
        # 检查变量到约束的邻接
        for var_id in graph.variable_nodes:
            stored_neighbors = graph.variable_to_constraints.get(var_id, set())
            edge_neighbors = edges_var_to_cons.get(var_id, set())
            
            if stored_neighbors != edge_neighbors:
                issues.append(ValidationIssue(
                    issue_id=f"var_adjacency_mismatch_{var_id}",
                    rule_id="adjacency_consistency",
                    severity='error',
                    component_type='variable_node',
                    component_id=var_id,
                    message=f"变量邻接信息与边不一致"
                ))
        
        # 检查约束到变量的邻接
        for cons_id in graph.constraint_nodes:
            stored_neighbors = graph.constraint_to_variables.get(cons_id, set())
            edge_neighbors = edges_cons_to_var.get(cons_id, set())
            
            if stored_neighbors != edge_neighbors:
                issues.append(ValidationIssue(
                    issue_id=f"cons_adjacency_mismatch_{cons_id}",
                    rule_id="adjacency_consistency",
                    severity='error',
                    component_type='constraint_node',
                    component_id=cons_id,
                    message=f"约束邻接信息与边不一致"
                ))
        
        return issues
    
    def _check_statistics_accuracy(self, graph: BipartiteGraph) -> List[ValidationIssue]:
        """检查统计信息准确性"""
        issues = []
        
        stats = graph.statistics
        
        # 检查基本计数
        actual_var_nodes = len(graph.variable_nodes)
        actual_cons_nodes = len(graph.constraint_nodes)
        actual_edges = len(graph.edges)
        
        if stats.n_variable_nodes != actual_var_nodes:
            issues.append(ValidationIssue(
                issue_id="stats_var_count_wrong",
                rule_id="statistics_accuracy",
                severity='warning',
                component_type='graph',
                component_id=graph.graph_id,
                message=f"统计中的变量节点数错误: {stats.n_variable_nodes} != {actual_var_nodes}"
            ))
        
        if stats.n_constraint_nodes != actual_cons_nodes:
            issues.append(ValidationIssue(
                issue_id="stats_cons_count_wrong",
                rule_id="statistics_accuracy",
                severity='warning',
                component_type='graph',
                component_id=graph.graph_id,
                message=f"统计中的约束节点数错误: {stats.n_constraint_nodes} != {actual_cons_nodes}"
            ))
        
        if stats.n_edges != actual_edges:
            issues.append(ValidationIssue(
                issue_id="stats_edge_count_wrong",
                rule_id="statistics_accuracy",
                severity='warning',
                component_type='graph',
                component_id=graph.graph_id,
                message=f"统计中的边数错误: {stats.n_edges} != {actual_edges}"
            ))
        
        # 检查密度计算
        if actual_var_nodes > 0 and actual_cons_nodes > 0:
            expected_density = actual_edges / (actual_var_nodes * actual_cons_nodes)
            if abs(stats.density - expected_density) > 1e-6:
                issues.append(ValidationIssue(
                    issue_id="stats_density_wrong",
                    rule_id="statistics_accuracy",
                    severity='warning',
                    component_type='graph',
                    component_id=graph.graph_id,
                    message=f"统计中的密度计算错误: {stats.density} != {expected_density}"
                ))
        
        return issues
    
    def _check_performance_metrics(self, graph: BipartiteGraph) -> List[ValidationIssue]:
        """检查性能指标"""
        issues = []
        
        # 这里主要是信息性检查，不产生错误
        n_var = len(graph.variable_nodes)
        n_cons = len(graph.constraint_nodes)
        n_edges = len(graph.edges)
        
        # 大规模图警告
        if n_var > 10000 or n_cons > 10000:
            issues.append(ValidationIssue(
                issue_id="large_graph_warning",
                rule_id="performance_metrics",
                severity='info',
                component_type='graph',
                component_id=graph.graph_id,
                message=f"大规模图: {n_var} 变量, {n_cons} 约束，可能影响处理性能"
            ))
        
        # 高密度图警告
        if graph.statistics.density > 0.1:
            issues.append(ValidationIssue(
                issue_id="high_density_warning",
                rule_id="performance_metrics",
                severity='info',
                component_type='graph',
                component_id=graph.graph_id,
                message=f"高密度图: {graph.statistics.density:.4f}，内存使用可能较高"
            ))
        
        # 度数不平衡警告
        if graph.statistics.max_variable_degree > graph.statistics.avg_variable_degree * 10:
            issues.append(ValidationIssue(
                issue_id="variable_degree_imbalance",
                rule_id="performance_metrics",
                severity='info',
                component_type='graph',
                component_id=graph.graph_id,
                message="变量节点度数分布不平衡，存在高度数节点"
            ))
        
        return issues
    
    def _compute_performance_metrics(self, graph: BipartiteGraph) -> Dict[str, Any]:
        """计算性能指标"""
        metrics = {}
        
        try:
            # 基本规模指标
            metrics['graph_size'] = {
                'n_variables': len(graph.variable_nodes),
                'n_constraints': len(graph.constraint_nodes),
                'n_edges': len(graph.edges),
                'density': graph.statistics.density
            }
            
            # 度数分布指标
            var_degrees = [node.degree for node in graph.variable_nodes.values()]
            cons_degrees = [node.degree for node in graph.constraint_nodes.values()]
            
            if var_degrees:
                metrics['variable_degree_stats'] = {
                    'mean': float(np.mean(var_degrees)),
                    'std': float(np.std(var_degrees)),
                    'min': int(np.min(var_degrees)),
                    'max': int(np.max(var_degrees)),
                    'median': float(np.median(var_degrees))
                }
            
            if cons_degrees:
                metrics['constraint_degree_stats'] = {
                    'mean': float(np.mean(cons_degrees)),
                    'std': float(np.std(cons_degrees)),
                    'min': int(np.min(cons_degrees)),
                    'max': int(np.max(cons_degrees)),
                    'median': float(np.median(cons_degrees))
                }
            
            # 系数分布指标
            coefficients = [edge.coefficient for edge in graph.edges.values()]
            if coefficients:
                metrics['coefficient_stats'] = {
                    'mean': float(np.mean(coefficients)),
                    'std': float(np.std(coefficients)),
                    'min': float(np.min(coefficients)),
                    'max': float(np.max(coefficients)),
                    'mean_abs': float(np.mean(np.abs(coefficients))),
                    'nnz_ratio': float(np.count_nonzero(coefficients) / len(coefficients))
                }
            
            # 内存估算（粗略）
            estimated_memory_mb = (
                len(graph.variable_nodes) * 0.001 +  # 变量节点大约1KB
                len(graph.constraint_nodes) * 0.001 +  # 约束节点大约1KB  
                len(graph.edges) * 0.0005  # 边大约0.5KB
            )
            metrics['estimated_memory_mb'] = estimated_memory_mb
            
        except Exception as e:
            metrics['error'] = f"性能指标计算失败: {e}"
        
        return metrics
    
    def _compute_overall_assessment(self, report: ValidationReport):
        """计算整体评估"""
        # 计算评分
        if report.total_checks > 0:
            base_score = report.passed_checks / report.total_checks * 100
            
            # 根据错误和警告调整评分
            error_penalty = report.errors * 10
            warning_penalty = report.warnings * 2
            
            report.overall_score = max(0, base_score - error_penalty - warning_penalty)
        else:
            report.overall_score = 0
        
        # 确定状态
        if report.errors > 0:
            report.overall_status = "FAILED"
        elif report.warnings > 0:
            report.overall_status = "WARNING"
        else:
            report.overall_status = "PASSED"
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """获取验证历史摘要"""
        if not self.validation_history:
            return {"message": "暂无验证历史"}
        
        total_validations = len(self.validation_history)
        passed_validations = sum(1 for r in self.validation_history if r.overall_status == "PASSED")
        
        return {
            'total_validations': total_validations,
            'passed_validations': passed_validations,
            'success_rate': passed_validations / total_validations,
            'registered_rules': len(self.validation_rules),
            'recent_validations': [r.get_summary() for r in self.validation_history[-5:]]
        }


# 便捷函数
def validate_bipartite_graph(graph: BipartiteGraph, 
                           strict_mode: bool = False) -> ValidationReport:
    """
    验证二分图（便捷函数）
    
    Args:
        graph: 二分图对象
        strict_mode: 是否使用严格模式
        
    Returns:
        验证报告
    """
    validator = BipartiteGraphValidator(strict_mode=strict_mode)
    return validator.validate_graph(graph)


if __name__ == "__main__":
    """测试验证工具"""
    logger.info("二分图验证工具测试")
    print("✅ 验证工具模块加载成功!")
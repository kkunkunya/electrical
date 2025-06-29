"""
CVXPY到MILP标准形式提取器
从CVXPY问题对象中提取约束矩阵、变量信息和目标函数

主要功能:
1. 解析CVXPY问题结构
2. 提取约束矩阵和右侧向量
3. 识别变量类型和边界
4. 处理目标函数系数
5. 标准化约束形式
"""

import cvxpy as cp
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime
import scipy.sparse as sp

from .data_structures import VariableType, ConstraintType

logger = logging.getLogger(__name__)


@dataclass
class MILPStandardForm:
    """MILP标准形式数据结构"""
    # 约束矩阵和向量
    constraint_matrix: Union[np.ndarray, sp.csr_matrix]  # A矩阵
    rhs_vector: np.ndarray                               # b向量  
    constraint_senses: List[str]                         # 约束方向列表
    
    # 变量信息
    variable_info: List[Dict[str, Any]]                  # 变量详细信息
    variable_types: List[VariableType]                   # 变量类型
    lower_bounds: np.ndarray                             # 变量下界
    upper_bounds: np.ndarray                             # 变量上界
    
    # 目标函数
    objective_coefficients: np.ndarray                   # 目标函数系数
    objective_sense: str                                 # "minimize" 或 "maximize"
    objective_constant: float                            # 目标函数常数项
    
    # 约束信息
    constraint_info: List[Dict[str, Any]]                # 约束详细信息
    constraint_types: List[ConstraintType]               # 约束类型
    
    # 元信息
    problem_name: str = "unnamed_problem"
    n_variables: int = 0
    n_constraints: int = 0
    extraction_time: datetime = None
    source_cvxpy_problem: cp.Problem = None


class CVXPYToMILPExtractor:
    """
    CVXPY到MILP标准形式提取器
    负责从CVXPY问题对象中提取所有必要信息
    """
    
    def __init__(self, problem: cp.Problem, problem_name: str = None):
        """
        初始化提取器
        
        Args:
            problem: CVXPY问题对象
            problem_name: 问题名称
        """
        self.problem = problem
        self.problem_name = problem_name or f"cvxpy_problem_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # 提取结果
        self.standard_form: Optional[MILPStandardForm] = None
        self.extraction_log: List[str] = []
        
        logger.info(f"CVXPY提取器初始化完成: {self.problem_name}")
    
    def extract(self, use_sparse: bool = True, 
                tolerance: float = 1e-12) -> MILPStandardForm:
        """
        执行提取过程
        
        Args:
            use_sparse: 是否使用稀疏矩阵格式
            tolerance: 数值容差
            
        Returns:
            MILP标准形式对象
        """
        logger.info("=" * 60)
        logger.info("开始CVXPY到MILP标准形式提取")
        logger.info("=" * 60)
        
        start_time = datetime.now()
        
        try:
            # 1. 验证问题状态
            self._validate_problem()
            
            # 2. 获取问题的canonical形式
            logger.info("获取问题canonical形式...")
            data, chain, inverse_data = self.problem.get_problem_data(
                solver=cp.ECOS,  # 使用ECOS求解器的数据格式
                verbose=False
            )
            
            # 3. 提取约束信息
            logger.info("提取约束信息...")
            constraint_matrix, rhs_vector, constraint_senses, constraint_info = self._extract_constraints(
                data, use_sparse
            )
            
            # 4. 提取变量信息
            logger.info("提取变量信息...")
            variable_info, variable_types, lower_bounds, upper_bounds = self._extract_variables(
                data, chain
            )
            
            # 5. 提取目标函数
            logger.info("提取目标函数...")
            obj_coeffs, obj_sense, obj_constant = self._extract_objective(data)
            
            # 6. 分类约束类型
            logger.info("分类约束类型...")
            constraint_types = self._classify_constraint_types(constraint_info)
            
            # 7. 创建标准形式对象
            self.standard_form = MILPStandardForm(
                constraint_matrix=constraint_matrix,
                rhs_vector=rhs_vector,
                constraint_senses=constraint_senses,
                variable_info=variable_info,
                variable_types=variable_types,
                lower_bounds=lower_bounds,
                upper_bounds=upper_bounds,
                objective_coefficients=obj_coeffs,
                objective_sense=obj_sense,
                objective_constant=obj_constant,
                constraint_info=constraint_info,
                constraint_types=constraint_types,
                problem_name=self.problem_name,
                n_variables=len(variable_info),
                n_constraints=len(constraint_info),
                extraction_time=start_time,
                source_cvxpy_problem=self.problem
            )
            
            # 8. 验证提取结果
            self._validate_extraction()
            
            extraction_duration = (datetime.now() - start_time).total_seconds()
            
            logger.info("=" * 60)
            logger.info("✅ CVXPY提取完成!")
            logger.info("=" * 60)
            logger.info(f"⏱️  提取耗时: {extraction_duration:.3f} 秒")
            logger.info(f"📊 变量数量: {self.standard_form.n_variables}")
            logger.info(f"📊 约束数量: {self.standard_form.n_constraints}")
            logger.info(f"📊 矩阵密度: {self._compute_matrix_density():.4f}")
            logger.info("=" * 60)
            
            return self.standard_form
            
        except Exception as e:
            logger.error(f"CVXPY提取失败: {e}")
            self.extraction_log.append(f"提取失败: {e}")
            raise
    
    def _validate_problem(self):
        """验证CVXPY问题的有效性"""
        if self.problem is None:
            raise ValueError("CVXPY问题对象不能为空")
        
        if not hasattr(self.problem, 'variables') or not self.problem.variables():
            raise ValueError("CVXPY问题没有定义变量")
        
        if not hasattr(self.problem, 'constraints'):
            raise ValueError("CVXPY问题没有定义约束")
        
        logger.info(f"问题验证通过: {len(self.problem.variables())} 个变量, {len(self.problem.constraints)} 个约束")
    
    def _extract_constraints(self, data: Dict, use_sparse: bool) -> Tuple[
        Union[np.ndarray, sp.csr_matrix], np.ndarray, List[str], List[Dict[str, Any]]
    ]:
        """提取约束信息"""
        # ECOS数据格式解析
        # data包含: 'A', 'b', 'G', 'h', 'dims'等
        
        constraint_matrices = []
        rhs_vectors = []
        senses = []
        constraint_info = []
        
        # 处理等式约束 (Ax = b)
        if 'A' in data and data['A'] is not None:
            A_eq = data['A']
            b_eq = data['b']
            
            n_eq_constraints = A_eq.shape[0]
            constraint_matrices.append(A_eq)
            rhs_vectors.append(b_eq)
            senses.extend(['=='] * n_eq_constraints)
            
            for i in range(n_eq_constraints):
                constraint_info.append({
                    'constraint_id': f"eq_constraint_{i}",
                    'constraint_name': f"等式约束_{i}",
                    'type': 'equality',
                    'original_index': i,
                    'source': 'equality_constraints'
                })
            
            logger.info(f"提取等式约束: {n_eq_constraints} 个")
        
        # 处理不等式约束 (Gx <= h)
        if 'G' in data and data['G'] is not None:
            G_ineq = data['G']
            h_ineq = data['h']
            
            # 需要根据dims信息解析不同类型的约束
            dims = data.get('dims', {})
            
            # 线性不等式约束
            # ConeDims对象使用属性访问，不是字典
            if hasattr(dims, 'l'):
                n_linear_ineq = dims.l if dims.l is not None else 0
            else:
                # 如果dims是字典格式（向后兼容）
                n_linear_ineq = dims.get('l', 0) if isinstance(dims, dict) else 0
            if n_linear_ineq > 0:
                G_linear = G_ineq[:n_linear_ineq, :]
                h_linear = h_ineq[:n_linear_ineq]
                
                constraint_matrices.append(G_linear)
                rhs_vectors.append(h_linear)
                senses.extend(['<='] * n_linear_ineq)
                
                for i in range(n_linear_ineq):
                    constraint_info.append({
                        'constraint_id': f"ineq_constraint_{i}",
                        'constraint_name': f"不等式约束_{i}",
                        'type': 'inequality',
                        'original_index': i,
                        'source': 'linear_inequality_constraints'
                    })
                
                logger.info(f"提取线性不等式约束: {n_linear_ineq} 个")
            
            # SOC约束
            # ConeDims对象使用属性访问，不是字典
            if hasattr(dims, 'q'):
                soc_dims = dims.q if dims.q is not None else []
            else:
                # 如果dims是字典格式（向后兼容）
                soc_dims = dims.get('q', []) if isinstance(dims, dict) else []
            if soc_dims:
                offset = n_linear_ineq
                soc_constraint_idx = 0
                
                for soc_dim in soc_dims:
                    G_soc = G_ineq[offset:offset+soc_dim, :]
                    h_soc = h_ineq[offset:offset+soc_dim]
                    
                    # SOC约束通常表示为 ||Ax + b|| <= c
                    # 这里简化为不等式约束处理
                    constraint_matrices.append(G_soc)
                    rhs_vectors.append(h_soc)
                    senses.extend(['soc'] * soc_dim)  # 特殊标记
                    
                    for i in range(soc_dim):
                        constraint_info.append({
                            'constraint_id': f"soc_constraint_{soc_constraint_idx}_{i}",
                            'constraint_name': f"SOC约束_{soc_constraint_idx}_{i}",
                            'type': 'second_order_cone',
                            'original_index': offset + i,
                            'soc_group': soc_constraint_idx,
                            'source': 'soc_constraints'
                        })
                    
                    offset += soc_dim
                    soc_constraint_idx += 1
                
                logger.info(f"提取SOC约束: {len(soc_dims)} 组")
        
        # 合并所有约束矩阵
        if constraint_matrices:
            if use_sparse:
                # 确保所有矩阵都是稀疏格式
                sparse_matrices = []
                for mat in constraint_matrices:
                    if sp.issparse(mat):
                        sparse_matrices.append(mat)
                    else:
                        sparse_matrices.append(sp.csr_matrix(mat))
                
                constraint_matrix = sp.vstack(sparse_matrices)
            else:
                # 转换为稠密矩阵
                dense_matrices = []
                for mat in constraint_matrices:
                    if sp.issparse(mat):
                        dense_matrices.append(mat.toarray())
                    else:
                        dense_matrices.append(mat)
                
                constraint_matrix = np.vstack(dense_matrices)
            
            rhs_vector = np.concatenate(rhs_vectors)
        else:
            # 空约束情况
            n_vars = len(self.problem.variables()[0].flatten()) if self.problem.variables() else 0
            if use_sparse:
                constraint_matrix = sp.csr_matrix((0, n_vars))
            else:
                constraint_matrix = np.zeros((0, n_vars))
            rhs_vector = np.array([])
            senses = []
            constraint_info = []
        
        return constraint_matrix, rhs_vector, senses, constraint_info
    
    def _extract_variables(self, data: Dict, chain) -> Tuple[
        List[Dict[str, Any]], List[VariableType], np.ndarray, np.ndarray
    ]:
        """提取变量信息"""
        variable_info = []
        variable_types = []
        lower_bounds = []
        upper_bounds = []
        
        # 获取所有变量
        all_variables = self.problem.variables()
        
        # 构建变量到索引的映射
        var_offset = 0
        
        for var in all_variables:
            # 获取变量的基本属性
            var_size = var.size
            var_shape = var.shape
            var_name = var.name() if hasattr(var, 'name') and var.name() else f"var_{var.id}"
            
            # 确定变量类型
            if hasattr(var, 'attributes'):
                if var.attributes.get('boolean', False):
                    var_type = VariableType.BINARY
                elif var.attributes.get('integer', False):
                    var_type = VariableType.INTEGER
                else:
                    var_type = VariableType.CONTINUOUS
            else:
                var_type = VariableType.CONTINUOUS
            
            # 获取变量边界
            if hasattr(var, 'project'):
                # 尝试获取变量的边界信息
                lb = getattr(var, 'lower', None)
                ub = getattr(var, 'upper', None)
                
                if lb is not None:
                    if np.isscalar(lb):
                        var_lower_bounds = [lb] * var_size
                    else:
                        var_lower_bounds = lb.flatten() if hasattr(lb, 'flatten') else [lb] * var_size
                else:
                    var_lower_bounds = [-np.inf] * var_size
                
                if ub is not None:
                    if np.isscalar(ub):
                        var_upper_bounds = [ub] * var_size
                    else:
                        var_upper_bounds = ub.flatten() if hasattr(ub, 'flatten') else [ub] * var_size
                else:
                    var_upper_bounds = [np.inf] * var_size
            else:
                # 默认边界
                if var_type == VariableType.BINARY:
                    var_lower_bounds = [0.0] * var_size
                    var_upper_bounds = [1.0] * var_size
                else:
                    var_lower_bounds = [-np.inf] * var_size
                    var_upper_bounds = [np.inf] * var_size
            
            # 为每个标量变量创建信息条目
            for i in range(var_size):
                flat_index = var_offset + i
                
                # 计算在原始变量中的位置
                if len(var_shape) == 1:
                    original_index = (i,)
                elif len(var_shape) == 2:
                    original_index = (i // var_shape[1], i % var_shape[1])
                else:
                    original_index = np.unravel_index(i, var_shape)
                
                var_info = {
                    'variable_id': f"{var_name}_{i}",
                    'cvxpy_var_name': var_name,
                    'cvxpy_var_id': var.id,
                    'original_shape': var_shape,
                    'original_index': original_index,
                    'flat_index': flat_index,
                    'size': var_size,
                    'offset_in_problem': var_offset
                }
                
                variable_info.append(var_info)
                variable_types.append(var_type)
                lower_bounds.append(var_lower_bounds[i])
                upper_bounds.append(var_upper_bounds[i])
            
            var_offset += var_size
            
            logger.debug(f"处理变量 {var_name}: 形状{var_shape}, 大小{var_size}, 类型{var_type}")
        
        return variable_info, variable_types, np.array(lower_bounds), np.array(upper_bounds)
    
    def _extract_objective(self, data: Dict) -> Tuple[np.ndarray, str, float]:
        """提取目标函数信息"""
        # 从ECOS数据格式中提取目标函数
        obj_coeffs = data.get('c', None)
        
        if obj_coeffs is None:
            # 如果没有线性目标函数，创建零向量
            n_vars = self.standard_form.n_variables if self.standard_form else len(self.problem.variables()[0].flatten())
            obj_coeffs = np.zeros(n_vars)
        else:
            if sp.issparse(obj_coeffs):
                obj_coeffs = obj_coeffs.toarray().flatten()
            else:
                obj_coeffs = np.array(obj_coeffs).flatten()
        
        # 确定目标函数方向
        obj_sense = "minimize"  # CVXPY默认是最小化
        if hasattr(self.problem.objective, 'NAME') and 'MAXIMIZE' in self.problem.objective.NAME:
            obj_sense = "maximize"
        
        # 目标函数常数项
        obj_constant = 0.0
        if hasattr(self.problem.objective, 'args') and len(self.problem.objective.args) > 0:
            # 尝试提取常数项（复杂情况下可能需要更详细的分析）
            pass
        
        logger.info(f"目标函数: {obj_sense}, 非零系数: {np.count_nonzero(obj_coeffs)} / {len(obj_coeffs)}")
        
        return obj_coeffs, obj_sense, obj_constant
    
    def _classify_constraint_types(self, constraint_info: List[Dict[str, Any]]) -> List[ConstraintType]:
        """分类约束类型"""
        constraint_types = []
        
        for info in constraint_info:
            constraint_type = info.get('type', 'unknown')
            
            if constraint_type == 'equality':
                constraint_types.append(ConstraintType.LINEAR_EQ)
            elif constraint_type == 'inequality':
                constraint_types.append(ConstraintType.LINEAR_INEQ)
            elif constraint_type == 'second_order_cone':
                constraint_types.append(ConstraintType.SOC)
            elif constraint_type == 'quadratic':
                constraint_types.append(ConstraintType.QUADRATIC)
            else:
                constraint_types.append(ConstraintType.LINEAR_INEQ)  # 默认类型
        
        return constraint_types
    
    def _validate_extraction(self):
        """验证提取结果的有效性"""
        if self.standard_form is None:
            raise RuntimeError("标准形式对象未创建")
        
        # 检查维度一致性
        n_vars = self.standard_form.n_variables
        n_constraints = self.standard_form.n_constraints
        
        # 约束矩阵维度检查
        if sp.issparse(self.standard_form.constraint_matrix):
            matrix_shape = self.standard_form.constraint_matrix.shape
        else:
            matrix_shape = self.standard_form.constraint_matrix.shape
        
        # CVXPY在canonical形式中可能会添加辅助变量，所以约束矩阵列数可能大于原始变量数
        # 这是正常现象，我们需要调整变量数量以匹配约束矩阵
        if matrix_shape[1] != n_vars:
            logger.warning(f"约束矩阵列数 ({matrix_shape[1]}) 与原始变量数 ({n_vars}) 不匹配")
            logger.info(f"CVXPY添加了 {matrix_shape[1] - n_vars} 个辅助变量")
            
            # 更新变量数量以匹配约束矩阵
            actual_n_vars = matrix_shape[1]
            
            # 扩展目标函数系数向量（新变量的系数设为0）
            if len(self.standard_form.objective_coefficients) < actual_n_vars:
                additional_coeffs = actual_n_vars - len(self.standard_form.objective_coefficients)
                extended_obj_coeffs = np.zeros(actual_n_vars)
                extended_obj_coeffs[:len(self.standard_form.objective_coefficients)] = self.standard_form.objective_coefficients
                self.standard_form.objective_coefficients = extended_obj_coeffs
                logger.info(f"目标函数系数向量已扩展，添加了 {additional_coeffs} 个零系数")
            
            # 扩展变量边界向量
            if len(self.standard_form.lower_bounds) < actual_n_vars:
                additional_bounds = actual_n_vars - len(self.standard_form.lower_bounds)
                extended_lower = np.full(actual_n_vars, -np.inf)
                extended_lower[:len(self.standard_form.lower_bounds)] = self.standard_form.lower_bounds
                self.standard_form.lower_bounds = extended_lower
                
                extended_upper = np.full(actual_n_vars, np.inf)
                extended_upper[:len(self.standard_form.upper_bounds)] = self.standard_form.upper_bounds
                self.standard_form.upper_bounds = extended_upper
                logger.info(f"变量边界向量已扩展，添加了 {additional_bounds} 个无界变量")
            
            # 扩展变量类型向量（新变量默认为连续变量）
            if len(self.standard_form.variable_types) < actual_n_vars:
                additional_types = actual_n_vars - len(self.standard_form.variable_types)
                extended_types = np.zeros(actual_n_vars, dtype=int)  # 0表示连续变量
                
                # 转换VariableType枚举为整数值
                original_types = self.standard_form.variable_types
                
                # 变量类型映射：字符串 -> 整数
                type_mapping = {
                    'continuous': 0,
                    'binary': 1, 
                    'integer': 2,
                    'boolean': 1,  # boolean等同于binary
                    0: 0,  # 连续
                    1: 1,  # 二进制
                    2: 2   # 整数
                }
                
                for i, var_type in enumerate(original_types):
                    if hasattr(var_type, 'value'):
                        # VariableType枚举类型，获取其值
                        type_value = var_type.value
                        if isinstance(type_value, str):
                            extended_types[i] = type_mapping.get(type_value.lower(), 0)
                        else:
                            extended_types[i] = type_mapping.get(type_value, 0)
                    elif isinstance(var_type, str):
                        # 字符串类型
                        extended_types[i] = type_mapping.get(var_type.lower(), 0)
                    else:
                        # 数值类型
                        extended_types[i] = type_mapping.get(var_type, 0)
                
                self.standard_form.variable_types = extended_types
                logger.info(f"变量类型向量已扩展，添加了 {additional_types} 个连续变量")
            
            # 扩展变量信息列表（为辅助变量添加默认信息）
            if len(self.standard_form.variable_info) < actual_n_vars:
                additional_vars = actual_n_vars - len(self.standard_form.variable_info)
                for i in range(additional_vars):
                    aux_var_index = len(self.standard_form.variable_info) + i
                    aux_var_info = {
                        'variable_id': f'aux_var_{aux_var_index}',
                        'cvxpy_var_name': f'aux_var_{aux_var_index}',
                        'cvxpy_var_id': f'aux_{aux_var_index}',
                        'original_shape': (1,),
                        'original_index': (0,),
                        'flat_index': aux_var_index,
                        'size': 1,
                        'offset_in_problem': aux_var_index
                    }
                    self.standard_form.variable_info.append(aux_var_info)
                logger.info(f"变量信息列表已扩展，添加了 {additional_vars} 个辅助变量信息")
            
            # 更新变量计数
            self.standard_form.n_variables = actual_n_vars
            logger.info(f"变量数量已更新: {n_vars} -> {actual_n_vars}")
            
            # 重新获取更新后的变量数
            n_vars = actual_n_vars
        
        if matrix_shape[0] != n_constraints:
            raise ValueError(f"约束矩阵行数 ({matrix_shape[0]}) 与约束数 ({n_constraints}) 不匹配")
        
        # 向量长度检查
        if len(self.standard_form.rhs_vector) != n_constraints:
            raise ValueError(f"右侧向量长度与约束数不匹配")
        
        if len(self.standard_form.objective_coefficients) != n_vars:
            raise ValueError(f"目标函数系数向量长度与变量数不匹配")
        
        if len(self.standard_form.lower_bounds) != n_vars:
            raise ValueError(f"下界向量长度与变量数不匹配")
            
        if len(self.standard_form.upper_bounds) != n_vars:
            raise ValueError(f"上界向量长度与变量数不匹配")
        
        logger.info("✅ 提取结果验证通过")
    
    def _compute_matrix_density(self) -> float:
        """计算约束矩阵密度"""
        if self.standard_form is None:
            return 0.0
        
        matrix = self.standard_form.constraint_matrix
        
        if sp.issparse(matrix):
            nnz = matrix.nnz
            total = matrix.shape[0] * matrix.shape[1]
        else:
            nnz = np.count_nonzero(matrix)
            total = matrix.size
        
        return nnz / total if total > 0 else 0.0
    
    def get_extraction_report(self) -> Dict[str, Any]:
        """获取提取过程报告"""
        if self.standard_form is None:
            return {"status": "not_extracted"}
        
        report = {
            "problem_name": self.standard_form.problem_name,
            "extraction_time": self.standard_form.extraction_time.isoformat(),
            "dimensions": {
                "n_variables": self.standard_form.n_variables,
                "n_constraints": self.standard_form.n_constraints,
                "matrix_density": self._compute_matrix_density()
            },
            "variable_statistics": {
                "continuous": sum(1 for vt in self.standard_form.variable_types if vt == VariableType.CONTINUOUS),
                "binary": sum(1 for vt in self.standard_form.variable_types if vt == VariableType.BINARY),
                "integer": sum(1 for vt in self.standard_form.variable_types if vt == VariableType.INTEGER)
            },
            "constraint_statistics": {
                "equality": sum(1 for ct in self.standard_form.constraint_types if ct == ConstraintType.LINEAR_EQ),
                "inequality": sum(1 for ct in self.standard_form.constraint_types if ct == ConstraintType.LINEAR_INEQ),
                "soc": sum(1 for ct in self.standard_form.constraint_types if ct == ConstraintType.SOC),
                "quadratic": sum(1 for ct in self.standard_form.constraint_types if ct == ConstraintType.QUADRATIC)
            },
            "objective": {
                "sense": self.standard_form.objective_sense,
                "constant": self.standard_form.objective_constant,
                "nnz_coefficients": np.count_nonzero(self.standard_form.objective_coefficients)
            },
            "extraction_log": self.extraction_log
        }
        
        return report


def extract_from_cvxpy_problem(problem: cp.Problem, 
                              problem_name: str = None,
                              use_sparse: bool = True) -> MILPStandardForm:
    """
    便捷函数：从CVXPY问题中提取MILP标准形式
    
    Args:
        problem: CVXPY问题对象
        problem_name: 问题名称
        use_sparse: 是否使用稀疏矩阵
        
    Returns:
        MILP标准形式对象
    """
    extractor = CVXPYToMILPExtractor(problem, problem_name)
    return extractor.extract(use_sparse=use_sparse)


if __name__ == "__main__":
    """测试CVXPY提取器"""
    import sys
    from pathlib import Path
    
    # 添加项目路径
    project_root = Path(__file__).parent.parent.parent.parent
    sys.path.insert(0, str(project_root))
    
    # 创建简单的测试问题
    logger.info("创建测试CVXPY问题...")
    
    # 变量
    x = cp.Variable(3, name='x')
    y = cp.Variable(2, boolean=True, name='y')
    
    # 约束
    constraints = [
        x[0] + x[1] + x[2] <= 10,      # 不等式约束
        2*x[0] - x[1] == 5,            # 等式约束
        x >= 0,                        # 变量边界
        y[0] + y[1] <= 1               # 二进制变量约束
    ]
    
    # 目标函数
    objective = cp.Minimize(3*x[0] + 2*x[1] + x[2] + 5*y[0] + 3*y[1])
    
    # 创建问题
    problem = cp.Problem(objective, constraints)
    
    try:
        # 测试提取器
        extractor = CVXPYToMILPExtractor(problem, "测试问题")
        standard_form = extractor.extract()
        
        print("✅ CVXPY提取测试成功!")
        print(f"变量数: {standard_form.n_variables}")
        print(f"约束数: {standard_form.n_constraints}")
        print(f"矩阵密度: {extractor._compute_matrix_density():.4f}")
        
        # 打印提取报告
        report = extractor.get_extraction_report()
        print(f"\n提取报告:")
        print(f"  连续变量: {report['variable_statistics']['continuous']}")
        print(f"  二进制变量: {report['variable_statistics']['binary']}")
        print(f"  等式约束: {report['constraint_statistics']['equality']}")
        print(f"  不等式约束: {report['constraint_statistics']['inequality']}")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
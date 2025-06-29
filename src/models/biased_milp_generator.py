"""
有偏差MILP实例生成器
BiasedMILPGenerator - 基于扰动数据生成多种场景的MILP实例

功能特性：
1. 对SystemData进行多种类型的扰动
2. 基于扰动数据构建与Demo 1相同的CVXPY优化模型
3. 支持批量生成不同扰动场景的MILP实例
4. 提供MILP实例的序列化保存功能
5. 记录扰动参数和构建过程的元信息
6. 提供问题规模分析和统计信息
7. 完整的中文注释和错误处理
"""

import cvxpy as cp
import numpy as np
import pandas as pd
import pickle
import json
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime
import copy

from src.datasets.loader import SystemData, Generator, Load, Branch, SystemParameters
from src.models.post_disaster_dynamic_multi_period import PostDisasterDynamicModel, TrafficModel

# G2MILP二分图支持
try:
    from src.models.g2milp_bipartite import (
        G2MILPBipartiteGenerator, 
        BipartiteGraphRepresentation,
        create_g2milp_generator
    )
    G2MILP_AVAILABLE = True
except ImportError as e:
    logger.warning(f"G2MILP二分图模块不可用: {e}")
    G2MILP_AVAILABLE = False

# 设置日志
logger = logging.getLogger(__name__)


@dataclass
class PerturbationConfig:
    """扰动配置数据类"""
    # 负荷扰动参数
    load_perturbation_type: str = "gaussian"  # gaussian, uniform, percentage
    load_noise_std: float = 0.1  # 高斯噪声标准差
    load_noise_mean: float = 0.0  # 高斯噪声均值
    load_uniform_range: Tuple[float, float] = (-0.2, 0.2)  # 均匀分布范围
    load_percentage_range: Tuple[float, float] = (0.8, 1.2)  # 比例缩放范围
    
    # 发电机扰动参数
    generator_perturbation_type: str = "gaussian"
    generator_noise_std: float = 0.05
    generator_noise_mean: float = 0.0
    generator_uniform_range: Tuple[float, float] = (-0.1, 0.1)
    generator_percentage_range: Tuple[float, float] = (0.9, 1.1)
    
    # 支路参数扰动
    branch_perturbation_type: str = "gaussian"
    branch_noise_std: float = 0.02
    branch_noise_mean: float = 0.0
    branch_uniform_range: Tuple[float, float] = (-0.05, 0.05)
    branch_percentage_range: Tuple[float, float] = (0.95, 1.05)
    
    # PV出力扰动参数
    pv_perturbation_type: str = "gaussian"
    pv_noise_std: float = 0.15
    pv_noise_mean: float = 0.0
    pv_uniform_range: Tuple[float, float] = (-0.3, 0.3)
    pv_percentage_range: Tuple[float, float] = (0.7, 1.3)
    
    # 交通拥堵扰动参数
    traffic_perturbation_type: str = "gaussian"
    traffic_noise_std: float = 0.05
    traffic_noise_mean: float = 0.0
    traffic_uniform_range: Tuple[float, float] = (-0.1, 0.1)
    traffic_percentage_range: Tuple[float, float] = (0.9, 1.1)
    
    # 扰动种子（用于可重现性）
    random_seed: Optional[int] = None
    
    # 扰动强度控制
    perturbation_intensity: float = 1.0  # 整体扰动强度缩放因子


@dataclass
class ProblemStatistics:
    """MILP问题统计信息"""
    # 基本信息
    n_variables: int = 0
    n_continuous_vars: int = 0
    n_binary_vars: int = 0
    n_constraints: int = 0
    n_equality_constraints: int = 0
    n_inequality_constraints: int = 0
    n_soc_constraints: int = 0  # 二阶锥约束数量
    
    # 问题规模
    n_buses: int = 0
    n_branches: int = 0
    n_time_periods: int = 0
    n_generators: int = 0
    n_mess_units: int = 0
    n_evs_units: int = 0
    
    # 矩阵密度信息
    constraint_matrix_density: float = 0.0
    variable_bounds_count: int = 0
    
    # 数据统计
    max_load_value: float = 0.0
    min_load_value: float = 0.0
    total_load: float = 0.0
    max_generation_capacity: float = 0.0
    total_generation_capacity: float = 0.0


@dataclass
class MILPInstance:
    """MILP实例数据类"""
    # 基本信息
    instance_id: str
    creation_time: datetime
    problem_name: str = "Post-Disaster Dynamic Dispatch with Perturbation"
    
    # CVXPY问题对象
    cvxpy_problem: cp.Problem = None
    
    # 扰动后的系统数据
    perturbed_system_data: SystemData = None
    
    # 扰动配置
    perturbation_config: PerturbationConfig = None
    
    # 扰动记录
    perturbation_log: Dict[str, Any] = None
    
    # 问题统计
    statistics: ProblemStatistics = None
    
    # G2MILP二分图表示（可选）
    bipartite_graph: Optional[Any] = None  # BipartiteGraphRepresentation类型
    
    # 额外元信息
    metadata: Dict[str, Any] = None
    
    def generate_bipartite_graph(self, include_power_system_semantics: bool = True) -> bool:
        """
        生成G2MILP二分图表示
        
        Args:
            include_power_system_semantics: 是否包含电力系统语义增强
            
        Returns:
            是否生成成功
        """
        if not G2MILP_AVAILABLE:
            logger.warning("G2MILP模块不可用，无法生成二分图表示")
            return False
        
        if self.cvxpy_problem is None:
            logger.error("CVXPY问题对象不存在，无法生成二分图表示")
            return False
        
        try:
            logger.info(f"为实例 {self.instance_id} 生成G2MILP二分图表示...")
            
            # 创建G2MILP生成器
            g2milp_generator = create_g2milp_generator(include_power_system_semantics)
            
            # 生成二分图表示
            self.bipartite_graph = g2milp_generator.generate_from_milp_instance(
                self, include_perturbation_info=True
            )
            
            logger.info(f"二分图生成成功 - 约束节点: {self.bipartite_graph.n_constraint_nodes}, "
                       f"变量节点: {self.bipartite_graph.n_variable_nodes}, "
                       f"边: {self.bipartite_graph.n_edges}")
            
            # 更新元信息
            if self.metadata is None:
                self.metadata = {}
            self.metadata['bipartite_graph_generated'] = True
            self.metadata['bipartite_graph_stats'] = self.bipartite_graph.graph_statistics
            
            return True
            
        except Exception as e:
            logger.error(f"生成二分图表示失败: {e}")
            return False
    
    def save_bipartite_graph(self, filepath: str) -> bool:
        """
        保存二分图表示到文件
        
        Args:
            filepath: 保存路径
            
        Returns:
            是否保存成功
        """
        if self.bipartite_graph is None:
            logger.warning("二分图表示不存在，请先调用generate_bipartite_graph()")
            return False
        
        try:
            return self.bipartite_graph.save_to_file(filepath)
        except Exception as e:
            logger.error(f"保存二分图表示失败: {e}")
            return False
    
    def export_pytorch_geometric(self):
        """导出为PyTorch Geometric格式"""
        if self.bipartite_graph is None:
            logger.warning("二分图表示不存在，请先调用generate_bipartite_graph()")
            return None
        
        try:
            return self.bipartite_graph.to_pytorch_geometric()
        except Exception as e:
            logger.error(f"导出PyTorch Geometric格式失败: {e}")
            return None
    
    def export_dgl_graph(self):
        """导出为DGL图格式"""
        if self.bipartite_graph is None:
            logger.warning("二分图表示不存在，请先调用generate_bipartite_graph()")
            return None
        
        try:
            return self.bipartite_graph.to_dgl_graph()
        except Exception as e:
            logger.error(f"导出DGL图格式失败: {e}")
            return None


class DataPerturbation:
    """数据扰动模块"""
    
    def __init__(self, config: PerturbationConfig):
        """
        初始化数据扰动器
        
        Args:
            config: 扰动配置
        """
        self.config = config
        if config.random_seed is not None:
            np.random.seed(config.random_seed)
            logger.info(f"设置随机种子: {config.random_seed}")
    
    def perturb_system_data(self, original_data: SystemData) -> Tuple[SystemData, Dict[str, Any]]:
        """
        对系统数据进行扰动
        
        Args:
            original_data: 原始系统数据
            
        Returns:
            扰动后的系统数据和扰动记录
        """
        logger.info("开始数据扰动...")
        
        # 深拷贝原始数据
        perturbed_data = copy.deepcopy(original_data)
        perturbation_log = {}
        
        # 扰动负荷数据
        load_log = self._perturb_loads(perturbed_data.loads)
        perturbation_log['loads'] = load_log
        
        # 扰动发电机数据
        generator_log = self._perturb_generators(perturbed_data.generators)
        perturbation_log['generators'] = generator_log
        
        # 扰动支路参数
        branch_log = self._perturb_branches(perturbed_data.branches)
        perturbation_log['branches'] = branch_log
        
        # 扰动时变数据（PV出力、交通拥堵）
        if perturbed_data.load_profiles is not None:
            pv_log = self._perturb_pv_profiles(perturbed_data.load_profiles)
            perturbation_log['pv_profiles'] = pv_log
        
        if perturbed_data.traffic_profiles is not None:
            traffic_log = self._perturb_traffic_profiles(perturbed_data.traffic_profiles)
            perturbation_log['traffic_profiles'] = traffic_log
        
        logger.info("数据扰动完成")
        return perturbed_data, perturbation_log
    
    def _perturb_loads(self, loads: Dict[str, Load]) -> Dict[str, Dict]:
        """扰动负荷数据"""
        log = {}
        
        for load_name, load in loads.items():
            original_p = load.active_power
            original_q = load.reactive_power
            
            # 扰动有功功率
            perturbation_p = self._generate_perturbation(
                original_p, self.config.load_perturbation_type,
                self.config.load_noise_std, self.config.load_noise_mean,
                self.config.load_uniform_range, self.config.load_percentage_range
            )
            
            # 扰动无功功率
            perturbation_q = self._generate_perturbation(
                original_q, self.config.load_perturbation_type,
                self.config.load_noise_std, self.config.load_noise_mean,
                self.config.load_uniform_range, self.config.load_percentage_range
            )
            
            # 应用扰动强度缩放
            perturbation_p *= self.config.perturbation_intensity
            perturbation_q *= self.config.perturbation_intensity
            
            # 更新负荷值（确保非负）
            load.active_power = max(0, original_p + perturbation_p)
            load.reactive_power = max(0, original_q + perturbation_q)
            
            # 记录扰动信息
            log[load_name] = {
                'original_active_power': original_p,
                'original_reactive_power': original_q,
                'perturbation_active_power': perturbation_p,
                'perturbation_reactive_power': perturbation_q,
                'final_active_power': load.active_power,
                'final_reactive_power': load.reactive_power,
                'perturbation_ratio_p': perturbation_p / original_p if original_p > 0 else 0,
                'perturbation_ratio_q': perturbation_q / original_q if original_q > 0 else 0,
            }
        
        return log
    
    def _perturb_generators(self, generators: Dict[str, Generator]) -> Dict[str, Dict]:
        """扰动发电机数据"""
        log = {}
        
        for gen_name, generator in generators.items():
            original_p_max = generator.active_power_max or 0
            original_q_max = generator.reactive_power_max or 0
            
            # 扰动最大有功功率
            if original_p_max > 0:
                perturbation_p = self._generate_perturbation(
                    original_p_max, self.config.generator_perturbation_type,
                    self.config.generator_noise_std, self.config.generator_noise_mean,
                    self.config.generator_uniform_range, self.config.generator_percentage_range
                )
                perturbation_p *= self.config.perturbation_intensity
                generator.active_power_max = max(0, original_p_max + perturbation_p)
            else:
                perturbation_p = 0
            
            # 扰动最大无功功率
            if original_q_max > 0:
                perturbation_q = self._generate_perturbation(
                    original_q_max, self.config.generator_perturbation_type,
                    self.config.generator_noise_std, self.config.generator_noise_mean,
                    self.config.generator_uniform_range, self.config.generator_percentage_range
                )
                perturbation_q *= self.config.perturbation_intensity
                generator.reactive_power_max = max(0, original_q_max + perturbation_q)
            else:
                perturbation_q = 0
            
            # 扰动光伏预测功率
            if generator.type == "PV" and generator.predicted_power:
                original_pv = generator.predicted_power
                perturbation_pv = self._generate_perturbation(
                    original_pv, self.config.pv_perturbation_type,
                    self.config.pv_noise_std, self.config.pv_noise_mean,
                    self.config.pv_uniform_range, self.config.pv_percentage_range
                )
                perturbation_pv *= self.config.perturbation_intensity
                generator.predicted_power = max(0, original_pv + perturbation_pv)
            else:
                original_pv = 0
                perturbation_pv = 0
            
            # 记录扰动信息
            log[gen_name] = {
                'type': generator.type,
                'original_active_power_max': original_p_max,
                'original_reactive_power_max': original_q_max,
                'original_predicted_power': original_pv,
                'perturbation_active_power': perturbation_p,
                'perturbation_reactive_power': perturbation_q,
                'perturbation_predicted_power': perturbation_pv,
                'final_active_power_max': generator.active_power_max,
                'final_reactive_power_max': generator.reactive_power_max,
                'final_predicted_power': generator.predicted_power,
            }
        
        return log
    
    def _perturb_branches(self, branches: Dict[str, Branch]) -> Dict[str, Dict]:
        """扰动支路参数"""
        log = {}
        
        for branch_name, branch in branches.items():
            original_r = branch.resistance
            original_x = branch.reactance
            original_cap = branch.capacity
            
            # 扰动电阻
            perturbation_r = self._generate_perturbation(
                original_r, self.config.branch_perturbation_type,
                self.config.branch_noise_std, self.config.branch_noise_mean,
                self.config.branch_uniform_range, self.config.branch_percentage_range
            )
            perturbation_r *= self.config.perturbation_intensity
            branch.resistance = max(1e-6, original_r + perturbation_r)  # 确保电阻为正
            
            # 扰动电抗
            perturbation_x = self._generate_perturbation(
                original_x, self.config.branch_perturbation_type,
                self.config.branch_noise_std, self.config.branch_noise_mean,
                self.config.branch_uniform_range, self.config.branch_percentage_range
            )
            perturbation_x *= self.config.perturbation_intensity
            branch.reactance = max(1e-6, original_x + perturbation_x)  # 确保电抗为正
            
            # 扰动容量
            perturbation_cap = self._generate_perturbation(
                original_cap, self.config.branch_perturbation_type,
                self.config.branch_noise_std, self.config.branch_noise_mean,
                self.config.branch_uniform_range, self.config.branch_percentage_range
            )
            perturbation_cap *= self.config.perturbation_intensity
            branch.capacity = max(1, original_cap + perturbation_cap)  # 确保容量为正
            
            # 记录扰动信息
            log[branch_name] = {
                'original_resistance': original_r,
                'original_reactance': original_x,
                'original_capacity': original_cap,
                'perturbation_resistance': perturbation_r,
                'perturbation_reactance': perturbation_x,
                'perturbation_capacity': perturbation_cap,
                'final_resistance': branch.resistance,
                'final_reactance': branch.reactance,
                'final_capacity': branch.capacity,
            }
        
        return log
    
    def _perturb_pv_profiles(self, pv_profiles: pd.DataFrame) -> Dict[str, Any]:
        """扰动光伏出力曲线"""
        log = {}
        
        # 获取PV相关列
        pv_columns = [col for col in pv_profiles.columns if col.startswith('PV') or 'pv' in col.lower()]
        
        for col in pv_columns:
            if col in pv_profiles.columns:
                original_values = pv_profiles[col].values.copy()
                perturbations = []
                
                for value in original_values:
                    if value > 0:  # 只对非零值进行扰动
                        perturbation = self._generate_perturbation(
                            value, self.config.pv_perturbation_type,
                            self.config.pv_noise_std, self.config.pv_noise_mean,
                            self.config.pv_uniform_range, self.config.pv_percentage_range
                        )
                        perturbation *= self.config.perturbation_intensity
                        perturbations.append(perturbation)
                        new_value = max(0, value + perturbation)
                        # 确保数据类型兼容
                        pv_profiles[col] = pv_profiles[col].astype(float)
                        pv_profiles.loc[pv_profiles[col] == value, col] = new_value
                    else:
                        perturbations.append(0)
                
                log[col] = {
                    'original_values': original_values.tolist(),
                    'perturbations': perturbations,
                    'final_values': pv_profiles[col].values.tolist(),
                }
        
        return log
    
    def _perturb_traffic_profiles(self, traffic_profiles: pd.DataFrame) -> Dict[str, Any]:
        """扰动交通拥堵曲线"""
        log = {}
        
        # 获取交通拥堵相关列
        traffic_columns = [col for col in traffic_profiles.columns if 'congestion' in col.lower() or 'traffic' in col.lower()]
        
        for col in traffic_columns:
            if col in traffic_profiles.columns:
                original_values = traffic_profiles[col].values.copy()
                perturbations = []
                
                for value in original_values:
                    perturbation = self._generate_perturbation(
                        value, self.config.traffic_perturbation_type,
                        self.config.traffic_noise_std, self.config.traffic_noise_mean,
                        self.config.traffic_uniform_range, self.config.traffic_percentage_range
                    )
                    perturbation *= self.config.perturbation_intensity
                    perturbations.append(perturbation)
                    # 交通拥堵程度限制在[0, 1]之间
                    new_value = max(0, min(1, value + perturbation))
                    # 确保数据类型兼容
                    traffic_profiles[col] = traffic_profiles[col].astype(float)
                    traffic_profiles.loc[traffic_profiles[col] == value, col] = new_value
                
                log[col] = {
                    'original_values': original_values.tolist(),
                    'perturbations': perturbations,
                    'final_values': traffic_profiles[col].values.tolist(),
                }
        
        return log
    
    def _generate_perturbation(self, base_value: float, perturbation_type: str,
                             noise_std: float, noise_mean: float,
                             uniform_range: Tuple[float, float],
                             percentage_range: Tuple[float, float]) -> float:
        """
        生成扰动值
        
        Args:
            base_value: 基础值
            perturbation_type: 扰动类型
            noise_std: 高斯噪声标准差
            noise_mean: 高斯噪声均值
            uniform_range: 均匀分布范围
            percentage_range: 比例缩放范围
            
        Returns:
            扰动值
        """
        if perturbation_type == "gaussian":
            # 高斯噪声：相对于基础值的比例扰动
            return np.random.normal(noise_mean, noise_std) * base_value
        elif perturbation_type == "uniform":
            # 均匀分布：相对于基础值的比例扰动
            return np.random.uniform(uniform_range[0], uniform_range[1]) * base_value
        elif perturbation_type == "percentage":
            # 比例缩放：直接乘以缩放因子再减去原值
            scale = np.random.uniform(percentage_range[0], percentage_range[1])
            return base_value * scale - base_value
        else:
            logger.warning(f"未知的扰动类型: {perturbation_type}, 使用默认高斯扰动")
            return np.random.normal(noise_mean, noise_std) * base_value


class ProblemAnalyzer:
    """MILP问题分析器"""
    
    @staticmethod
    def analyze_problem(cvxpy_problem: cp.Problem, model: PostDisasterDynamicModel) -> ProblemStatistics:
        """
        分析CVXPY问题的统计信息
        
        Args:
            cvxpy_problem: CVXPY问题对象
            model: 动态调度模型对象
            
        Returns:
            问题统计信息
        """
        logger.info("开始分析MILP问题统计信息...")
        
        stats = ProblemStatistics()
        
        # 基本问题信息
        if hasattr(cvxpy_problem, 'size_metrics'):
            metrics = cvxpy_problem.size_metrics
            stats.n_variables = metrics.num_scalar_variables
            stats.n_constraints = metrics.num_scalar_eq_constr + metrics.num_scalar_leq_constr
            stats.n_equality_constraints = metrics.num_scalar_eq_constr
            stats.n_inequality_constraints = metrics.num_scalar_leq_constr
        
        # 从模型中获取问题规模信息
        stats.n_buses = model.n_buses
        stats.n_branches = model.n_branches
        stats.n_time_periods = model.n_periods
        stats.n_mess_units = model.n_mess
        
        # 统计变量类型（CVXPY中的变量类型分析）
        continuous_count = 0
        binary_count = 0
        
        for var in cvxpy_problem.variables():
            if var.attributes.get('boolean', False):
                binary_count += var.size
            else:
                continuous_count += var.size
        
        stats.n_continuous_vars = continuous_count
        stats.n_binary_vars = binary_count
        
        # 统计二阶锥约束数量
        soc_count = 0
        try:
            for constraint in cvxpy_problem.constraints:
                if hasattr(constraint, 'args') and len(constraint.args) > 0:
                    # 检查是否为二阶锥约束 - 更安全的类型检查
                    for arg in constraint.args:
                        if hasattr(arg, '__class__') and 'norm' in str(arg.__class__):
                            soc_count += 1
                            break
        except Exception as e:
            logger.warning(f"统计二阶锥约束时出错: {e}")
        stats.n_soc_constraints = soc_count
        
        # 数据统计
        stats.max_load_value = float(np.max(model.P_load))
        stats.min_load_value = float(np.min(model.P_load[model.P_load > 0]))
        stats.total_load = float(np.sum(model.P_load))
        
        # 发电容量统计
        total_gen_capacity = 0
        if hasattr(model, 'P_deg_max'):
            total_gen_capacity += np.sum(model.P_deg_max)
        if hasattr(model, 'P_evs_max'):
            total_gen_capacity += np.sum(model.P_evs_max)
        if hasattr(model, 'P_mess_max'):
            total_gen_capacity += model.P_mess_max * model.n_mess
        
        stats.max_generation_capacity = float(total_gen_capacity)
        stats.total_generation_capacity = float(total_gen_capacity)
        
        # 统计EVS数量
        stats.n_evs_units = int(np.sum(model.E_evs_cap > 0)) if hasattr(model, 'E_evs_cap') else 0
        
        # 统计发电机数量
        stats.n_generators = len([g for g in model.data.generators.values() if g.active_power_max and g.active_power_max > 0])
        
        logger.info(f"问题分析完成 - 变量数: {stats.n_variables}, 约束数: {stats.n_constraints}")
        return stats


class MILPSerializer:
    """MILP实例序列化工具"""
    
    @staticmethod
    def save_instance(instance: MILPInstance, filepath: str) -> bool:
        """
        保存MILP实例到文件
        
        Args:
            instance: MILP实例对象
            filepath: 保存路径
            
        Returns:
            是否保存成功
        """
        try:
            logger.info(f"保存MILP实例到: {filepath}")
            
            # 创建保存目录
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            
            # 使用pickle保存完整实例
            with open(filepath, 'wb') as f:
                pickle.dump(instance, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # 同时保存元信息为JSON文件
            json_filepath = filepath.replace('.pkl', '_metadata.json')
            metadata = {
                'instance_id': instance.instance_id,
                'creation_time': instance.creation_time.isoformat(),
                'problem_name': instance.problem_name,
                'perturbation_config': asdict(instance.perturbation_config) if instance.perturbation_config else None,
                'statistics': asdict(instance.statistics) if instance.statistics else None,
                'metadata': instance.metadata,
            }
            
            with open(json_filepath, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"MILP实例保存成功: {filepath}")
            logger.info(f"元信息保存成功: {json_filepath}")
            return True
            
        except Exception as e:
            logger.error(f"保存MILP实例失败: {e}")
            return False
    
    @staticmethod
    def load_instance(filepath: str) -> Optional[MILPInstance]:
        """
        从文件加载MILP实例
        
        Args:
            filepath: 文件路径
            
        Returns:
            MILP实例对象或None
        """
        try:
            logger.info(f"加载MILP实例: {filepath}")
            
            if not Path(filepath).exists():
                logger.error(f"文件不存在: {filepath}")
                return None
            
            with open(filepath, 'rb') as f:
                instance = pickle.load(f)
            
            logger.info(f"MILP实例加载成功: {instance.instance_id}")
            return instance
            
        except Exception as e:
            logger.error(f"加载MILP实例失败: {e}")
            return None


class BiasedMILPGenerator:
    """有偏差MILP实例生成器"""
    
    def __init__(self, base_system_data: SystemData, 
                 output_dir: str = "output/milp_instances",
                 log_dir: str = "logs"):
        """
        初始化MILP实例生成器
        
        Args:
            base_system_data: 基础系统数据
            output_dir: 输出目录
            log_dir: 日志目录
        """
        self.base_system_data = base_system_data
        self.output_dir = Path(output_dir)
        self.log_dir = Path(log_dir)
        
        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置日志
        self._setup_logging()
        
        logger.info("BiasedMILPGenerator 初始化完成")
        logger.info(f"输出目录: {self.output_dir}")
        logger.info(f"日志目录: {self.log_dir}")
    
    def _setup_logging(self):
        """设置日志配置"""
        log_file = self.log_dir / f"milp_generator_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        # 创建文件处理器
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # 创建控制台处理器  
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # 设置格式
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # 添加处理器到logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        logger.setLevel(logging.INFO)
    
    def generate_single_instance(self, 
                               perturbation_config: PerturbationConfig,
                               instance_id: Optional[str] = None,
                               n_periods: int = 21,
                               start_hour: int = 3,
                               save_to_file: bool = True) -> MILPInstance:
        """
        生成单个MILP实例
        
        Args:
            perturbation_config: 扰动配置
            instance_id: 实例ID（如果为None则自动生成）
            n_periods: 时间段数量
            start_hour: 起始小时
            save_to_file: 是否保存到文件
            
        Returns:
            MILP实例对象
        """
        # 生成实例ID
        if instance_id is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]  # 精确到毫秒
            instance_id = f"milp_instance_{timestamp}"
        
        logger.info(f"开始生成MILP实例: {instance_id}")
        
        try:
            # 1. 数据扰动
            perturbator = DataPerturbation(perturbation_config)
            perturbed_data, perturbation_log = perturbator.perturb_system_data(self.base_system_data)
            
            # 2. 构建优化模型
            logger.info("构建优化模型...")
            model = PostDisasterDynamicModel(
                system_data=perturbed_data,
                n_periods=n_periods,
                start_hour=start_hour,
                traffic_profile_path=None,  # 使用默认数据
                pv_profile_path=None  # 使用默认数据
            )
            
            # 3. 分析问题统计信息
            logger.info("分析问题统计信息...")
            statistics = ProblemAnalyzer.analyze_problem(model.problem, model)
            
            # 4. 创建MILP实例
            instance = MILPInstance(
                instance_id=instance_id,
                creation_time=datetime.now(),
                cvxpy_problem=model.problem,
                perturbed_system_data=perturbed_data,
                perturbation_config=perturbation_config,
                perturbation_log=perturbation_log,
                statistics=statistics,
                metadata={
                    'n_periods': n_periods,
                    'start_hour': start_hour,
                    'base_data_info': {
                        'n_generators': len(self.base_system_data.generators),
                        'n_loads': len(self.base_system_data.loads),
                        'n_branches': len(self.base_system_data.branches),
                    }
                }
            )
            
            # 5. 保存到文件
            if save_to_file:
                filepath = self.output_dir / f"{instance_id}.pkl"
                success = MILPSerializer.save_instance(instance, str(filepath))
                if success:
                    instance.metadata['saved_filepath'] = str(filepath)
            
            logger.info(f"MILP实例生成完成: {instance_id}")
            return instance
            
        except Exception as e:
            logger.error(f"生成MILP实例失败: {e}")
            raise
    
    def generate_batch_instances(self,
                               perturbation_configs: List[PerturbationConfig],
                               instance_prefix: str = "batch",
                               n_periods: int = 21,
                               start_hour: int = 3,
                               save_to_file: bool = True) -> List[MILPInstance]:
        """
        批量生成MILP实例
        
        Args:
            perturbation_configs: 扰动配置列表
            instance_prefix: 实例ID前缀
            n_periods: 时间段数量
            start_hour: 起始小时
            save_to_file: 是否保存到文件
            
        Returns:
            MILP实例列表
        """
        logger.info(f"开始批量生成MILP实例，数量: {len(perturbation_configs)}")
        
        instances = []
        for i, config in enumerate(perturbation_configs):
            try:
                instance_id = f"{instance_prefix}_{i+1:03d}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                
                logger.info(f"生成实例 {i+1}/{len(perturbation_configs)}: {instance_id}")
                
                instance = self.generate_single_instance(
                    perturbation_config=config,
                    instance_id=instance_id,
                    n_periods=n_periods,
                    start_hour=start_hour,
                    save_to_file=save_to_file
                )
                
                instances.append(instance)
                
            except Exception as e:
                logger.error(f"生成第{i+1}个实例失败: {e}")
                continue
        
        logger.info(f"批量生成完成，成功生成 {len(instances)} 个实例")
        return instances
    
    def generate_scenario_instances(self,
                                  scenario_configs: Dict[str, PerturbationConfig],
                                  n_periods: int = 21,
                                  start_hour: int = 3,
                                  save_to_file: bool = True) -> Dict[str, MILPInstance]:
        """
        生成不同场景的MILP实例
        
        Args:
            scenario_configs: 场景配置字典 {场景名: 扰动配置}
            n_periods: 时间段数量
            start_hour: 起始小时
            save_to_file: 是否保存到文件
            
        Returns:
            场景实例字典 {场景名: MILP实例}
        """
        logger.info(f"开始生成场景实例，场景数量: {len(scenario_configs)}")
        
        scenario_instances = {}
        for scenario_name, config in scenario_configs.items():
            try:
                instance_id = f"scenario_{scenario_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                
                logger.info(f"生成场景实例: {scenario_name}")
                
                instance = self.generate_single_instance(
                    perturbation_config=config,
                    instance_id=instance_id,
                    n_periods=n_periods,
                    start_hour=start_hour,
                    save_to_file=save_to_file
                )
                
                scenario_instances[scenario_name] = instance
                
            except Exception as e:
                logger.error(f"生成场景 {scenario_name} 实例失败: {e}")
                continue
        
        logger.info(f"场景实例生成完成，成功生成 {len(scenario_instances)} 个场景")
        return scenario_instances
    
    def generate_bipartite_graphs_for_instances(self,
                                              instances: List[MILPInstance],
                                              include_power_system_semantics: bool = True,
                                              save_graphs: bool = True,
                                              graph_output_dir: Optional[str] = None) -> List[MILPInstance]:
        """
        为MILP实例批量生成G2MILP二分图表示
        
        Args:
            instances: MILP实例列表
            include_power_system_semantics: 是否包含电力系统语义增强
            save_graphs: 是否保存二分图到文件
            graph_output_dir: 二分图保存目录（可选）
            
        Returns:
            更新后的MILP实例列表（包含二分图）
        """
        if not G2MILP_AVAILABLE:
            logger.warning("G2MILP模块不可用，跳过二分图生成")
            return instances
        
        logger.info(f"开始为 {len(instances)} 个实例生成G2MILP二分图表示...")
        
        # 设置保存目录
        if save_graphs and graph_output_dir is None:
            graph_output_dir = self.output_dir / "bipartite_graphs"
        
        if save_graphs:
            Path(graph_output_dir).mkdir(parents=True, exist_ok=True)
        
        successful_count = 0
        for i, instance in enumerate(instances):
            try:
                logger.info(f"生成二分图 {i+1}/{len(instances)}: {instance.instance_id}")
                
                # 生成二分图表示
                success = instance.generate_bipartite_graph(include_power_system_semantics)
                
                if success:
                    successful_count += 1
                    
                    # 保存二分图到文件
                    if save_graphs:
                        graph_filepath = Path(graph_output_dir) / f"{instance.instance_id}_bipartite.pkl"
                        instance.save_bipartite_graph(str(graph_filepath))
                        
                        logger.info(f"二分图已保存: {graph_filepath}")
                else:
                    logger.warning(f"实例 {instance.instance_id} 二分图生成失败")
                    
            except Exception as e:
                logger.error(f"实例 {instance.instance_id} 二分图生成过程出错: {e}")
                continue
        
        logger.info(f"二分图生成完成，成功: {successful_count}/{len(instances)}")
        return instances
    
    def analyze_bipartite_graph_statistics(self, instances: List[MILPInstance]) -> Dict[str, Any]:
        """
        分析二分图统计信息
        
        Args:
            instances: 包含二分图的MILP实例列表
            
        Returns:
            二分图统计分析结果
        """
        if not G2MILP_AVAILABLE:
            logger.warning("G2MILP模块不可用，无法分析二分图统计")
            return {}
        
        logger.info("分析二分图统计信息...")
        
        valid_graphs = [inst for inst in instances if inst.bipartite_graph is not None]
        
        if not valid_graphs:
            logger.warning("没有找到有效的二分图表示")
            return {}
        
        # 收集统计数据
        stats = {
            'total_instances': len(instances),
            'valid_bipartite_graphs': len(valid_graphs),
            'constraint_nodes': [graph.bipartite_graph.n_constraint_nodes for graph in valid_graphs],
            'variable_nodes': [graph.bipartite_graph.n_variable_nodes for graph in valid_graphs],
            'edges': [graph.bipartite_graph.n_edges for graph in valid_graphs],
            'graph_densities': [graph.bipartite_graph.graph_statistics.get('bipartite_density', 0) for graph in valid_graphs]
        }
        
        # 计算聚合统计
        analysis = {
            'total_instances': stats['total_instances'],
            'valid_bipartite_graphs': stats['valid_bipartite_graphs'],
            'coverage_rate': stats['valid_bipartite_graphs'] / stats['total_instances'],
            
            'constraint_nodes_stats': {
                'mean': np.mean(stats['constraint_nodes']),
                'std': np.std(stats['constraint_nodes']),
                'min': np.min(stats['constraint_nodes']),
                'max': np.max(stats['constraint_nodes'])
            },
            
            'variable_nodes_stats': {
                'mean': np.mean(stats['variable_nodes']),
                'std': np.std(stats['variable_nodes']),
                'min': np.min(stats['variable_nodes']),
                'max': np.max(stats['variable_nodes'])
            },
            
            'edges_stats': {
                'mean': np.mean(stats['edges']),
                'std': np.std(stats['edges']),
                'min': np.min(stats['edges']),
                'max': np.max(stats['edges'])
            },
            
            'density_stats': {
                'mean': np.mean(stats['graph_densities']),
                'std': np.std(stats['graph_densities']),
                'min': np.min(stats['graph_densities']),
                'max': np.max(stats['graph_densities'])
            }
        }
        
        logger.info(f"二分图统计分析完成:")
        logger.info(f"  有效图数量: {analysis['valid_bipartite_graphs']}/{analysis['total_instances']}")
        logger.info(f"  平均约束节点数: {analysis['constraint_nodes_stats']['mean']:.1f}")
        logger.info(f"  平均变量节点数: {analysis['variable_nodes_stats']['mean']:.1f}")
        logger.info(f"  平均边数: {analysis['edges_stats']['mean']:.1f}")
        logger.info(f"  平均图密度: {analysis['density_stats']['mean']:.4f}")
        
        return analysis
    
    def print_instance_summary(self, instance: MILPInstance):
        """
        打印实例摘要信息
        
        Args:
            instance: MILP实例
        """
        print("\n" + "="*80)
        print(f"MILP实例摘要: {instance.instance_id}")
        print("="*80)
        
        print(f"创建时间: {instance.creation_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"问题名称: {instance.problem_name}")
        
        if instance.statistics:
            stats = instance.statistics
            print(f"\n问题规模:")
            print(f"  变量数量: {stats.n_variables} (连续: {stats.n_continuous_vars}, 二进制: {stats.n_binary_vars})")
            print(f"  约束数量: {stats.n_constraints} (等式: {stats.n_equality_constraints}, 不等式: {stats.n_inequality_constraints})")
            print(f"  二阶锥约束: {stats.n_soc_constraints}")
            print(f"  节点数量: {stats.n_buses}")
            print(f"  支路数量: {stats.n_branches}")
            print(f"  时间段数: {stats.n_time_periods}")
            print(f"  移动储能: {stats.n_mess_units}")
            print(f"  固定储能: {stats.n_evs_units}")
            print(f"  发电机数: {stats.n_generators}")
            
            print(f"\n数据统计:")
            print(f"  总负荷: {stats.total_load:.2f} MW")
            print(f"  最大负荷: {stats.max_load_value:.2f} MW")
            print(f"  最小负荷: {stats.min_load_value:.2f} MW")
            print(f"  总发电容量: {stats.total_generation_capacity:.2f} MW")
        
        if instance.perturbation_config:
            config = instance.perturbation_config
            print(f"\n扰动配置:")
            print(f"  负荷扰动类型: {config.load_perturbation_type}")
            print(f"  负荷扰动强度: {config.load_noise_std}")
            print(f"  发电机扰动类型: {config.generator_perturbation_type}")
            print(f"  发电机扰动强度: {config.generator_noise_std}")
            print(f"  整体扰动强度: {config.perturbation_intensity}")
            if config.random_seed is not None:
                print(f"  随机种子: {config.random_seed}")
        
        # G2MILP二分图信息
        if instance.bipartite_graph is not None:
            bg = instance.bipartite_graph
            print(f"\nG2MILP二分图表示:")
            print(f"  约束节点: {bg.n_constraint_nodes}")
            print(f"  变量节点: {bg.n_variable_nodes}")
            print(f"  边数量: {bg.n_edges}")
            print(f"  图密度: {bg.graph_statistics.get('bipartite_density', 0):.4f}")
            print(f"  平均约束度数: {bg.graph_statistics.get('avg_constraint_degree', 0):.2f}")
            print(f"  平均变量度数: {bg.graph_statistics.get('avg_variable_degree', 0):.2f}")
            print(f"  扰动应用: {'是' if bg.perturbation_applied else '否'}")
        elif G2MILP_AVAILABLE:
            print(f"\nG2MILP二分图表示: 未生成 (可调用 instance.generate_bipartite_graph())")
        
        if instance.metadata and 'saved_filepath' in instance.metadata:
            print(f"\n保存路径: {instance.metadata['saved_filepath']}")
        
        print("="*80)


def create_default_perturbation_configs() -> List[PerturbationConfig]:
    """
    创建默认扰动配置集合
    
    Returns:
        默认扰动配置列表
    """
    configs = []
    
    # 1. 轻微扰动
    configs.append(PerturbationConfig(
        load_perturbation_type="gaussian",
        load_noise_std=0.05,
        generator_noise_std=0.03,
        branch_noise_std=0.01,
        pv_noise_std=0.08,
        traffic_noise_std=0.03,
        perturbation_intensity=0.5,
        random_seed=42
    ))
    
    # 2. 中等扰动
    configs.append(PerturbationConfig(
        load_perturbation_type="gaussian",
        load_noise_std=0.1,
        generator_noise_std=0.05,
        branch_noise_std=0.02,
        pv_noise_std=0.15,
        traffic_noise_std=0.05,
        perturbation_intensity=1.0,
        random_seed=123
    ))
    
    # 3. 强扰动
    configs.append(PerturbationConfig(
        load_perturbation_type="gaussian",
        load_noise_std=0.2,
        generator_noise_std=0.1,
        branch_noise_std=0.05,
        pv_noise_std=0.25,
        traffic_noise_std=0.1,
        perturbation_intensity=1.5,
        random_seed=456
    ))
    
    # 4. 均匀分布扰动
    configs.append(PerturbationConfig(
        load_perturbation_type="uniform",
        load_uniform_range=(-0.15, 0.15),
        generator_perturbation_type="uniform",
        generator_uniform_range=(-0.08, 0.08),
        branch_perturbation_type="uniform",
        branch_uniform_range=(-0.03, 0.03),
        pv_perturbation_type="uniform",
        pv_uniform_range=(-0.2, 0.2),
        traffic_perturbation_type="uniform",
        traffic_uniform_range=(-0.08, 0.08),
        perturbation_intensity=1.0,
        random_seed=789
    ))
    
    # 5. 比例缩放扰动
    configs.append(PerturbationConfig(
        load_perturbation_type="percentage",
        load_percentage_range=(0.85, 1.15),
        generator_perturbation_type="percentage",
        generator_percentage_range=(0.9, 1.1),
        branch_perturbation_type="percentage",
        branch_percentage_range=(0.95, 1.05),
        pv_perturbation_type="percentage",
        pv_percentage_range=(0.8, 1.2),
        traffic_perturbation_type="percentage",
        traffic_percentage_range=(0.9, 1.1),
        perturbation_intensity=1.0,
        random_seed=321
    ))
    
    return configs


def create_scenario_perturbation_configs() -> Dict[str, PerturbationConfig]:
    """
    创建不同场景的扰动配置
    
    Returns:
        场景扰动配置字典
    """
    scenarios = {}
    
    # 负荷高峰场景
    scenarios["load_peak"] = PerturbationConfig(
        load_perturbation_type="percentage",
        load_percentage_range=(1.1, 1.3),  # 负荷增加10%-30%
        generator_noise_std=0.05,
        pv_noise_std=0.1,
        perturbation_intensity=1.0,
        random_seed=100
    )
    
    # 负荷低谷场景
    scenarios["load_valley"] = PerturbationConfig(
        load_perturbation_type="percentage",
        load_percentage_range=(0.7, 0.9),  # 负荷减少10%-30%
        generator_noise_std=0.05,
        pv_noise_std=0.1,
        perturbation_intensity=1.0,
        random_seed=200
    )
    
    # 光伏出力不稳定场景
    scenarios["pv_unstable"] = PerturbationConfig(
        load_noise_std=0.05,
        generator_noise_std=0.03,
        pv_perturbation_type="gaussian",
        pv_noise_std=0.3,  # 光伏出力高度不确定
        perturbation_intensity=1.2,
        random_seed=300
    )
    
    # 交通严重拥堵场景
    scenarios["traffic_jam"] = PerturbationConfig(
        load_noise_std=0.08,
        generator_noise_std=0.05,
        traffic_perturbation_type="percentage",
        traffic_percentage_range=(1.2, 1.5),  # 拥堵程度增加20%-50%
        perturbation_intensity=1.0,
        random_seed=400
    )
    
    # 设备故障场景
    scenarios["equipment_failure"] = PerturbationConfig(
        generator_perturbation_type="percentage",
        generator_percentage_range=(0.6, 0.8),  # 发电容量下降20%-40%
        branch_perturbation_type="percentage",
        branch_percentage_range=(0.8, 0.95),  # 支路容量下降5%-20%
        load_noise_std=0.1,
        perturbation_intensity=1.0,
        random_seed=500
    )
    
    return scenarios


if __name__ == "__main__":
    # 示例用法
    print("BiasedMILPGenerator 示例用法")
    
    # 这里只是演示代码结构，实际使用时需要加载真实的系统数据
    # from src.datasets.loader import load_system_data
    # system_data = load_system_data("data")
    # 
    # generator = BiasedMILPGenerator(system_data)
    # 
    # # 生成单个实例
    # config = PerturbationConfig(random_seed=42)
    # instance = generator.generate_single_instance(config)
    # generator.print_instance_summary(instance)
    # 
    # # 批量生成实例
    # configs = create_default_perturbation_configs()
    # instances = generator.generate_batch_instances(configs)
    # 
    # # 生成场景实例
    # scenario_configs = create_scenario_perturbation_configs()
    # scenario_instances = generator.generate_scenario_instances(scenario_configs)
    
    print("请参考主函数中的注释代码了解使用方法")
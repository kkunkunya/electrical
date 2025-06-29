"""
数据扰动模块
Data Perturbation Module

用于对配电网系统数据进行各种扰动，模拟实际运行中的不确定性因素
For perturbing distribution network system data to simulate uncertainties in real operations

主要功能：
- 负荷数据扰动：模拟负荷信息上报不全
- 光伏出力扰动：模拟光伏预测不准确
- 交通状况扰动：模拟交通拥堵系数变化
- 配置化扰动参数
- 扰动前后对比分析

Author: AI Assistant
Date: 2025-06-27
"""

import copy
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path
import json
from datetime import datetime

# 导入系统数据类
from ..datasets.loader import SystemData, Load, Generator


@dataclass
class PerturbationConfig:
    """扰动配置参数类"""
    
    # 负荷扰动参数
    load_perturbation_enabled: bool = True
    load_reduction_ratio: float = 0.1  # 负荷降低比例 (0.0-1.0)
    load_affected_node_ratio: float = 0.3  # 受影响节点比例 (0.0-1.0)
    load_perturbation_type: str = "random_reduction"  # 扰动类型: random_reduction, gaussian_noise
    load_noise_std: float = 0.05  # 高斯噪声标准差（用于gaussian_noise类型）
    
    # 光伏扰动参数
    pv_perturbation_enabled: bool = True
    pv_noise_type: str = "gaussian"  # 噪声类型: gaussian, multiplicative, additive
    pv_noise_std: float = 0.1  # 高斯噪声标准差
    pv_multiplicative_factor_range: Tuple[float, float] = (0.8, 1.2)  # 乘性因子范围
    pv_additive_noise_range: Tuple[float, float] = (-20.0, 20.0)  # 加性噪声范围(kW)
    
    # 交通扰动参数
    traffic_perturbation_enabled: bool = True
    traffic_noise_std: float = 0.05  # 交通拥堵系数噪声标准差
    traffic_congestion_factor_range: Tuple[float, float] = (0.9, 1.1)  # 拥堵系数调整范围
    
    # 随机种子
    random_seed: Optional[int] = None
    
    # 日志配置
    enable_logging: bool = True
    log_level: str = "INFO"


class DataPerturbationComparator:
    """数据扰动对比器"""
    
    def __init__(self, original_data: SystemData, perturbed_data: SystemData):
        """
        初始化对比器
        
        Args:
            original_data: 原始数据
            perturbed_data: 扰动后数据
        """
        self.original_data = original_data
        self.perturbed_data = perturbed_data
        self.comparison_results = {}
    
    def compare_loads(self) -> Dict:
        """对比负荷数据变化"""
        load_comparison = {
            "changed_nodes": [],
            "total_active_power_change": 0.0,
            "total_reactive_power_change": 0.0,
            "max_active_power_change_percent": 0.0,
            "max_reactive_power_change_percent": 0.0
        }
        
        for load_name in self.original_data.loads:
            orig_load = self.original_data.loads[load_name]
            pert_load = self.perturbed_data.loads[load_name]
            
            active_change = pert_load.active_power - orig_load.active_power
            reactive_change = pert_load.reactive_power - orig_load.reactive_power
            
            if abs(active_change) > 1e-6 or abs(reactive_change) > 1e-6:
                active_change_percent = (active_change / orig_load.active_power * 100) if orig_load.active_power > 0 else 0
                reactive_change_percent = (reactive_change / orig_load.reactive_power * 100) if orig_load.reactive_power > 0 else 0
                
                load_comparison["changed_nodes"].append({
                    "node_name": load_name,
                    "bus_id": orig_load.bus_id,
                    "active_power_change": active_change,
                    "reactive_power_change": reactive_change,
                    "active_power_change_percent": active_change_percent,
                    "reactive_power_change_percent": reactive_change_percent
                })
                
                load_comparison["total_active_power_change"] += active_change
                load_comparison["total_reactive_power_change"] += reactive_change
                
                load_comparison["max_active_power_change_percent"] = max(
                    load_comparison["max_active_power_change_percent"],
                    abs(active_change_percent)
                )
                load_comparison["max_reactive_power_change_percent"] = max(
                    load_comparison["max_reactive_power_change_percent"],
                    abs(reactive_change_percent)
                )
        
        return load_comparison
    
    def compare_pv_profiles(self) -> Dict:
        """对比光伏出力曲线变化"""
        if self.original_data.load_profiles is None or self.perturbed_data.load_profiles is None:
            return {"error": "光伏出力曲线数据缺失"}
        
        orig_pv = self.original_data.load_profiles
        pert_pv = self.perturbed_data.load_profiles
        
        pv_comparison = {
            "changed_columns": [],
            "total_energy_change": {},
            "max_power_change_percent": {},
            "rmse": {}
        }
        
        # 对比每个PV发电机的数据
        pv_columns = [col for col in orig_pv.columns if col.startswith('PV')]
        
        for col in pv_columns:
            if col in pert_pv.columns:
                orig_values = orig_pv[col].values
                pert_values = pert_pv[col].values
                
                # 计算变化
                differences = pert_values - orig_values
                
                if np.any(np.abs(differences) > 1e-6):
                    # 计算总能量变化
                    total_energy_change = np.sum(differences)
                    
                    # 计算最大功率变化百分比
                    max_orig = np.max(orig_values)
                    max_change_percent = (np.max(np.abs(differences)) / max_orig * 100) if max_orig > 0 else 0
                    
                    # 计算RMSE
                    rmse = np.sqrt(np.mean(differences**2))
                    
                    pv_comparison["changed_columns"].append(col)
                    pv_comparison["total_energy_change"][col] = total_energy_change
                    pv_comparison["max_power_change_percent"][col] = max_change_percent
                    pv_comparison["rmse"][col] = rmse
        
        return pv_comparison
    
    def compare_traffic_profiles(self) -> Dict:
        """对比交通拥堵曲线变化"""
        if self.original_data.traffic_profiles is None or self.perturbed_data.traffic_profiles is None:
            return {"error": "交通拥堵曲线数据缺失"}
        
        orig_traffic = self.original_data.traffic_profiles
        pert_traffic = self.perturbed_data.traffic_profiles
        
        traffic_comparison = {
            "congestion_level_changed": False,
            "max_congestion_change": 0.0,
            "avg_congestion_change": 0.0,
            "rmse": 0.0
        }
        
        if 'congestion_level' in orig_traffic.columns and 'congestion_level' in pert_traffic.columns:
            orig_values = orig_traffic['congestion_level'].values
            pert_values = pert_traffic['congestion_level'].values
            
            differences = pert_values - orig_values
            
            if np.any(np.abs(differences) > 1e-6):
                traffic_comparison["congestion_level_changed"] = True
                traffic_comparison["max_congestion_change"] = np.max(np.abs(differences))
                traffic_comparison["avg_congestion_change"] = np.mean(differences)
                traffic_comparison["rmse"] = np.sqrt(np.mean(differences**2))
        
        return traffic_comparison
    
    def generate_comparison_report(self) -> str:
        """生成对比报告"""
        load_comp = self.compare_loads()
        pv_comp = self.compare_pv_profiles()
        traffic_comp = self.compare_traffic_profiles()
        
        report = []
        report.append("=" * 60)
        report.append("数据扰动对比报告")
        report.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("=" * 60)
        
        # 负荷数据对比
        report.append("\n1. 负荷数据扰动分析")
        report.append("-" * 30)
        if load_comp["changed_nodes"]:
            report.append(f"受影响节点数量: {len(load_comp['changed_nodes'])}")
            report.append(f"总有功功率变化: {load_comp['total_active_power_change']:.2f} kW")
            report.append(f"总无功功率变化: {load_comp['total_reactive_power_change']:.2f} kvar")
            report.append(f"最大有功功率变化百分比: {load_comp['max_active_power_change_percent']:.2f}%")
            report.append(f"最大无功功率变化百分比: {load_comp['max_reactive_power_change_percent']:.2f}%")
            
            report.append("\n受影响节点详情:")
            for node_info in load_comp["changed_nodes"][:5]:  # 只显示前5个
                report.append(f"  - {node_info['node_name']} (节点{node_info['bus_id']}): "
                            f"有功{node_info['active_power_change']:+.2f}kW "
                            f"({node_info['active_power_change_percent']:+.1f}%), "
                            f"无功{node_info['reactive_power_change']:+.2f}kvar "
                            f"({node_info['reactive_power_change_percent']:+.1f}%)")
            
            if len(load_comp["changed_nodes"]) > 5:
                report.append(f"  ... 还有{len(load_comp['changed_nodes']) - 5}个节点受到影响")
        else:
            report.append("无负荷数据变化")
        
        # 光伏出力对比
        report.append("\n2. 光伏出力扰动分析")
        report.append("-" * 30)
        if "error" in pv_comp:
            report.append(f"错误: {pv_comp['error']}")
        elif pv_comp["changed_columns"]:
            report.append(f"受影响的光伏发电机: {', '.join(pv_comp['changed_columns'])}")
            for col in pv_comp["changed_columns"]:
                report.append(f"  - {col}: "
                            f"总能量变化 {pv_comp['total_energy_change'][col]:+.2f}kWh, "
                            f"最大功率变化 {pv_comp['max_power_change_percent'][col]:.2f}%, "
                            f"RMSE {pv_comp['rmse'][col]:.2f}kW")
        else:
            report.append("无光伏出力数据变化")
        
        # 交通拥堵对比
        report.append("\n3. 交通拥堵扰动分析")
        report.append("-" * 30)
        if "error" in traffic_comp:
            report.append(f"错误: {traffic_comp['error']}")
        elif traffic_comp["congestion_level_changed"]:
            report.append(f"最大拥堵系数变化: {traffic_comp['max_congestion_change']:.4f}")
            report.append(f"平均拥堵系数变化: {traffic_comp['avg_congestion_change']:+.4f}")
            report.append(f"拥堵系数RMSE: {traffic_comp['rmse']:.4f}")
        else:
            report.append("无交通拥堵数据变化")
        
        report.append("\n" + "=" * 60)
        
        return "\n".join(report)


class DataPerturbator:
    """数据扰动器
    
    用于对SystemData对象进行各种类型的数据扰动，模拟实际运行中的不确定性
    """
    
    def __init__(self, config: PerturbationConfig = None):
        """
        初始化数据扰动器
        
        Args:
            config: 扰动配置参数，如果为None则使用默认配置
        """
        self.config = config or PerturbationConfig()
        
        # 设置随机种子
        if self.config.random_seed is not None:
            np.random.seed(self.config.random_seed)
        
        # 设置日志
        self._setup_logging()
        
        self.logger.info("数据扰动器初始化完成")
        self.logger.info(f"扰动配置: 负荷扰动={'开启' if self.config.load_perturbation_enabled else '关闭'}, "
                        f"光伏扰动={'开启' if self.config.pv_perturbation_enabled else '关闭'}, "
                        f"交通扰动={'开启' if self.config.traffic_perturbation_enabled else '关闭'}")
    
    def _setup_logging(self):
        """设置日志系统"""
        self.logger = logging.getLogger(f"{__name__}.DataPerturbator")
        
        if self.config.enable_logging:
            # 设置日志级别
            level = getattr(logging, self.config.log_level.upper(), logging.INFO)
            self.logger.setLevel(level)
            
            # 如果没有处理器，添加控制台处理器
            if not self.logger.handlers:
                handler = logging.StreamHandler()
                formatter = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                )
                handler.setFormatter(formatter)
                self.logger.addHandler(handler)
        else:
            self.logger.disabled = True
    
    def perturb_loads(self, loads: Dict[str, Load]) -> Dict[str, Load]:
        """
        对负荷数据进行扰动
        
        Args:
            loads: 原始负荷数据字典
            
        Returns:
            扰动后的负荷数据字典
        """
        if not self.config.load_perturbation_enabled:
            self.logger.info("负荷扰动已禁用，返回原始数据")
            return copy.deepcopy(loads)
        
        perturbed_loads = copy.deepcopy(loads)
        load_names = list(loads.keys())
        
        # 确定受影响的节点数量
        num_affected = int(len(load_names) * self.config.load_affected_node_ratio)
        if num_affected == 0:
            num_affected = 1  # 至少影响一个节点
        
        # 随机选择受影响的节点
        affected_nodes = np.random.choice(load_names, size=num_affected, replace=False)
        
        self.logger.info(f"开始对{num_affected}个负荷节点进行扰动 (总共{len(load_names)}个节点)")
        self.logger.debug(f"受影响节点: {list(affected_nodes)}")
        
        affected_count = 0
        
        for load_name in affected_nodes:
            original_load = loads[load_name]
            perturbed_load = perturbed_loads[load_name]
            
            if self.config.load_perturbation_type == "random_reduction":
                # 随机降低负荷值
                reduction_factor = np.random.uniform(0, self.config.load_reduction_ratio)
                
                new_active = original_load.active_power * (1 - reduction_factor)
                new_reactive = original_load.reactive_power * (1 - reduction_factor)
                
                perturbed_load.active_power = max(0, new_active)
                perturbed_load.reactive_power = max(0, new_reactive)
                
                self.logger.debug(f"节点 {load_name}: 有功功率 {original_load.active_power:.2f} -> {perturbed_load.active_power:.2f} kW "
                                f"(降低 {reduction_factor*100:.1f}%)")
                
            elif self.config.load_perturbation_type == "gaussian_noise":
                # 添加高斯噪声
                active_noise = np.random.normal(0, original_load.active_power * self.config.load_noise_std)
                reactive_noise = np.random.normal(0, original_load.reactive_power * self.config.load_noise_std)
                
                perturbed_load.active_power = max(0, original_load.active_power + active_noise)
                perturbed_load.reactive_power = max(0, original_load.reactive_power + reactive_noise)
                
                self.logger.debug(f"节点 {load_name}: 有功功率添加噪声 {active_noise:+.2f} kW, "
                                f"无功功率添加噪声 {reactive_noise:+.2f} kvar")
            
            affected_count += 1
        
        self.logger.info(f"负荷扰动完成，实际影响了{affected_count}个节点")
        return perturbed_loads
    
    def perturb_pv_profiles(self, load_profiles: pd.DataFrame) -> pd.DataFrame:
        """
        对光伏出力曲线进行扰动
        
        Args:
            load_profiles: 原始光伏出力曲线数据
            
        Returns:
            扰动后的光伏出力曲线数据
        """
        if not self.config.pv_perturbation_enabled:
            self.logger.info("光伏扰动已禁用，返回原始数据")
            return load_profiles.copy()
        
        if load_profiles is None:
            self.logger.warning("光伏出力曲线数据为空，无法进行扰动")
            return load_profiles
        
        perturbed_profiles = load_profiles.copy()
        
        # 找到所有PV列
        pv_columns = [col for col in load_profiles.columns if col.startswith('PV')]
        
        if not pv_columns:
            self.logger.warning("未找到光伏出力数据列（列名应以'PV'开头）")
            return perturbed_profiles
        
        self.logger.info(f"开始对{len(pv_columns)}个光伏发电机进行扰动: {pv_columns}")
        
        for pv_col in pv_columns:
            original_values = load_profiles[pv_col].values
            
            if self.config.pv_noise_type == "gaussian":
                # 高斯噪声扰动
                noise = np.random.normal(0, self.config.pv_noise_std, size=len(original_values))
                # 噪声相对于当前值的百分比
                relative_noise = original_values * noise
                perturbed_values = original_values + relative_noise
                
                self.logger.debug(f"{pv_col}: 添加高斯噪声，标准差为原值的{self.config.pv_noise_std*100:.1f}%")
                
            elif self.config.pv_noise_type == "multiplicative":
                # 乘性扰动
                factors = np.random.uniform(
                    self.config.pv_multiplicative_factor_range[0],
                    self.config.pv_multiplicative_factor_range[1],
                    size=len(original_values)
                )
                perturbed_values = original_values * factors
                
                self.logger.debug(f"{pv_col}: 应用乘性因子，范围 {self.config.pv_multiplicative_factor_range}")
                
            elif self.config.pv_noise_type == "additive":
                # 加性噪声扰动
                noise = np.random.uniform(
                    self.config.pv_additive_noise_range[0],
                    self.config.pv_additive_noise_range[1],
                    size=len(original_values)
                )
                perturbed_values = original_values + noise
                
                self.logger.debug(f"{pv_col}: 添加固定范围噪声 {self.config.pv_additive_noise_range} kW")
            
            else:
                self.logger.warning(f"未知的光伏扰动类型: {self.config.pv_noise_type}")
                continue
            
            # 确保功率值非负
            perturbed_values = np.maximum(0, perturbed_values)
            perturbed_profiles[pv_col] = perturbed_values
            
            # 统计扰动效果
            max_change = np.max(np.abs(perturbed_values - original_values))
            avg_change = np.mean(perturbed_values - original_values)
            self.logger.debug(f"{pv_col}: 最大变化 {max_change:.2f} kW, 平均变化 {avg_change:+.2f} kW")
        
        self.logger.info("光伏出力扰动完成")
        return perturbed_profiles
    
    def perturb_traffic_profiles(self, traffic_profiles: pd.DataFrame) -> pd.DataFrame:
        """
        对交通拥堵曲线进行扰动
        
        Args:
            traffic_profiles: 原始交通拥堵曲线数据
            
        Returns:
            扰动后的交通拥堵曲线数据
        """
        if not self.config.traffic_perturbation_enabled:
            self.logger.info("交通扰动已禁用，返回原始数据")
            return traffic_profiles.copy()
        
        if traffic_profiles is None:
            self.logger.warning("交通拥堵曲线数据为空，无法进行扰动")
            return traffic_profiles
        
        perturbed_profiles = traffic_profiles.copy()
        
        if 'congestion_level' not in traffic_profiles.columns:
            self.logger.warning("未找到交通拥堵数据列'congestion_level'")
            return perturbed_profiles
        
        self.logger.info("开始对交通拥堵曲线进行扰动")
        
        original_values = traffic_profiles['congestion_level'].values
        
        # 添加高斯噪声
        noise = np.random.normal(0, self.config.traffic_noise_std, size=len(original_values))
        
        # 应用乘性调整因子
        factors = np.random.uniform(
            self.config.traffic_congestion_factor_range[0],
            self.config.traffic_congestion_factor_range[1],
            size=len(original_values)
        )
        
        # 组合扰动：先乘以因子，再加噪声
        perturbed_values = (original_values * factors) + noise
        
        # 确保拥堵系数在合理范围内 [0, 1]
        perturbed_values = np.clip(perturbed_values, 0.0, 1.0)
        
        perturbed_profiles['congestion_level'] = perturbed_values
        
        # 统计扰动效果
        max_change = np.max(np.abs(perturbed_values - original_values))
        avg_change = np.mean(perturbed_values - original_values)
        
        self.logger.info(f"交通拥堵扰动完成: 最大变化 {max_change:.4f}, 平均变化 {avg_change:+.4f}")
        
        return perturbed_profiles
    
    def perturb_system_data(self, system_data: SystemData) -> SystemData:
        """
        对完整的系统数据进行扰动
        
        Args:
            system_data: 原始系统数据
            
        Returns:
            扰动后的系统数据（深拷贝）
        """
        self.logger.info("开始对系统数据进行扰动")
        
        # 深拷贝原始数据，避免修改原始数据
        perturbed_data = copy.deepcopy(system_data)
        
        # 扰动负荷数据
        if self.config.load_perturbation_enabled:
            perturbed_data.loads = self.perturb_loads(system_data.loads)
        
        # 扰动光伏出力曲线
        if self.config.pv_perturbation_enabled and system_data.load_profiles is not None:
            perturbed_data.load_profiles = self.perturb_pv_profiles(system_data.load_profiles)
        
        # 扰动交通拥堵曲线
        if self.config.traffic_perturbation_enabled and system_data.traffic_profiles is not None:
            perturbed_data.traffic_profiles = self.perturb_traffic_profiles(system_data.traffic_profiles)
        
        self.logger.info("系统数据扰动完成")
        
        return perturbed_data
    
    def generate_perturbation_report(self, original_data: SystemData, perturbed_data: SystemData, 
                                   output_dir: str = None) -> str:
        """
        生成扰动报告
        
        Args:
            original_data: 原始数据
            perturbed_data: 扰动后数据
            output_dir: 输出目录，如果指定则保存报告文件
            
        Returns:
            报告内容字符串
        """
        self.logger.info("开始生成扰动对比报告")
        
        comparator = DataPerturbationComparator(original_data, perturbed_data)
        report_content = comparator.generate_comparison_report()
        
        # 添加配置信息到报告
        config_info = []
        config_info.append("\n4. 扰动配置参数")
        config_info.append("-" * 30)
        config_info.append(f"负荷扰动: {'开启' if self.config.load_perturbation_enabled else '关闭'}")
        if self.config.load_perturbation_enabled:
            config_info.append(f"  - 扰动类型: {self.config.load_perturbation_type}")
            config_info.append(f"  - 负荷降低比例: {self.config.load_reduction_ratio*100:.1f}%")
            config_info.append(f"  - 受影响节点比例: {self.config.load_affected_node_ratio*100:.1f}%")
        
        config_info.append(f"光伏扰动: {'开启' if self.config.pv_perturbation_enabled else '关闭'}")
        if self.config.pv_perturbation_enabled:
            config_info.append(f"  - 噪声类型: {self.config.pv_noise_type}")
            config_info.append(f"  - 噪声标准差: {self.config.pv_noise_std*100:.1f}%")
        
        config_info.append(f"交通扰动: {'开启' if self.config.traffic_perturbation_enabled else '关闭'}")
        if self.config.traffic_perturbation_enabled:
            config_info.append(f"  - 噪声标准差: {self.config.traffic_noise_std:.4f}")
            config_info.append(f"  - 拥堵因子范围: {self.config.traffic_congestion_factor_range}")
        
        config_info.append(f"随机种子: {self.config.random_seed if self.config.random_seed is not None else '未设置'}")
        
        full_report = report_content + "\n".join(config_info)
        
        # 保存报告文件
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = output_path / f"perturbation_report_{timestamp}.txt"
            
            try:
                with open(report_file, 'w', encoding='utf-8') as f:
                    f.write(full_report)
                
                self.logger.info(f"扰动报告已保存到: {report_file}")
            
            except Exception as e:
                self.logger.error(f"保存扰动报告失败: {e}")
        
        return full_report
    
    def export_perturbed_data(self, perturbed_data: SystemData, output_dir: str, 
                            file_prefix: str = "perturbed") -> Dict[str, str]:
        """
        导出扰动后的数据到文件
        
        Args:
            perturbed_data: 扰动后的数据
            output_dir: 输出目录
            file_prefix: 文件前缀
            
        Returns:
            导出的文件路径字典
        """
        self.logger.info(f"开始将扰动数据导出到: {output_dir}")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        exported_files = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        try:
            # 导出光伏出力曲线
            if perturbed_data.load_profiles is not None:
                pv_file = output_path / f"{file_prefix}_pv_profile_{timestamp}.csv"
                perturbed_data.load_profiles.to_csv(pv_file, index=False)
                exported_files["pv_profile"] = str(pv_file)
                self.logger.debug(f"光伏出力曲线导出到: {pv_file}")
            
            # 导出交通拥堵曲线
            if perturbed_data.traffic_profiles is not None:
                traffic_file = output_path / f"{file_prefix}_traffic_profile_{timestamp}.csv"
                perturbed_data.traffic_profiles.to_csv(traffic_file, index=False)
                exported_files["traffic_profile"] = str(traffic_file)
                self.logger.debug(f"交通拥堵曲线导出到: {traffic_file}")
            
            # 导出负荷数据摘要
            load_summary = []
            for load_name, load_data in perturbed_data.loads.items():
                load_summary.append({
                    "load_name": load_name,
                    "bus_id": load_data.bus_id,
                    "active_power": load_data.active_power,
                    "reactive_power": load_data.reactive_power,
                    "unit_cost": load_data.unit_load_shedding_cost
                })
            
            load_file = output_path / f"{file_prefix}_loads_summary_{timestamp}.json"
            with open(load_file, 'w', encoding='utf-8') as f:
                json.dump(load_summary, f, ensure_ascii=False, indent=2)
            exported_files["loads_summary"] = str(load_file)
            self.logger.debug(f"负荷数据摘要导出到: {load_file}")
            
            self.logger.info(f"扰动数据导出完成，共导出{len(exported_files)}个文件")
            
        except Exception as e:
            self.logger.error(f"导出扰动数据失败: {e}")
            raise
        
        return exported_files


def create_default_perturbation_config() -> PerturbationConfig:
    """创建默认的扰动配置"""
    return PerturbationConfig(
        load_perturbation_enabled=True,
        load_reduction_ratio=0.15,
        load_affected_node_ratio=0.25,
        load_perturbation_type="random_reduction",
        
        pv_perturbation_enabled=True,
        pv_noise_type="gaussian",
        pv_noise_std=0.1,
        
        traffic_perturbation_enabled=True,
        traffic_noise_std=0.05,
        
        random_seed=42,
        enable_logging=True,
        log_level="INFO"
    )


def create_severe_perturbation_config() -> PerturbationConfig:
    """创建严重扰动配置（用于测试系统鲁棒性）"""
    return PerturbationConfig(
        load_perturbation_enabled=True,
        load_reduction_ratio=0.3,
        load_affected_node_ratio=0.5,
        load_perturbation_type="random_reduction",
        
        pv_perturbation_enabled=True,
        pv_noise_type="multiplicative",
        pv_multiplicative_factor_range=(0.6, 1.4),
        
        traffic_perturbation_enabled=True,
        traffic_noise_std=0.1,
        traffic_congestion_factor_range=(0.8, 1.2),
        
        random_seed=None,  # 每次运行使用不同的随机种子
        enable_logging=True,
        log_level="DEBUG"
    )


if __name__ == "__main__":
    # 示例用法
    import sys
    import os
    
    # 添加项目根目录到路径
    sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
    
    from src.datasets.loader import load_system_data
    
    try:
        # 加载原始数据
        print("加载原始系统数据...")
        original_data = load_system_data("data")
        
        # 创建扰动配置
        config = create_default_perturbation_config()
        
        # 创建扰动器
        perturbator = DataPerturbator(config)
        
        # 执行扰动
        print("执行数据扰动...")
        perturbed_data = perturbator.perturb_system_data(original_data)
        
        # 生成对比报告
        print("生成扰动报告...")
        report = perturbator.generate_perturbation_report(
            original_data, perturbed_data, 
            output_dir="output/perturbation_test"
        )
        
        print("扰动测试完成！")
        print("\n" + "="*50)
        print("扰动报告预览:")
        print("="*50)
        print(report[:1000] + "..." if len(report) > 1000 else report)
        
    except Exception as e:
        print(f"扰动测试失败: {e}")
        import traceback
        traceback.print_exc()
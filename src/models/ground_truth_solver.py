"""
基准解生成器 (Ground Truth Solver)
实现"金标准"基准解的生成、验证和保存功能

主要功能：
1. 继承Demo 1的PostDisasterDynamicModel
2. 使用未扰动的原始数据进行求解
3. 生成详细的基准解结果和报告
4. 提供标准化的接口供后续对比分析使用
"""

import json
import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

import cvxpy as cp
import numpy as np
import pandas as pd

from src.datasets.loader import SystemData
from src.models.post_disaster_dynamic_multi_period import PostDisasterDynamicModel

logger = logging.getLogger(__name__)


class GroundTruthSolver(PostDisasterDynamicModel):
    """
    基准解生成器
    
    继承PostDisasterDynamicModel，专门用于生成未扰动情况下的"金标准"基准解
    提供标准化的求解、验证和保存接口
    """
    
    def __init__(self, system_data: SystemData, n_periods: int = 21, 
                 start_hour: int = 3, traffic_profile_path: Optional[str] = None,
                 pv_profile_path: Optional[str] = None, output_dir: Optional[str] = None):
        """
        初始化基准解生成器
        
        Args:
            system_data: 系统数据对象
            n_periods: 时间段数量(默认21个小时: 3:00-23:00)
            start_hour: 起始小时(默认3点)
            traffic_profile_path: 交通拥堵数据路径
            pv_profile_path: 光伏出力数据路径
            output_dir: 输出目录路径
        """
        # 调用父类初始化
        super().__init__(
            system_data=system_data,
            n_periods=n_periods,
            start_hour=start_hour,
            traffic_profile_path=traffic_profile_path,
            pv_profile_path=pv_profile_path
        )
        
        # 设置输出目录
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = Path("output") / f"ground_truth_{timestamp}"
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 基准解相关属性
        self.ground_truth_results = None
        self.solve_status = None
        self.validation_report = None
        
        # 性能优化配置
        self.solver_options = {
            'SCS': {
                'max_iters': 15000,
                'eps': 1e-5,
                'alpha': 1.0,
                'scale': 1.0,
                'adaptive_scale': True,
                'normalize': True,
                'verbose': False
            },
            'MOSEK': {
                'verbose': False,
                'mosek_params': {
                    'MSK_IPAR_INTPNT_MAX_ITERATIONS': 15000,
                    'MSK_DPAR_INTPNT_TOL_PFEAS': 1e-5,
                    'MSK_DPAR_INTPNT_TOL_DFEAS': 1e-5
                }
            },
            'CLARABEL': {
                'max_iter': 15000,
                'tol_feas': 1e-5,
                'tol_gap_abs': 1e-5,
                'tol_gap_rel': 1e-5,
                'verbose': False
            },
            'ECOS': {
                'max_iters': 15000,
                'abstol': 1e-5,
                'reltol': 1e-5,
                'verbose': False
            }
        }
        
        # 配置日志
        self._setup_logging()
        
        logger.info("基准解生成器初始化完成")
        logger.info(f"输出目录: {self.output_dir}")
        logger.info(f"时间段: {self.n_periods} (从{self.start_hour}:00到{self.time_periods[-1]}:00)")
        
    def _setup_logging(self):
        """设置专用日志"""
        log_file = self.output_dir / "ground_truth_solver.log"
        
        # 创建文件处理器
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # 设置格式
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        
        # 添加到logger
        logger.addHandler(file_handler)
    
    def get_optimal_solver_config(self, solver_preference: Optional[str] = None) -> Dict:
        """
        根据问题规模和可用求解器自动选择最优配置
        
        Args:
            solver_preference: 首选求解器 ('SCS', 'MOSEK', 'CLARABEL', 'ECOS')
            
        Returns:
            优化的求解器配置字典
        """
        # 检查可用的求解器
        available_solvers = []
        for solver_name in ['MOSEK', 'CLARABEL', 'SCS', 'ECOS']:
            try:
                import cvxpy as cp
                test_var = cp.Variable()
                test_prob = cp.Problem(cp.Minimize(test_var), [test_var >= 0])
                solver_class = getattr(cp, solver_name, None)
                if solver_class and solver_class.available():
                    available_solvers.append(solver_name)
            except:
                continue
        
        logger.info(f"可用求解器: {available_solvers}")
        
        # 根据问题规模和首选项选择求解器
        problem_size = self.n_periods * self.n_buses * self.n_branches
        
        selected_solver = None
        if solver_preference and solver_preference in available_solvers:
            selected_solver = solver_preference
            logger.info(f"使用指定的求解器: {selected_solver}")
        else:
            # 自动选择策略
            if 'MOSEK' in available_solvers and problem_size > 50000:
                selected_solver = 'MOSEK'  # 大规模问题优选MOSEK
                logger.info("检测到大规模问题，自动选择MOSEK求解器")
            elif 'CLARABEL' in available_solvers:
                selected_solver = 'CLARABEL'  # 中等规模问题优选CLARABEL
                logger.info("自动选择CLARABEL求解器")
            elif 'SCS' in available_solvers:
                selected_solver = 'SCS'  # 默认选择SCS
                logger.info("自动选择SCS求解器")
            elif available_solvers:
                selected_solver = available_solvers[0]
                logger.info(f"使用第一个可用求解器: {selected_solver}")
        
        if not selected_solver:
            logger.warning("没有找到可用的求解器，使用默认配置")
            return {'solver': 'SCS', 'max_iters': 15000, 'eps': 1e-5}
        
        # 获取求解器特定配置
        solver_config = self.solver_options.get(selected_solver, {}).copy()
        solver_config['solver'] = selected_solver
        
        return solver_config
    
    def solve_ground_truth(self, solver_config: Optional[Dict] = None, 
                          verbose: bool = True) -> Optional[Dict]:
        """
        求解基准解 - 生成未扰动情况下的"金标准"解
        
        Args:
            solver_config: 求解器配置参数字典，可包含：
                - max_iters: 最大迭代次数 (默认: 15000)
                - eps: 求解精度 (默认: 1e-5)
                - solver: 求解器类型 ('SCS', 'MOSEK', 'CLARABEL'等)
                - verbose: 求解器详细输出 (默认: False)
            verbose: 是否输出详细的求解过程信息
            
        Returns:
            基准解结果字典，包含目标函数值、求解状态、详细结果等
            如果求解失败返回None
        """
        logger.info("=" * 60)
        logger.info("开始求解基准解 (Ground Truth Solution)")
        logger.info("=" * 60)
        
        # 自动选择最优求解器配置
        solver_preference = solver_config.get('solver') if solver_config else None
        optimal_config = self.get_optimal_solver_config(solver_preference)
        
        # 应用用户自定义配置
        if solver_config:
            optimal_config.update(solver_config)
            logger.info(f"应用自定义配置覆盖: {solver_config}")
        
        logger.info(f"最终求解器配置: solver={optimal_config.get('solver')}, max_iters={optimal_config.get('max_iters', optimal_config.get('max_iter', 'N/A'))}, eps={optimal_config.get('eps', optimal_config.get('tol_feas', 'N/A'))}")
        
        # 记录求解开始时间
        solve_start_time = datetime.now()
        
        try:
            # 记录求解前的模型统计信息
            logger.info(f"模型规模: {self.n_periods}个时段, {self.n_buses}个节点, {self.n_branches}条支路")
            logger.info(f"MESS设备数量: {self.n_mess}")
            
            # 从配置中分离求解器参数
            solve_params = {k: v for k, v in optimal_config.items() if k not in ['verbose', 'solver']}
            
            # 设置求解器类型
            if 'solver' in optimal_config:
                solve_params['solver'] = optimal_config['solver']
            
            # 调用父类求解方法
            logger.info("调用优化求解器...")
            results = self.solve(verbose=verbose, **solve_params)
            
            if results is None:
                logger.error("基准解求解失败")
                self.solve_status = "failed"
                return None
            
            # 记录求解结束时间
            solve_end_time = datetime.now()
            solve_duration = (solve_end_time - solve_start_time).total_seconds()
            
            # 扩展结果信息
            results['solve_start_time'] = solve_start_time.isoformat()
            results['solve_end_time'] = solve_end_time.isoformat()
            results['solve_duration'] = solve_duration
            results['solver_config'] = optimal_config
            results['model_type'] = 'ground_truth'
            results['is_baseline'] = True
            
            # 保存基准解结果
            self.ground_truth_results = results
            self.solve_status = results['status']
            
            logger.info("=" * 40)
            logger.info("✅ 基准解求解成功!")
            logger.info("=" * 40)
            logger.info(f"📊 求解状态: {results['status']}")
            logger.info(f"🔧 使用求解器: {results.get('solver', 'unknown')}")
            logger.info(f"⏱️  求解时间: {solve_duration:.3f} 秒")
            logger.info(f"💰 目标函数值: {results['objective']:.2f} 元")
            logger.info(f"⚡ 总负荷削减: {results.get('total_load_shed', 0):.2f} kW")
            
            # 进行解的验证
            self._validate_solution()
            
            # 保存结果
            self._save_results()
            
            return results
            
        except Exception as e:
            logger.error(f"求解过程出现异常: {e}")
            self.solve_status = "error"
            return None
    
    def _validate_solution(self):
        """验证求解结果的有效性和合理性"""
        logger.info("=" * 40)
        logger.info("🔍 开始验证基准解...")
        logger.info("=" * 40)
        
        if self.ground_truth_results is None:
            logger.warning("⚠️  没有可验证的基准解结果")
            return
        
        validation_results = {
            'timestamp': datetime.now().isoformat(),
            'status': self.ground_truth_results['status'],
            'checks': {}
        }
        
        try:
            # 1. 检查求解状态
            if self.ground_truth_results['status'] in ['optimal', 'optimal_inaccurate']:
                validation_results['checks']['solve_status'] = {'passed': True, 'message': '求解状态正常'}
            else:
                validation_results['checks']['solve_status'] = {
                    'passed': False, 
                    'message': f"求解状态异常: {self.ground_truth_results['status']}"
                }
            
            # 2. 检查目标函数值
            objective_value = self.ground_truth_results.get('objective', float('inf'))
            if np.isfinite(objective_value) and objective_value >= 0:
                validation_results['checks']['objective_value'] = {
                    'passed': True, 
                    'value': objective_value,
                    'message': '目标函数值有效'
                }
            else:
                validation_results['checks']['objective_value'] = {
                    'passed': False,
                    'value': objective_value,
                    'message': '目标函数值无效'
                }
            
            # 3. 检查负荷削减的合理性
            total_load_shed = self.ground_truth_results.get('total_load_shed', 0)
            total_load = sum(self.P_load) * 1000 * self.n_periods  # 转换为kW
            load_shed_ratio = total_load_shed / total_load if total_load > 0 else 0
            
            validation_results['checks']['load_shedding'] = {
                'passed': True,
                'total_shed_kw': total_load_shed,
                'total_load_kw': total_load,
                'shed_ratio': load_shed_ratio,
                'message': f'负荷削减比例: {load_shed_ratio:.1%}'
            }
            
            # 4. 检查MESS调度的连续性
            mess_schedule = self.ground_truth_results.get('mess_schedule', {})
            mess_continuity_check = self._check_mess_schedule_continuity(mess_schedule)
            validation_results['checks']['mess_schedule'] = mess_continuity_check
            
            # 5. 检查电压范围
            voltage_check = self._check_voltage_constraints()
            validation_results['checks']['voltage_constraints'] = voltage_check
            
            # 6. 检查功率平衡
            power_balance_check = self._check_power_balance()
            validation_results['checks']['power_balance'] = power_balance_check
            
            # 7. 检查MESS能量平衡
            mess_energy_check = self._check_mess_energy_balance()
            validation_results['checks']['mess_energy_balance'] = mess_energy_check
            
            # 8. 检查求解器收敛性
            convergence_check = self._check_solver_convergence()
            validation_results['checks']['solver_convergence'] = convergence_check
            
            # 9. 统计验证结果
            passed_checks = sum(1 for check in validation_results['checks'].values() 
                              if check.get('passed', False))
            total_checks = len(validation_results['checks'])
            
            validation_results['summary'] = {
                'total_checks': total_checks,
                'passed_checks': passed_checks,
                'validation_score': passed_checks / total_checks if total_checks > 0 else 0,
                'overall_status': 'PASSED' if passed_checks == total_checks else 'PARTIAL'
            }
            
            self.validation_report = validation_results
            
            logger.info("=" * 40)
            logger.info(f"✅ 基准解验证完成: {passed_checks}/{total_checks} 项检查通过")
            logger.info(f"📊 验证得分: {validation_results['summary']['validation_score']:.1%}")
            logger.info(f"🏆 总体状态: {validation_results['summary']['overall_status']}")
            logger.info("=" * 40)
            
        except Exception as e:
            logger.error(f"验证过程出现异常: {e}")
            validation_results['error'] = str(e)
            self.validation_report = validation_results
    
    def _check_mess_schedule_continuity(self, mess_schedule: Dict) -> Dict:
        """检查MESS调度的连续性"""
        try:
            continuity_issues = []
            
            for mess_id in range(1, self.n_mess + 1):
                locations = []
                soc_values = []
                
                for hour in self.time_periods:
                    if str(hour) in mess_schedule and mess_id in mess_schedule[str(hour)]:
                        schedule = mess_schedule[str(hour)][mess_id]
                        locations.append(schedule.get('location'))
                        soc_values.append(schedule.get('SOC', 0))
                
                # 检查SOC变化的合理性
                for i in range(1, len(soc_values)):
                    soc_change = abs(soc_values[i] - soc_values[i-1])
                    if soc_change > 50:  # SOC变化超过50%
                        continuity_issues.append(f"MESS{mess_id}在时段{self.time_periods[i]}SOC变化异常: {soc_change:.1f}%")
            
            return {
                'passed': len(continuity_issues) == 0,
                'issues': continuity_issues,
                'message': '调度连续性正常' if len(continuity_issues) == 0 else f'发现{len(continuity_issues)}个连续性问题'
            }
            
        except Exception as e:
            return {
                'passed': False,
                'error': str(e),
                'message': '调度连续性检查失败'
            }
    
    def _check_voltage_constraints(self) -> Dict:
        """检查电压约束"""
        try:
            if self.ground_truth_results is None:
                return {'passed': False, 'message': '无解可检查'}
            
            voltage_violations = []
            voltage_time = self.ground_truth_results.get('voltage_time', {})
            
            for hour_str, voltages in voltage_time.items():
                for bus_id, voltage in voltages.items():
                    if voltage < self.V_base * np.sqrt(self.V_min_sqr):
                        voltage_violations.append(f"时段{hour_str}节点{bus_id}电压过低: {voltage:.3f}kV")
                    elif voltage > self.V_base * np.sqrt(self.V_max_sqr):
                        voltage_violations.append(f"时段{hour_str}节点{bus_id}电压过高: {voltage:.3f}kV")
            
            return {
                'passed': len(voltage_violations) == 0,
                'violations': voltage_violations,
                'message': '电压约束满足' if len(voltage_violations) == 0 else f'发现{len(voltage_violations)}个电压违约'
            }
            
        except Exception as e:
            return {
                'passed': False,
                'error': str(e),
                'message': '电压约束检查失败'
            }
    
    def _check_power_balance(self) -> Dict:
        """检查系统功率平衡"""
        try:
            if self.ground_truth_results is None:
                return {'passed': False, 'message': '无解可检查'}
            
            power_balance_violations = []
            P_gen_time = self.ground_truth_results.get('P_gen_time', {})
            P_shed_time = self.ground_truth_results.get('P_shed_time', {})
            mess_schedule = self.ground_truth_results.get('mess_schedule', {})
            
            for hour_str in P_gen_time.keys():
                hour = int(hour_str)
                
                # 计算总发电量
                total_gen = sum(P_gen_time[hour_str].values())
                
                # 计算总负荷削减
                total_shed = sum(P_shed_time.get(hour_str, {}).values())
                
                # 计算MESS净功率
                mess_net_power = 0
                if hour_str in mess_schedule:
                    for mess_data in mess_schedule[hour_str].values():
                        mess_net_power += mess_data.get('P_dch', 0) - mess_data.get('P_ch', 0)
                
                # 计算原始负荷
                hour_idx = hour - self.start_hour
                if 0 <= hour_idx < len(self.P_load):
                    original_load = sum(self.P_load) * 1000  # 转换为kW
                    
                    # 功率平衡检查: 发电 + MESS放电 = 原始负荷 - 负荷削减
                    power_supply = total_gen + mess_net_power
                    power_demand = original_load - total_shed
                    
                    balance_error = abs(power_supply - power_demand)
                    tolerance = max(1.0, original_load * 0.001)  # 0.1%容差
                    
                    if balance_error > tolerance:
                        power_balance_violations.append(
                            f"时段{hour}功率不平衡: 供给={power_supply:.2f}kW, 需求={power_demand:.2f}kW, 误差={balance_error:.2f}kW"
                        )
            
            return {
                'passed': len(power_balance_violations) == 0,
                'violations': power_balance_violations,
                'message': '功率平衡满足' if len(power_balance_violations) == 0 else f'发现{len(power_balance_violations)}个功率不平衡'
            }
            
        except Exception as e:
            return {
                'passed': False,
                'error': str(e),
                'message': '功率平衡检查失败'
            }
    
    def _check_mess_energy_balance(self) -> Dict:
        """检查MESS能量平衡约束"""
        try:
            if self.ground_truth_results is None:
                return {'passed': False, 'message': '无解可检查'}
            
            energy_violations = []
            mess_schedule = self.ground_truth_results.get('mess_schedule', {})
            
            for mess_id in range(1, self.n_mess + 1):
                previous_soc = None
                
                for hour in self.time_periods:
                    if str(hour) in mess_schedule and mess_id in mess_schedule[str(hour)]:
                        schedule = mess_schedule[str(hour)][mess_id]
                        current_soc = schedule.get('SOC', 0)
                        p_ch = schedule.get('P_ch', 0)
                        p_dch = schedule.get('P_dch', 0)
                        
                        # 检查SOC范围
                        if current_soc < 0 or current_soc > 100:
                            energy_violations.append(f"MESS{mess_id}在时段{hour}SOC超出范围: {current_soc:.1f}%")
                        
                        # 检查充放电不能同时进行
                        if p_ch > 0 and p_dch > 0:
                            energy_violations.append(f"MESS{mess_id}在时段{hour}同时充放电: 充电{p_ch:.1f}kW, 放电{p_dch:.1f}kW")
                        
                        # 检查能量平衡 (如果有前一时段数据)
                        if previous_soc is not None:
                            # 简化的能量平衡检查 (假设容量为100kWh)
                            capacity = 100  # kWh
                            energy_change = (p_ch * 0.95 - p_dch / 0.95) * 1  # 1小时，考虑效率
                            expected_soc = previous_soc + (energy_change / capacity) * 100
                            
                            soc_error = abs(current_soc - expected_soc)
                            if soc_error > 5:  # 5%容差
                                energy_violations.append(
                                    f"MESS{mess_id}在时段{hour}能量不平衡: 实际SOC={current_soc:.1f}%, 预期SOC={expected_soc:.1f}%"
                                )
                        
                        previous_soc = current_soc
            
            return {
                'passed': len(energy_violations) == 0,
                'violations': energy_violations,
                'message': 'MESS能量平衡正常' if len(energy_violations) == 0 else f'发现{len(energy_violations)}个能量平衡问题'
            }
            
        except Exception as e:
            return {
                'passed': False,
                'error': str(e),
                'message': 'MESS能量平衡检查失败'
            }
    
    def _check_solver_convergence(self) -> Dict:
        """检查求解器收敛性"""
        try:
            if self.ground_truth_results is None:
                return {'passed': False, 'message': '无解可检查'}
            
            solve_status = self.ground_truth_results.get('status', 'unknown')
            solve_time = self.ground_truth_results.get('solve_time', 0)
            
            # 检查求解状态
            good_statuses = ['optimal', 'optimal_inaccurate']
            status_ok = solve_status in good_statuses
            
            # 检查求解时间是否合理 (不超过30分钟)
            time_reasonable = solve_time < 1800
            
            # 检查目标函数值是否有限
            objective = self.ground_truth_results.get('objective', float('inf'))
            objective_finite = np.isfinite(objective)
            
            convergence_issues = []
            if not status_ok:
                convergence_issues.append(f"求解状态异常: {solve_status}")
            if not time_reasonable:
                convergence_issues.append(f"求解时间过长: {solve_time:.1f}秒")
            if not objective_finite:
                convergence_issues.append(f"目标函数值无效: {objective}")
            
            return {
                'passed': len(convergence_issues) == 0,
                'issues': convergence_issues,
                'solve_status': solve_status,
                'solve_time': solve_time,
                'objective_finite': objective_finite,
                'message': '求解器收敛正常' if len(convergence_issues) == 0 else f'发现{len(convergence_issues)}个收敛问题'
            }
            
        except Exception as e:
            return {
                'passed': False,
                'error': str(e),
                'message': '收敛性检查失败'
            }
    
    def _save_results(self):
        """保存基准解结果到文件系统"""
        logger.info("=" * 40)
        logger.info("💾 保存基准解结果...")
        logger.info("=" * 40)
        
        try:
            # 1. 保存JSON格式的详细结果
            results_file = self.output_dir / "ground_truth_results.json"
            serializable_results = self._make_serializable(self.ground_truth_results)
            
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_results, f, ensure_ascii=False, indent=2)
            
            # 2. 保存验证报告
            if self.validation_report:
                validation_file = self.output_dir / "validation_report.json"
                with open(validation_file, 'w', encoding='utf-8') as f:
                    json.dump(self.validation_report, f, ensure_ascii=False, indent=2)
            
            # 3. 保存Pickle格式的完整对象（用于后续分析）
            pickle_file = self.output_dir / "ground_truth_solver.pkl"
            with open(pickle_file, 'wb') as f:
                pickle.dump({
                    'solver': self,
                    'results': self.ground_truth_results,
                    'validation': self.validation_report
                }, f)
            
            # 4. 生成文本报告
            self._generate_text_report()
            
            # 5. 保存模型LP文件
            lp_file = self.output_dir / "ground_truth_model.lp"
            self.write_lp(str(lp_file))
            
            logger.info("=" * 40)
            logger.info("✅ 基准解结果保存完成!")
            logger.info("=" * 40)
            logger.info(f"📁 结果目录: {self.output_dir}")
            logger.info("📄 生成的文件:")
            logger.info("   • ground_truth_results.json    - 详细求解结果")
            logger.info("   • validation_report.json       - 验证报告")
            logger.info("   • ground_truth_solver.pkl      - 完整求解器对象")
            logger.info("   • ground_truth_model.lp        - 模型描述文件")
            logger.info("   • ground_truth_report.txt      - 文本报告")
            logger.info("   • ground_truth_solver.log      - 求解日志")
            logger.info("=" * 40)
            
        except Exception as e:
            logger.error(f"保存结果失败: {e}")
    
    def _make_serializable(self, obj: Any) -> Any:
        """将对象转换为JSON可序列化格式"""
        if isinstance(obj, dict):
            return {key: self._make_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, (int, float, str, bool)) or obj is None:
            return obj
        else:
            return str(obj)
    
    def _generate_text_report(self):
        """生成文本格式的基准解报告"""
        report_file = self.output_dir / "ground_truth_report.txt"
        
        try:
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write("="*80 + "\n")
                f.write("基准解生成器 - 详细报告\n")
                f.write("Ground Truth Solver - Detailed Report\n")
                f.write("="*80 + "\n\n")
                
                # 基本信息
                f.write("1. 基本信息\n")
                f.write("-" * 40 + "\n")
                if self.ground_truth_results:
                    f.write(f"求解状态: {self.ground_truth_results.get('status', 'unknown')}\n")
                    f.write(f"求解器: {self.ground_truth_results.get('solver', 'unknown')}\n")
                    f.write(f"求解时间: {self.ground_truth_results.get('solve_time', 0):.3f} 秒\n")
                    f.write(f"目标函数值: {self.ground_truth_results.get('objective', 0):.2f} 元\n")
                    f.write(f"总求解时长: {self.ground_truth_results.get('solve_duration', 0):.3f} 秒\n")
                f.write(f"仿真时间段: {self.n_periods} 小时 ({self.start_hour}:00-{self.time_periods[-1]}:00)\n")
                f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                # 关键指标
                f.write("2. 关键指标\n")
                f.write("-" * 40 + "\n")
                if self.ground_truth_results:
                    f.write(f"总负荷削减: {self.ground_truth_results.get('total_load_shed', 0):.2f} kW\n")
                    
                    # 计算负荷削减比例
                    total_system_load = sum(self.P_load) * 1000 * self.n_periods
                    load_shed_ratio = self.ground_truth_results.get('total_load_shed', 0) / total_system_load
                    f.write(f"负荷削减比例: {load_shed_ratio:.2%}\n")
                    
                    # MESS调度统计
                    mess_schedule = self.ground_truth_results.get('mess_schedule', {})
                    if mess_schedule:
                        total_mess_energy = 0
                        for hour_schedule in mess_schedule.values():
                            for mess_data in hour_schedule.values():
                                total_mess_energy += mess_data.get('P_dch', 0) - mess_data.get('P_ch', 0)
                        f.write(f"MESS总净放电量: {total_mess_energy:.2f} kWh\n")
                f.write("\n")
                
                # 验证结果
                f.write("3. 验证结果\n")
                f.write("-" * 40 + "\n")
                if self.validation_report:
                    summary = self.validation_report.get('summary', {})
                    f.write(f"验证状态: {summary.get('overall_status', 'unknown')}\n")
                    f.write(f"通过检查: {summary.get('passed_checks', 0)}/{summary.get('total_checks', 0)}\n")
                    f.write(f"验证得分: {summary.get('validation_score', 0):.1%}\n\n")
                    
                    # 详细检查结果
                    checks = self.validation_report.get('checks', {})
                    for check_name, check_result in checks.items():
                        status = "✓" if check_result.get('passed', False) else "✗"
                        message = check_result.get('message', 'No message')
                        f.write(f"  {status} {check_name}: {message}\n")
                f.write("\n")
                
                # MESS调度摘要
                f.write("4. MESS调度摘要\n")
                f.write("-" * 40 + "\n")
                if self.ground_truth_results and 'mess_schedule' in self.ground_truth_results:
                    mess_schedule = self.ground_truth_results['mess_schedule']
                    
                    # 显示前10个时段的调度
                    display_hours = self.time_periods[:10]
                    for hour in display_hours:
                        f.write(f"时段 {hour}:00\n")
                        if str(hour) in mess_schedule:
                            for mess_id, schedule in mess_schedule[str(hour)].items():
                                location = schedule.get('location', '移动中')
                                soc = schedule.get('SOC', 0)
                                p_ch = schedule.get('P_ch', 0)
                                p_dch = schedule.get('P_dch', 0)
                                
                                action = "待机"
                                if p_ch > 0:
                                    action = f"充电 {p_ch:.1f}kW"
                                elif p_dch > 0:
                                    action = f"放电 {p_dch:.1f}kW"
                                
                                f.write(f"  MESS{mess_id}: 位置=节点{location}, SOC={soc:.1f}%, {action}\n")
                        f.write("\n")
                
                # 文件列表
                f.write("5. 输出文件列表\n")
                f.write("-" * 40 + "\n")
                f.write("- ground_truth_results.json    # 详细求解结果\n")
                f.write("- validation_report.json       # 验证报告\n")
                f.write("- ground_truth_solver.pkl      # 完整求解器对象\n")
                f.write("- ground_truth_model.lp        # 模型描述文件\n")
                f.write("- ground_truth_report.txt      # 本报告文件\n")
                f.write("- ground_truth_solver.log      # 求解日志\n\n")
                
                f.write("="*80 + "\n")
                f.write("报告生成完成\n")
                
            logger.info(f"文本报告已生成: {report_file}")
            
        except Exception as e:
            logger.error(f"生成文本报告失败: {e}")
    
    def get_baseline_metrics(self) -> Dict:
        """
        获取基准解的关键指标
        
        Returns:
            包含基准指标的字典
        """
        if self.ground_truth_results is None:
            logger.warning("没有可用的基准解结果")
            return {}
        
        metrics = {
            'objective_value': self.ground_truth_results.get('objective', 0),
            'total_load_shed': self.ground_truth_results.get('total_load_shed', 0),
            'solve_time': self.ground_truth_results.get('solve_time', 0),
            'solve_status': self.ground_truth_results.get('status', 'unknown'),
            'validation_score': 0,
            'n_periods': self.n_periods,
            'time_range': f"{self.start_hour}:00-{self.time_periods[-1]}:00"
        }
        
        # 添加验证得分
        if self.validation_report:
            summary = self.validation_report.get('summary', {})
            metrics['validation_score'] = summary.get('validation_score', 0)
        
        # 计算负荷削减率
        total_system_load = sum(self.P_load) * 1000 * self.n_periods
        if total_system_load > 0:
            metrics['load_shed_ratio'] = metrics['total_load_shed'] / total_system_load
        else:
            metrics['load_shed_ratio'] = 0
        
        return metrics
    
    def export_for_comparison(self, export_path: Optional[str] = None) -> str:
        """
        导出用于对比分析的标准格式数据
        
        Args:
            export_path: 导出文件路径，如果为None则使用默认路径
            
        Returns:
            导出文件的路径
        """
        if export_path is None:
            export_path = self.output_dir / "ground_truth_for_comparison.json"
        else:
            export_path = Path(export_path)
        
        if self.ground_truth_results is None:
            logger.error("没有可导出的基准解结果")
            return ""
        
        try:
            # 构建标准格式的对比数据
            comparison_data = {
                'metadata': {
                    'type': 'ground_truth',
                    'generated_at': datetime.now().isoformat(),
                    'model_version': 'PostDisasterDynamicModel_v1.0',
                    'n_periods': self.n_periods,
                    'time_range': f"{self.start_hour}:00-{self.time_periods[-1]}:00"
                },
                'metrics': self.get_baseline_metrics(),
                'results': {
                    'objective_value': self.ground_truth_results.get('objective'),
                    'solve_status': self.ground_truth_results.get('status'),
                    'solve_time': self.ground_truth_results.get('solve_time'),
                    'load_shedding': self.ground_truth_results.get('P_shed_time'),
                    'mess_schedule': self.ground_truth_results.get('mess_schedule'),
                    'voltage_profile': self.ground_truth_results.get('voltage_time')
                },
                'validation': self.validation_report
            }
            
            # 序列化并保存
            serializable_data = self._make_serializable(comparison_data)
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"对比数据已导出到: {export_path}")
            return str(export_path)
            
        except Exception as e:
            logger.error(f"导出对比数据失败: {e}")
            return ""
    
    @classmethod
    def load_from_file(cls, file_path: str) -> 'GroundTruthSolver':
        """
        从文件加载已保存的基准解生成器
        
        Args:
            file_path: pickle文件路径
            
        Returns:
            GroundTruthSolver实例
        """
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            
            solver = data['solver']
            solver.ground_truth_results = data['results']
            solver.validation_report = data['validation']
            
            logger.info(f"基准解生成器已从文件加载: {file_path}")
            return solver
            
        except Exception as e:
            logger.error(f"从文件加载失败: {e}")
            raise
    
    def compare_with_solution(self, other_results: Dict) -> Dict:
        """
        将基准解与其他解进行对比
        
        Args:
            other_results: 其他求解结果
            
        Returns:
            对比分析结果
        """
        if self.ground_truth_results is None:
            logger.error("没有可用的基准解进行对比")
            return {}
        
        try:
            comparison = {
                'timestamp': datetime.now().isoformat(),
                'baseline_metrics': self.get_baseline_metrics(),
                'comparison_metrics': {},
                'differences': {}
            }
            
            # 目标函数值对比
            baseline_obj = self.ground_truth_results.get('objective', 0)
            other_obj = other_results.get('objective', 0)
            
            comparison['differences']['objective'] = {
                'baseline': baseline_obj,
                'compared': other_obj,
                'absolute_diff': other_obj - baseline_obj,
                'relative_diff': (other_obj - baseline_obj) / baseline_obj if baseline_obj != 0 else float('inf')
            }
            
            # 负荷削减对比
            baseline_shed = self.ground_truth_results.get('total_load_shed', 0)
            other_shed = other_results.get('total_load_shed', 0)
            
            comparison['differences']['load_shedding'] = {
                'baseline': baseline_shed,
                'compared': other_shed,
                'absolute_diff': other_shed - baseline_shed,
                'relative_diff': (other_shed - baseline_shed) / baseline_shed if baseline_shed != 0 else float('inf')
            }
            
            # 求解时间对比
            baseline_time = self.ground_truth_results.get('solve_time', 0)
            other_time = other_results.get('solve_time', 0)
            
            comparison['differences']['solve_time'] = {
                'baseline': baseline_time,
                'compared': other_time,
                'absolute_diff': other_time - baseline_time,
                'relative_diff': (other_time - baseline_time) / baseline_time if baseline_time != 0 else float('inf')
            }
            
            return comparison
            
        except Exception as e:
            logger.error(f"对比分析失败: {e}")
            return {'error': str(e)}


def create_ground_truth_solver(system_data: SystemData, 
                             config: Optional[Dict] = None) -> GroundTruthSolver:
    """
    便捷函数：创建基准解生成器实例
    
    Args:
        system_data: 系统数据
        config: 配置参数字典
        
    Returns:
        GroundTruthSolver实例
    """
    default_config = {
        'n_periods': 21,
        'start_hour': 3,
        'traffic_profile_path': None,
        'pv_profile_path': None,
        'output_dir': None
    }
    
    if config:
        default_config.update(config)
    
    return GroundTruthSolver(
        system_data=system_data,
        **default_config
    )


if __name__ == "__main__":
    """测试基准解生成器"""
    import sys
    from pathlib import Path
    
    # 添加项目路径
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))
    
    try:
        from src.datasets.loader import load_system_data
    except ImportError:
        # 如果相对导入失败，尝试绝对导入
        sys.path.append(str(project_root / "src"))
        from datasets.loader import load_system_data
    
    try:
        # 设置日志
        logging.basicConfig(level=logging.INFO)
        
        # 加载数据
        print("加载系统数据...")
        data_dir = project_root / "data"
        system_data = load_system_data(str(data_dir))
        
        # 创建基准解生成器
        print("创建基准解生成器...")
        config = {
            'n_periods': 21,
            'start_hour': 3,
            'traffic_profile_path': str(data_dir / "traffic_profile.csv"),
            'pv_profile_path': str(data_dir / "pv_profile.csv")
        }
        
        solver = create_ground_truth_solver(system_data, config)
        
        # 求解基准解
        print("求解基准解...")
        results = solver.solve_ground_truth(verbose=True)
        
        if results:
            print(f"✅ 基准解求解成功!")
            print(f"📊 目标函数值: {results['objective']:.2f} 元")
            print(f"⚡ 总负荷削减: {results['total_load_shed']:.2f} kW")
            print(f"⏱️  求解时间: {results['solve_time']:.3f} 秒")
            print(f"📁 结果保存在: {solver.output_dir}")
            
            # 导出对比数据
            export_path = solver.export_for_comparison()
            print(f"📄 对比数据已导出: {export_path}")
            
        else:
            print("❌ 基准解求解失败")
            
    except Exception as e:
        print(f"❌ 程序执行出错: {e}")
        import traceback
        traceback.print_exc()
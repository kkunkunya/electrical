"""
灾后静态调度模型 V2 (Demo 2) - 引入固定MESS和DEG
基于CVXPY实现的MISOCP最优负荷削减模型
"""

import cvxpy as cp
import numpy as np
from typing import Dict, List, Tuple, Optional
from src.datasets.loader import SystemData
import logging

logger = logging.getLogger(__name__)


class PostDisasterStaticModelV2:
    """灾后静态调度MISOCP模型 V2 - 包含固定MESS和DEG"""
    
    def __init__(self, system_data: SystemData):
        """
        初始化模型
        
        Args:
            system_data: 系统数据对象
        """
        self.data = system_data
        
        # 提取数据
        self._extract_data()
        
        # 创建优化问题
        self._build_model()
        
    def _extract_data(self):
        """从系统数据中提取所需参数"""
        # 节点集合
        self.n_buses = len(self.data.loads) + 1  # 33个节点
        self.nodes = list(range(1, self.n_buses + 1))
        
        # 支路集合
        branch_list = list(self.data.branches.values()) if isinstance(self.data.branches, dict) else self.data.branches
        self.branches = [(b.from_bus, b.to_bus) for b in branch_list]
        self.n_branches = len(self.branches)
        
        # 构建节点-支路关联矩阵
        self.A = np.zeros((self.n_buses, self.n_branches))
        for k, (i, j) in enumerate(self.branches):
            self.A[i-1, k] = 1   # 流出节点i
            self.A[j-1, k] = -1  # 流入节点j
            
        # 负荷数据（MW）
        self.P_load = np.zeros(self.n_buses)
        self.Q_load = np.zeros(self.n_buses)
        self.load_cost = np.ones(self.n_buses)  # 默认成本为1
        
        load_list = list(self.data.loads.values()) if isinstance(self.data.loads, dict) else self.data.loads
        for load in load_list:
            bus_idx = load.bus_id - 1
            self.P_load[bus_idx] = load.active_power / 1000  # kW转MW
            self.Q_load[bus_idx] = load.reactive_power / 1000  # kvar转Mvar
            self.load_cost[bus_idx] = load.unit_load_shedding_cost
            
        # 支路参数
        self.R = np.zeros(self.n_branches)
        self.X = np.zeros(self.n_branches)
        self.I_max = np.zeros(self.n_branches)
        
        for k, branch in enumerate(branch_list):
            self.R[k] = branch.resistance
            self.X[k] = branch.reactance
            self.I_max[k] = branch.capacity / 1000  # A转kA
            
        # 系统参数
        params = self.data.system_parameters
        self.V_min_sqr = (params.voltage_lower_limit / params.nominal_voltage) ** 2
        self.V_max_sqr = (params.voltage_upper_limit / params.nominal_voltage) ** 2
        self.V_base = params.nominal_voltage  # kV
        
        # 初始化各类DG参数
        self.P_deg_max = np.zeros(self.n_buses)
        self.Q_deg_max = np.zeros(self.n_buses)
        self.P_mess_max = np.zeros(self.n_buses)
        self.Q_mess_max = np.zeros(self.n_buses)
        self.E_mess_cap = np.zeros(self.n_buses)  # 储能容量
        self.eta_mess = np.zeros(self.n_buses)    # 充放电效率
        self.P_evs_max = np.zeros(self.n_buses)
        self.Q_evs_max = np.zeros(self.n_buses)
        self.P_pv_max = np.zeros(self.n_buses)
        
        # 分配各类发电机到节点（基于表B1的分布）
        generator_bus_map = {
            # DEG分布
            'DEG1': 6,   # 节点6
            'DEG2': 12,  # 节点12
            'DEG3': 18,  # 节点18
            'DEG4': 25,  # 节点25
            'DEG5': 30,  # 节点30
            # MESS分布
            'MESS': 15,  # 节点15（固定位置）
            # EVS分布
            'EVS1': 10,  # 节点10
            'EVS2': 20,  # 节点20
            'EVS3': 28,  # 节点28
            # PV分布
            'PV1': 8,    # 节点8
            'PV2': 14,   # 节点14
            'PV3': 22,   # 节点22
            'PV4': 26,   # 节点26
            'PV5': 32,   # 节点32
        }
        
        # 处理各类发电机
        gen_dict = self.data.generators if isinstance(self.data.generators, dict) else {}
        
        for gen_name, gen in gen_dict.items():
            if gen_name in generator_bus_map:
                bus_id = generator_bus_map[gen_name]
                bus_idx = bus_id - 1
                
                if gen.type == 'DEG':
                    self.P_deg_max[bus_idx] = gen.active_power_max / 1000 if gen.active_power_max else 0
                    self.Q_deg_max[bus_idx] = gen.reactive_power_max / 1000 if gen.reactive_power_max else 0
                    
                elif gen.type == 'MESS':
                    self.P_mess_max[bus_idx] = gen.active_power_max / 1000 if gen.active_power_max else 0
                    self.Q_mess_max[bus_idx] = gen.reactive_power_max / 1000 if gen.reactive_power_max else 0
                    self.E_mess_cap[bus_idx] = gen.storage_capacity / 1000 if gen.storage_capacity else 0
                    self.eta_mess[bus_idx] = gen.charging_efficiency if gen.charging_efficiency else 0.98
                    
                elif gen.type == 'EVS':
                    self.P_evs_max[bus_idx] = gen.active_power_max / 1000 if gen.active_power_max else 0
                    self.Q_evs_max[bus_idx] = gen.reactive_power_max / 1000 if gen.reactive_power_max else 0
                    
                elif gen.type == 'PV':
                    self.P_pv_max[bus_idx] = gen.predicted_power / 1000 if hasattr(gen, 'predicted_power') and gen.predicted_power else 0
                    
        # DG调度成本（元/MW）
        self.deg_cost = 50    # DEG发电成本
        self.mess_cost = 30   # MESS放电成本
        self.evs_cost = 40    # EVS放电成本
        
    def _build_model(self):
        """构建CVXPY优化模型"""
        # 决策变量
        # 负荷削减
        self.P_shed = cp.Variable(self.n_buses, nonneg=True, name="P_shed")
        self.Q_shed = cp.Variable(self.n_buses, nonneg=True, name="Q_shed")
        
        # 各类DG输出（分开处理）
        self.P_deg = cp.Variable(self.n_buses, nonneg=True, name="P_deg")
        self.Q_deg = cp.Variable(self.n_buses, name="Q_deg")
        
        self.P_mess = cp.Variable(self.n_buses, name="P_mess")  # 可正可负（充放电）
        self.Q_mess = cp.Variable(self.n_buses, name="Q_mess")
        
        self.P_evs = cp.Variable(self.n_buses, nonneg=True, name="P_evs")
        self.Q_evs = cp.Variable(self.n_buses, name="Q_evs")
        
        self.P_pv = cp.Variable(self.n_buses, nonneg=True, name="P_pv")
        
        # 支路功率
        self.P_branch = cp.Variable(self.n_branches, name="P_branch")
        self.Q_branch = cp.Variable(self.n_branches, name="Q_branch")
        
        # 节点电压平方
        self.V_sqr = cp.Variable(self.n_buses, nonneg=True, name="V_sqr")
        
        # 支路电流平方
        self.I_sqr = cp.Variable(self.n_branches, nonneg=True, name="I_sqr")
        
        # 约束列表
        constraints = []
        
        # 1. 负荷削减约束 (8)-(9)
        constraints.append(self.P_shed <= self.P_load)
        
        for i in range(self.n_buses):
            if self.P_load[i] > 1e-6:
                constraints.append(
                    self.Q_shed[i] == (self.Q_load[i] / self.P_load[i]) * self.P_shed[i]
                )
            else:
                constraints.append(self.Q_shed[i] == 0)
                
        # 2. DEG约束 (10)-(12)
        constraints.append(self.P_deg <= self.P_deg_max)
        constraints.append(self.Q_deg <= 0.33 * self.P_deg)   # 功率因数约束
        constraints.append(self.Q_deg >= -0.33 * self.P_deg)
        
        # 只有配置DEG的节点才能发电
        for i in range(self.n_buses):
            if self.P_deg_max[i] == 0:
                constraints.append(self.P_deg[i] == 0)
                constraints.append(self.Q_deg[i] == 0)
                
        # 3. MESS约束
        # 充放电功率限制
        constraints.append(self.P_mess <= self.P_mess_max)
        constraints.append(self.P_mess >= -self.P_mess_max)  # 充电为负
        
        # 无功功率约束（需要线性化abs函数）
        # 引入辅助变量表示|P_mess|
        P_mess_abs = cp.Variable(self.n_buses, nonneg=True, name="P_mess_abs")
        constraints.append(P_mess_abs >= self.P_mess)
        constraints.append(P_mess_abs >= -self.P_mess)
        
        constraints.append(self.Q_mess <= 0.33 * P_mess_abs)
        constraints.append(self.Q_mess >= -0.33 * P_mess_abs)
        
        # 只有配置MESS的节点才能充放电
        for i in range(self.n_buses):
            if self.P_mess_max[i] == 0:
                constraints.append(self.P_mess[i] == 0)
                constraints.append(self.Q_mess[i] == 0)
                
        # 4. EVS约束（Demo 2暂不考虑EVS调度）
        for i in range(self.n_buses):
            constraints.append(self.P_evs[i] == 0)
            constraints.append(self.Q_evs[i] == 0)
            
        # 5. PV约束（固定出力）
        constraints.append(self.P_pv <= self.P_pv_max)
        for i in range(self.n_buses):
            if self.P_pv_max[i] == 0:
                constraints.append(self.P_pv[i] == 0)
                
        # 6. 功率平衡约束 (13)-(14)
        # 总DG出力
        P_gen_total = self.P_deg + self.P_mess + self.P_evs + self.P_pv
        Q_gen_total = self.Q_deg + self.Q_mess + self.Q_evs
        
        # 净注入功率
        P_net = P_gen_total - self.P_load + self.P_shed
        Q_net = Q_gen_total - self.Q_load + self.Q_shed
        
        # 功率平衡
        constraints.append(self.A @ self.P_branch == P_net)
        constraints.append(self.A @ self.Q_branch == Q_net)
        
        # 7. 电压约束 (17)
        constraints.append(self.V_sqr[0] == 1.0)  # 平衡节点
        constraints.append(self.V_sqr[1:] >= self.V_min_sqr)
        constraints.append(self.V_sqr[1:] <= self.V_max_sqr)
        
        # 8. 支路约束
        for k, (i, j) in enumerate(self.branches):
            i_idx, j_idx = i - 1, j - 1
            
            # 简化的电压降落约束
            constraints.append(
                self.V_sqr[i_idx] - self.V_sqr[j_idx] >= 
                2 * (self.R[k] * self.P_branch[k] + self.X[k] * self.Q_branch[k]) - 0.2
            )
            constraints.append(
                self.V_sqr[i_idx] - self.V_sqr[j_idx] <= 
                2 * (self.R[k] * self.P_branch[k] + self.X[k] * self.Q_branch[k]) + 0.2
            )
            
            # 电流限制
            if self.I_max[k] > 0:
                constraints.append(self.I_sqr[k] <= self.I_max[k]**2)
                
        # 9. 二阶锥约束 (20)
        for k, (i, j) in enumerate(self.branches):
            i_idx = i - 1
            constraints.append(
                cp.norm(cp.vstack([
                    2 * self.P_branch[k],
                    2 * self.Q_branch[k], 
                    self.I_sqr[k] - self.V_sqr[i_idx]
                ])) <= self.I_sqr[k] + self.V_sqr[i_idx]
            )
            
        # 目标函数 (21) - 最小化负荷削减成本 + DG调度成本
        obj_load_shed = cp.sum(cp.multiply(self.load_cost, self.P_shed)) * 1000  # MW转kW
        obj_deg = cp.sum(self.P_deg) * self.deg_cost * 1000
        obj_mess = cp.sum(cp.pos(self.P_mess)) * self.mess_cost * 1000  # 只计算放电成本
        
        objective = cp.Minimize(obj_load_shed + obj_deg + obj_mess)
        
        # 创建问题
        self.problem = cp.Problem(objective, constraints)
        
    def solve(self, solver=None, verbose: bool = True, **kwargs) -> Optional[Dict]:
        """
        求解模型
        
        Args:
            solver: CVXPY求解器
            verbose: 是否输出详细信息
            **kwargs: 传递给求解器的其他参数
            
        Returns:
            求解结果字典
        """
        if solver is None:
            try:
                import ecos
                solver = cp.ECOS
            except ImportError:
                solver = cp.SCS
                
        # 求解
        try:
            self.problem.solve(solver=solver, verbose=verbose, **kwargs)
        except Exception as e:
            logger.error(f"求解失败: {e}")
            return None
            
        if self.problem.status not in ["optimal", "optimal_inaccurate"]:
            logger.error(f"求解状态异常: {self.problem.status}")
            return None
            
        # 提取结果
        results = self._extract_results()
        
        if verbose:
            self._print_results(results)
            
        return results
        
    def _extract_results(self) -> Dict:
        """提取求解结果"""
        results = {
            'status': self.problem.status,
            'objective': self.problem.value,
            'solver': self.problem.solver_stats.solver_name,
            'solve_time': self.problem.solver_stats.solve_time,
        }
        
        # 提取变量值
        results['P_shed'] = {i+1: val * 1000 for i, val in enumerate(self.P_shed.value) 
                            if val > 1e-6}
        results['P_deg'] = {i+1: val * 1000 for i, val in enumerate(self.P_deg.value) 
                           if val > 1e-6}
        results['P_mess'] = {i+1: val * 1000 for i, val in enumerate(self.P_mess.value) 
                            if abs(val) > 1e-6}
        results['P_pv'] = {i+1: val * 1000 for i, val in enumerate(self.P_pv.value) 
                          if val > 1e-6}
        
        # 电压
        results['V_sqr'] = self.V_sqr.value
        results['V_kV'] = {i+1: np.sqrt(v) * self.V_base for i, v in enumerate(self.V_sqr.value)}
        
        # 统计信息
        results['total_P_shed'] = sum(results['P_shed'].values()) if results['P_shed'] else 0
        results['total_P_deg'] = sum(results['P_deg'].values()) if results['P_deg'] else 0
        results['total_P_mess'] = sum(v for v in results['P_mess'].values() if v > 0) if results['P_mess'] else 0
        results['total_P_pv'] = sum(results['P_pv'].values()) if results['P_pv'] else 0
        
        # 电压统计
        voltages = list(results['V_kV'].values())
        results['max_voltage'] = max(voltages)
        results['min_voltage'] = min(voltages[1:])  # 排除平衡节点
        
        # 成本分解
        results['cost_breakdown'] = {
            'load_shed': sum(self.load_cost[i] * self.P_shed.value[i] * 1000 
                           for i in range(self.n_buses)),
            'deg': sum(self.P_deg.value) * self.deg_cost * 1000,
            'mess': sum(max(0, self.P_mess.value[i]) * self.mess_cost * 1000 
                       for i in range(self.n_buses))
        }
        
        return results
        
    def _print_results(self, results: Dict):
        """打印求解结果"""
        print("\n" + "="*60)
        print("灾后静态调度优化结果 (Demo 2 - 含固定MESS和DEG)")
        print("="*60)
        
        print(f"\n求解状态: {results['status']}")
        print(f"求解器: {results['solver']}")
        print(f"求解时间: {results['solve_time']:.3f} 秒")
        
        print(f"\n目标函数值: {results['objective']:.2f} 元")
        print(f"  - 负荷削减成本: {results['cost_breakdown']['load_shed']:.2f} 元")
        print(f"  - DEG发电成本: {results['cost_breakdown']['deg']:.2f} 元")
        print(f"  - MESS放电成本: {results['cost_breakdown']['mess']:.2f} 元")
        
        print(f"\n功率汇总:")
        print(f"  负荷削减: {results['total_P_shed']:.2f} kW")
        print(f"  DEG发电: {results['total_P_deg']:.2f} kW")
        print(f"  MESS放电: {results['total_P_mess']:.2f} kW")
        print(f"  PV发电: {results['total_P_pv']:.2f} kW")
        
        print(f"\n电压范围:")
        print(f"  最大电压: {results['max_voltage']:.3f} kV")
        print(f"  最小电压: {results['min_voltage']:.3f} kV")
        
        if results['P_shed']:
            print(f"\n负荷削减详情:")
            for node, p_shed in sorted(results['P_shed'].items()):
                print(f"  节点 {node}: {p_shed:.2f} kW")
                
        if results['P_deg']:
            print(f"\nDEG出力:")
            for node, p_deg in sorted(results['P_deg'].items()):
                print(f"  节点 {node}: {p_deg:.2f} kW")
                
        if results['P_mess']:
            print(f"\nMESS充放电:")
            for node, p_mess in sorted(results['P_mess'].items()):
                status = "放电" if p_mess > 0 else "充电"
                print(f"  节点 {node}: {abs(p_mess):.2f} kW ({status})")
                
    def write_lp(self, filename: str):
        """输出问题描述"""
        with open(filename, 'w') as f:
            f.write(f"Problem: Post-Disaster Static Dispatch V2 (Demo 2)\n")
            f.write(f"Variables: {self.problem.size_metrics.num_scalar_variables}\n")
            f.write(f"Constraints: {self.problem.size_metrics.num_scalar_eq_constr + self.problem.size_metrics.num_scalar_leq_constr}\n")
            f.write(f"Status: {self.problem.status if hasattr(self.problem, 'status') else 'Not solved'}\n")
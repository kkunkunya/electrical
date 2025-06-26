"""
灾后静态调度模型 (t=0时刻) - 基于CVXPY实现
使用开源求解器的MISOCP最优负荷削减模型
"""

import cvxpy as cp
import numpy as np
from typing import Dict, List, Tuple, Optional
from src.datasets.loader import SystemData
import logging

logger = logging.getLogger(__name__)


class PostDisasterStaticModel:
    """灾后静态调度MISOCP模型 - CVXPY版本"""
    
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
        # branches是字典，需要获取值
        branch_list = list(self.data.branches.values()) if isinstance(self.data.branches, dict) else self.data.branches
        self.branches = [(b.from_bus, b.to_bus) for b in branch_list]
        self.n_branches = len(self.branches)
        
        # 构建节点-支路关联矩阵
        self.A = np.zeros((self.n_buses, self.n_branches))  # 节点-支路关联矩阵
        for k, (i, j) in enumerate(self.branches):
            self.A[i-1, k] = 1   # 流出节点i
            self.A[j-1, k] = -1  # 流入节点j
            
        # 负荷数据（MW）
        self.P_load = np.zeros(self.n_buses)
        self.Q_load = np.zeros(self.n_buses)
        self.load_cost = np.ones(self.n_buses)  # 默认成本为1
        
        # loads是字典，需要获取值
        load_list = list(self.data.loads.values()) if isinstance(self.data.loads, dict) else self.data.loads
        for i, load in enumerate(load_list):  # 从节点2开始有负荷
            bus_idx = load.bus_id - 1  # 使用实际的bus_id
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
            self.I_max[k] = branch.capacity / 1000  # A转kA（注意：capacity字段是电流容量）
            
        # 系统参数
        params = self.data.system_parameters
        self.V_min_sqr = (params.voltage_lower_limit / params.nominal_voltage) ** 2  # 标幺值平方
        self.V_max_sqr = (params.voltage_upper_limit / params.nominal_voltage) ** 2  # 标幺值平方
        self.V_base = params.nominal_voltage  # kV (已经是kV单位)
        
        # 发电机数据
        self.P_gen_max = np.zeros(self.n_buses)
        self.Q_gen_max = np.zeros(self.n_buses)
        
        # 分配发电机到节点
        generator_bus_map = {
            'DEG1': 5,    # 节点6（索引5）
            'MESS1': 14,  # 节点15（索引14）
            'EVS1': 21,   # 节点22（索引21）
            'PV1': 24     # 节点25（索引24）
        }
        
        # generators是字典
        gen_dict = self.data.generators if isinstance(self.data.generators, dict) else {f'gen_{i}': g for i, g in enumerate(self.data.generators)}
        
        for gen_name, gen in gen_dict.items():
            if gen_name in generator_bus_map:
                bus_idx = generator_bus_map[gen_name]
                if hasattr(gen, 'active_power_max') and gen.active_power_max:
                    self.P_gen_max[bus_idx] = gen.active_power_max / 1000  # MW
                if hasattr(gen, 'reactive_power_max') and gen.reactive_power_max:
                    self.Q_gen_max[bus_idx] = gen.reactive_power_max / 1000  # Mvar
                    
    def _build_model(self):
        """构建CVXPY优化模型"""
        # 决策变量
        # 负荷削减
        self.P_shed = cp.Variable(self.n_buses, nonneg=True, name="P_shed")
        self.Q_shed = cp.Variable(self.n_buses, nonneg=True, name="Q_shed")
        
        # 分布式电源输出
        self.P_gen = cp.Variable(self.n_buses, nonneg=True, name="P_gen")
        self.Q_gen = cp.Variable(self.n_buses, name="Q_gen")
        
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
        # P_shed <= P_load
        constraints.append(self.P_shed <= self.P_load)
        
        # Q_shed与P_shed成比例
        for i in range(self.n_buses):
            if self.P_load[i] > 1e-6:
                constraints.append(
                    self.Q_shed[i] == (self.Q_load[i] / self.P_load[i]) * self.P_shed[i]
                )
            else:
                constraints.append(self.Q_shed[i] == 0)
                
        # 2. 分布式电源约束 (10)-(12)
        # 有功功率限制
        constraints.append(self.P_gen <= self.P_gen_max)
        
        # 功率因数约束（假设最小功率因数0.95，tan(acos(0.95))≈0.33）
        constraints.append(self.Q_gen <= 0.33 * self.P_gen)
        constraints.append(self.Q_gen >= -0.33 * self.P_gen)
        
        # 只有特定节点有发电机
        for i in range(self.n_buses):
            if self.P_gen_max[i] == 0:
                constraints.append(self.P_gen[i] == 0)
                constraints.append(self.Q_gen[i] == 0)
                
        # 3. 功率平衡约束 (13)-(14)
        # 净注入功率
        P_net = self.P_gen - self.P_load + self.P_shed
        Q_net = self.Q_gen - self.Q_load + self.Q_shed
        
        # 考虑支路损耗的功率平衡
        # A @ P_branch = P_net + diag(R) @ I_sqr（损耗作为额外负荷）
        P_loss = cp.multiply(self.R, self.I_sqr)
        constraints.append(self.A @ self.P_branch == P_net)
        
        Q_loss = cp.multiply(self.X, self.I_sqr)
        constraints.append(self.A @ self.Q_branch == Q_net)
        
        # 4. 电压约束 (17)
        # 平衡节点（节点1）
        constraints.append(self.V_sqr[0] == 1.0)
        
        # 其他节点电压限制
        constraints.append(self.V_sqr[1:] >= self.V_min_sqr)
        constraints.append(self.V_sqr[1:] <= self.V_max_sqr)
        
        # 5. 支路约束
        for k, (i, j) in enumerate(self.branches):
            i_idx, j_idx = i - 1, j - 1
            
            # 简化的电压降落约束（线性化近似）
            # V_i - V_j ≈ 2(R*P + X*Q)/V_nom
            constraints.append(
                self.V_sqr[i_idx] - self.V_sqr[j_idx] >= 
                2 * (self.R[k] * self.P_branch[k] + self.X[k] * self.Q_branch[k]) - 0.2
            )
            constraints.append(
                self.V_sqr[i_idx] - self.V_sqr[j_idx] <= 
                2 * (self.R[k] * self.P_branch[k] + self.X[k] * self.Q_branch[k]) + 0.2
            )
            
            # 电流限制 (18)
            if self.I_max[k] > 0:
                constraints.append(self.I_sqr[k] <= self.I_max[k]**2)
                
        # 6. 二阶锥约束 (20) - 功率与电流关系
        for k, (i, j) in enumerate(self.branches):
            i_idx = i - 1
            # ||[2*P; 2*Q; I^2 - V^2]||_2 <= I^2 + V^2
            constraints.append(
                cp.norm(cp.vstack([
                    2 * self.P_branch[k],
                    2 * self.Q_branch[k], 
                    self.I_sqr[k] - self.V_sqr[i_idx]
                ])) <= self.I_sqr[k] + self.V_sqr[i_idx]
            )
            
        # 目标函数 (21) - 最小化负荷削减成本
        objective = cp.Minimize(
            cp.sum(cp.multiply(self.load_cost, self.P_shed)) * 1000  # MW转kW
        )
        
        # 创建问题
        self.problem = cp.Problem(objective, constraints)
        
    def solve(self, solver=None, verbose: bool = True, **kwargs) -> Optional[Dict]:
        """
        求解模型
        
        Args:
            solver: CVXPY求解器（默认自动选择）
            verbose: 是否输出详细信息
            **kwargs: 传递给求解器的其他参数
            
        Returns:
            求解结果字典
        """
        # 选择求解器
        if solver is None:
            # 优先使用ECOS（对SOCP效果好）
            try:
                import ecos
                solver = cp.ECOS
            except ImportError:
                # 退而使用SCS
                solver = cp.SCS
                
        # 求解
        try:
            self.problem.solve(solver=solver, verbose=verbose, **kwargs)
        except Exception as e:
            logger.error(f"求解失败: {e}")
            return None
            
        # 检查求解状态
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
        results['Q_shed'] = {i+1: val * 1000 for i, val in enumerate(self.Q_shed.value) 
                            if val > 1e-6}
        results['P_gen'] = {i+1: val * 1000 for i, val in enumerate(self.P_gen.value) 
                           if val > 1e-6}
        results['Q_gen'] = {i+1: val * 1000 for i, val in enumerate(self.Q_gen.value) 
                           if abs(val) > 1e-6}
        
        # 电压
        results['V_sqr'] = self.V_sqr.value
        results['V_kV'] = {i+1: np.sqrt(v) * self.V_base for i, v in enumerate(self.V_sqr.value)}
        
        # 支路功率
        results['P_branch'] = {self.branches[k]: val * 1000 
                              for k, val in enumerate(self.P_branch.value)}
        results['Q_branch'] = {self.branches[k]: val * 1000 
                              for k, val in enumerate(self.Q_branch.value)}
        results['I_sqr'] = self.I_sqr.value
        
        # 统计信息
        results['total_P_shed'] = sum(results['P_shed'].values())
        results['total_Q_shed'] = sum(results['Q_shed'].values())
        results['total_P_gen'] = sum(results['P_gen'].values())
        
        # 电压统计
        voltages = list(results['V_kV'].values())
        results['max_voltage'] = max(voltages)
        results['min_voltage'] = min(voltages[1:])  # 排除平衡节点
        
        # 损耗计算
        results['losses'] = sum(self.R[k] * self.I_sqr.value[k] * 1000 
                               for k in range(self.n_branches))
        
        return results
        
    def _print_results(self, results: Dict):
        """打印求解结果"""
        print("\n" + "="*60)
        print("灾后静态调度优化结果 (CVXPY)")
        print("="*60)
        
        print(f"\n求解状态: {results['status']}")
        print(f"求解器: {results['solver']}")
        print(f"求解时间: {results['solve_time']:.3f} 秒")
        
        print(f"\n目标函数值 (负荷削减成本): {results['objective']:.2f} 元")
        print(f"总有功负荷削减: {results['total_P_shed']:.2f} kW")
        print(f"总无功负荷削减: {results['total_Q_shed']:.2f} kvar")
        print(f"总发电功率: {results['total_P_gen']:.2f} kW")
        print(f"总损耗: {results['losses']:.2f} kW")
        
        print(f"\n电压范围:")
        print(f"  最大电压: {results['max_voltage']:.3f} kV")
        print(f"  最小电压: {results['min_voltage']:.3f} kV")
        
        if results['P_shed']:
            print(f"\n负荷削减详情:")
            for node, p_shed in sorted(results['P_shed'].items()):
                if p_shed > 0.1:
                    print(f"  节点 {node}: {p_shed:.2f} kW")
                    
        if results['P_gen']:
            print(f"\n分布式电源出力:")
            for node, p_gen in sorted(results['P_gen'].items()):
                print(f"  节点 {node}: {p_gen:.2f} kW")
                
    def write_lp(self, filename: str):
        """
        输出问题的LP格式（用于调试）
        
        Args:
            filename: 输出文件名
        """
        # CVXPY不直接支持LP输出，但可以输出问题描述
        with open(filename, 'w') as f:
            f.write(f"Problem name: Post-Disaster Static Dispatch\n")
            f.write(f"Variables: {self.problem.size_metrics.num_scalar_variables}\n")
            f.write(f"Constraints: {self.problem.size_metrics.num_scalar_eq_constr + self.problem.size_metrics.num_scalar_leq_constr}\n")
            f.write(f"Status: {self.problem.status if hasattr(self.problem, 'status') else 'Not solved'}\n")
            
            if hasattr(self, 'P_shed') and self.P_shed.value is not None:
                f.write(f"\nObjective value: {self.problem.value:.2f}\n")
                f.write("\nLoad shedding:\n")
                for i, val in enumerate(self.P_shed.value):
                    if val > 1e-6:
                        f.write(f"  P_shed[{i+1}] = {val*1000:.2f} kW\n")
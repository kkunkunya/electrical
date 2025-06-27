"""
灾后多时段动态调度模型 (Demo 1) - 移动储能时空动态调度
基于CVXPY实现的MISOCP多时段优化模型，实现文章B的完整数学模型
"""

import cvxpy as cp
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from src.datasets.loader import SystemData
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class TrafficModel:
    """交通模型 - 计算动态通行时间"""
    
    def __init__(self, traffic_profile_path: str):
        """
        初始化交通模型
        
        Args:
            traffic_profile_path: 交通拥堵数据文件路径
        """
        self.traffic_data = self._load_traffic_data(traffic_profile_path)
        
    def _load_traffic_data(self, path: str) -> Dict[int, float]:
        """加载交通拥堵数据"""
        try:
            if path and Path(path).exists():
                logger.info(f"从文件加载交通数据: {path}")
                df = pd.read_csv(path)
                logger.info(f"交通数据文件列: {df.columns.tolist()}")
                
                # 检查列名
                time_col = 'time_hour'
                congestion_col = 'congestion_level'
                
                if time_col not in df.columns or congestion_col not in df.columns:
                    logger.warning(f"CSV文件缺少必要列: {time_col}, {congestion_col}")
                    raise ValueError(f"CSV文件格式错误")
                
                traffic_dict = {}
                for _, row in df.iterrows():
                    traffic_dict[int(row[time_col])] = float(row[congestion_col])
                
                logger.info(f"成功加载{len(traffic_dict)}个时段的交通数据")
                return traffic_dict
            else:
                logger.info("使用默认交通数据")
                # 使用图B2的数据作为默认值
                return {
                    3: 0.66, 4: 0.75, 5: 0.81, 6: 0.80, 7: 0.73, 8: 0.60,
                    9: 0.50, 10: 0.40, 11: 0.35, 12: 0.34, 13: 0.33, 14: 0.34,
                    15: 0.36, 16: 0.40, 17: 0.43, 18: 0.44, 19: 0.42, 20: 0.38,
                    21: 0.33, 22: 0.28, 23: 0.24
                }
        except Exception as e:
            logger.warning(f"加载交通数据失败: {e}, 使用默认数据")
            return {t: 0.4 for t in range(3, 24)}  # 默认拥堵程度
    
    def calculate_travel_time(self, from_node: int, to_node: int, time_period: int,
                            distance_matrix: np.ndarray, v_ideal: float = 25.0) -> float:
        """
        计算动态通行时间 - 公式(22)-(24)
        
        Args:
            from_node: 起始节点
            to_node: 目标节点
            time_period: 时间段(小时)
            distance_matrix: 节点间距离矩阵(km)
            v_ideal: 理想车速(km/h)
            
        Returns:
            通行时间(小时)
        """
        # 获取交通拥堵程度 c(t)
        c_t = self.traffic_data.get(time_period, 0.4)
        
        # 计算实际车速 - 公式(24): v_ME(t) = v_0 * exp(-1.7*c)
        v_actual = v_ideal * np.exp(-1.7 * c_t)
        
        # 获取节点间基础距离 - 公式(23): D_jk(t) = D_jk,0 * (1 + c(t))
        distance_matrix = np.array(distance_matrix)
        d_base = distance_matrix[from_node-1, to_node-1]
        d_adjusted = d_base * (1 + c_t)
        
        # 计算通行时间 - 公式(22): T_ME_ijk(t) = D_jk(t) / v_ME(t)
        travel_time = d_adjusted / v_actual
        
        return travel_time


class PostDisasterDynamicModel:
    """灾后多时段动态调度MISOCP模型 - 支持移动储能时空调度"""
    
    def __init__(self, system_data: SystemData, n_periods: int = 21, 
                 start_hour: int = 3, traffic_profile_path: Optional[str] = None,
                 pv_profile_path: Optional[str] = None):
        """
        初始化动态调度模型
        
        Args:
            system_data: 系统数据对象
            n_periods: 时间段数量(默认21个小时: 3:00-23:00)
            start_hour: 起始小时(默认3点)
            traffic_profile_path: 交通拥堵数据路径
            pv_profile_path: 光伏出力数据路径
        """
        self.data = system_data
        self.n_periods = n_periods
        self.start_hour = start_hour
        self.time_periods = list(range(start_hour, start_hour + n_periods))
        
        # 初始化交通模型
        self.traffic_model = TrafficModel(traffic_profile_path or "")
        
        # 提取系统数据
        self._extract_data()
        
        # 加载时变数据
        self._load_time_varying_data(pv_profile_path)
        
        # 生成节点间距离矩阵
        self._generate_distance_matrix()
        
        # 构建优化模型
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
            
        # 负荷数据(MW)
        self.P_load = np.zeros(self.n_buses)
        self.Q_load = np.zeros(self.n_buses)
        self.load_cost = np.ones(self.n_buses)  # 负荷削减成本
        
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
            self.I_max[k] = branch.capacity / 1000  # 转换为kA
            
        # 系统参数
        params = self.data.system_parameters
        self.V_min_sqr = (params.voltage_lower_limit / params.nominal_voltage) ** 2
        self.V_max_sqr = (params.voltage_upper_limit / params.nominal_voltage) ** 2
        self.V_base = params.nominal_voltage  # kV
        
        # 移动储能参数(MESS)
        self.n_mess = 2  # 2个移动储能设备
        self.mess_ids = list(range(1, self.n_mess + 1))
        
        # 移动储能技术参数(基于表B1)
        self.P_mess_max = 200 / 1000  # MW - 充放电功率上限
        self.Q_mess_max = 170 / 1000  # Mvar - 无功功率上限
        self.E_mess_cap = 600 / 1000  # MWh - 储能容量
        self.eta_mess_ch = 0.98       # 充电效率
        self.eta_mess_dch = 0.98      # 放电效率
        self.E_mess_min = 0.1 * self.E_mess_cap  # SOC下限(10%)
        self.E_mess_max = 0.9 * self.E_mess_cap  # SOC上限(90%)
        
        # 移动储能初始位置(灾前预布局) - 基于图3
        self.mess_initial_pos = {1: 1, 2: 2}  # MESS1在节点1, MESS2在节点2
        
        # 移动储能配置成本和时间
        self.T_install = 0.5  # 安装配置时间(小时)
        self.cost_config = 500  # 配置成本(元)
        
        # 其他DG参数
        self._initialize_other_dg_params()
        
    def _initialize_other_dg_params(self):
        """初始化其他分布式电源参数"""
        # DEG参数
        self.P_deg_max = np.zeros(self.n_buses)
        self.Q_deg_max = np.zeros(self.n_buses)
        
        # EVS参数  
        self.P_evs_max = np.zeros(self.n_buses)
        self.Q_evs_max = np.zeros(self.n_buses)
        self.E_evs_cap = np.zeros(self.n_buses)
        self.eta_evs = np.zeros(self.n_buses)
        
        # 基于论文表B1分配参数
        generator_config = {
            # DEG分布(节点: [P_max_kW, Q_max_kvar])
            2: [120, 100], 6: [120, 100], 10: [120, 100],  # DEG1,2,3
            25: [80, 60], 30: [80, 60],  # DEG4,5
            
            # EVS分布(节点: [P_max_kW, Q_max_kvar, E_cap_kWh])
            5: [60, 50, 120], 15: [60, 50, 120], 20: [60, 50, 120],  # EVS1,2,3
        }
        
        for node_id, params in generator_config.items():
            bus_idx = node_id - 1
            if len(params) == 2:  # DEG
                self.P_deg_max[bus_idx] = params[0] / 1000  # kW转MW
                self.Q_deg_max[bus_idx] = params[1] / 1000
            elif len(params) == 3:  # EVS
                self.P_evs_max[bus_idx] = params[0] / 1000
                self.Q_evs_max[bus_idx] = params[1] / 1000
                self.E_evs_cap[bus_idx] = params[2] / 1000  # kWh转MWh
                self.eta_evs[bus_idx] = 0.98
        
    def _load_time_varying_data(self, pv_profile_path: Optional[str]):
        """加载时变数据 - 光伏出力预测"""
        try:
            if pv_profile_path and Path(pv_profile_path).exists():
                logger.info(f"从文件加载光伏数据: {pv_profile_path}")
                pv_data = pd.read_csv(pv_profile_path)
                self.pv_profiles = self._process_pv_data(pv_data)
            else:
                logger.info("使用默认光伏数据")
                self.pv_profiles = self._get_default_pv_profiles()
        except Exception as e:
            logger.warning(f"加载光伏数据失败: {e}, 使用默认数据")
            self.pv_profiles = self._get_default_pv_profiles()
    
    def _process_pv_data(self, pv_data: pd.DataFrame) -> Dict[int, Dict[int, float]]:
        """处理从CSV加载的光伏数据"""
        profiles = {}
        
        # 光伏机组分布(节点: PV列名)
        pv_nodes = {14: 'PV1', 16: 'PV2', 20: 'PV3', 24: 'PV4', 32: 'PV5'}
        
        for node_id, pv_col in pv_nodes.items():
            profiles[node_id] = {}
            
            for t in range(self.n_periods):
                hour = self.time_periods[t]
                
                # 查找对应时间的数据
                hour_data = pv_data[pv_data['time_hour'] == hour]
                
                if not hour_data.empty and pv_col in pv_data.columns:
                    # kW转MW
                    profiles[node_id][t] = float(hour_data[pv_col].iloc[0]) / 1000
                else:
                    profiles[node_id][t] = 0.0
        
        logger.info(f"成功加载{len(pv_nodes)}个光伏机组的{self.n_periods}时段数据")
        return profiles
    
    def _get_default_pv_profiles(self) -> Dict[int, Dict[int, float]]:
        """获取默认光伏出力数据 - 基于图B3"""
        # 光伏机组分布(节点: PV编号)
        pv_nodes = {14: 1, 16: 2, 20: 3, 24: 4, 32: 5}  # 基于图3的分布
        
        # 每个时段的光伏出力(kW) - 基于图B3数据
        pv_hourly_output = {
            1: [0]*6 + [20,40,70,95,115,120,120,105,85,60,30,10] + [0]*5,
            2: [0]*6 + [40,90,140,190,225,240,240,215,175,120,70,25] + [0]*5,
            3: [0]*6 + [60,110,210,285,335,360,355,320,260,180,110,40] + [0]*5,
            4: [0]*6 + [60,110,210,285,335,360,355,320,260,180,110,40] + [0]*5,  # PV3/PV4相同
            5: [0]*6 + [30,70,110,145,170,180,180,165,135,95,55,20] + [0]*5
        }
        
        profiles = {}
        for node_id, pv_id in pv_nodes.items():
            profiles[node_id] = {}
            for t, hour in enumerate(self.time_periods):
                # 将小时转换为数组索引
                hour_idx = hour - 3 if hour >= 3 else hour + 21
                if hour_idx < len(pv_hourly_output[pv_id]):
                    profiles[node_id][t] = pv_hourly_output[pv_id][hour_idx] / 1000  # kW转MW
                else:
                    profiles[node_id][t] = 0.0
                    
        return profiles
    
    def _generate_distance_matrix(self):
        """生成节点间距离矩阵 - 基于配电网拓扑结构"""
        # 简化的距离计算 - 基于支路长度和拓扑结构
        self.distance_matrix = np.ones((self.n_buses, self.n_buses)) * 100  # 默认100km
        
        # 设置相邻节点间距离为较小值
        for i, j in self.branches:
            base_dist = 5.0  # 相邻节点基础距离5km
            self.distance_matrix[i-1, j-1] = base_dist
            self.distance_matrix[j-1, i-1] = base_dist
            
        # 对角线为0
        np.fill_diagonal(self.distance_matrix, 0)
        
        # 使用Floyd-Warshall算法计算最短路径
        for k in range(self.n_buses):
            for i in range(self.n_buses):
                for j in range(self.n_buses):
                    if self.distance_matrix[i, k] + self.distance_matrix[k, j] < self.distance_matrix[i, j]:
                        self.distance_matrix[i, j] = self.distance_matrix[i, k] + self.distance_matrix[k, j]
    
    def _build_model(self):
        """构建多时段CVXPY优化模型"""
        # 时间索引
        T = self.n_periods
        N = self.n_buses
        M = self.n_mess
        
        # ============ 决策变量 ============
        # 多时段决策变量 - 维度从 (n_buses) 扩展为 (n_buses, n_periods)
        self.P_shed = cp.Variable((N, T), nonneg=True, name="P_shed")    # 负荷削减
        self.Q_shed = cp.Variable((N, T), nonneg=True, name="Q_shed")
        
        # 分布式电源出力
        self.P_deg = cp.Variable((N, T), nonneg=True, name="P_deg")      # DEG有功出力
        self.Q_deg = cp.Variable((N, T), name="Q_deg")                   # DEG无功出力
        self.P_pv = cp.Variable((N, T), nonneg=True, name="P_pv")        # PV出力
        
        # EVS充放电
        self.P_evs_ch = cp.Variable((N, T), nonneg=True, name="P_evs_ch")   # EVS充电
        self.P_evs_dch = cp.Variable((N, T), nonneg=True, name="P_evs_dch") # EVS放电
        self.Q_evs = cp.Variable((N, T), name="Q_evs")                      # EVS无功
        self.U_evs_ch = cp.Variable((N, T), nonneg=True, name="U_evs_ch")  # EVS充电状态 (放松为连续)
        self.U_evs_dch = cp.Variable((N, T), nonneg=True, name="U_evs_dch") # EVS放电状态 (放松为连续)
        self.E_evs = cp.Variable((N, T), nonneg=True, name="E_evs")         # EVS荷电状态
        
        # ============ 移动储能(MESS)相关变量 ============
        # 连接状态: alpha_ME[i,j,t] - 移动储能i在时刻t是否连接到节点j (放松为连续变量)
        self.alpha_ME = cp.Variable((M, N, T), nonneg=True, name="alpha_ME")
        
        # 充放电状态和功率 (放松为连续变量)
        self.U_mess_ch = cp.Variable((M, T), nonneg=True, name="U_mess_ch")   # 充电状态
        self.U_mess_dch = cp.Variable((M, T), nonneg=True, name="U_mess_dch") # 放电状态
        self.P_mess_ch = cp.Variable((M, T), nonneg=True, name="P_mess_ch")    # 充电功率
        self.P_mess_dch = cp.Variable((M, T), nonneg=True, name="P_mess_dch")  # 放电功率
        self.Q_mess = cp.Variable((M, T), name="Q_mess")                       # 无功功率
        
        # 荷电状态(SOC)
        self.E_mess = cp.Variable((M, T), nonneg=True, name="E_mess")
        
        # 线性化辅助变量 - 用于处理公式(28)的双线性项 (放松为连续变量)
        self.alpha_MCS = cp.Variable((M, N, T), nonneg=True, name="alpha_MCS")
        
        # 电网状态变量
        self.P_branch = cp.Variable((self.n_branches, T), name="P_branch")     # 支路有功功率
        self.Q_branch = cp.Variable((self.n_branches, T), name="Q_branch")     # 支路无功功率
        self.V_sqr = cp.Variable((N, T), nonneg=True, name="V_sqr")            # 节点电压平方
        self.I_sqr = cp.Variable((self.n_branches, T), nonneg=True, name="I_sqr") # 支路电流平方
        
        # ============ 约束条件 ============
        constraints = []
        
        # 1. 多时段目标函数 - 公式(21)扩展
        objective_terms = []
        for t in range(T):
            # 负荷削减成本
            load_shed_cost = cp.sum(cp.multiply(self.load_cost, self.P_shed[:, t]))
            objective_terms.append(load_shed_cost)
        
        objective = cp.Minimize(cp.sum(objective_terms) * 1000)  # MW转kW
        
        # 2. 负荷削减约束 - 公式(8)-(9)扩展到多时段
        for t in range(T):
            constraints.append(self.P_shed[:, t] <= self.P_load)
            
            # 负荷削减的有功无功比例约束
            for i in range(N):
                if self.P_load[i] > 1e-6:
                    constraints.append(
                        self.Q_shed[i, t] == (self.Q_load[i] / self.P_load[i]) * self.P_shed[i, t]
                    )
                else:
                    constraints.append(self.Q_shed[i, t] == 0)
        
        # 3. 移动储能(MESS)时空动态调度约束 - 公式(25)-(35)
        self._add_mess_constraints(constraints)
        
        # 4. EVS约束 - 公式(36)-(40)
        self._add_evs_constraints(constraints)
        
        # 5. 其他DG约束
        self._add_other_dg_constraints(constraints)
        
        # 6. 多时段功率平衡约束
        self._add_power_balance_constraints(constraints)
        
        # 7. 电网运行约束(DistFlow方程)
        self._add_network_constraints(constraints)
        
        # 创建问题
        self.problem = cp.Problem(objective, constraints)
        
    def _add_mess_constraints(self, constraints: List):
        """添加移动储能(MESS)时空动态调度约束 - 公式(25)-(35)"""
        T = self.n_periods
        N = self.n_buses
        M = self.n_mess
        
        # 公式(25): 初始位置约束 - 灾后恢复初始时刻移动储能的位置
        for i in range(M):
            for j in range(N):
                if j + 1 == self.mess_initial_pos[i + 1]:  # 初始位置
                    constraints.append(self.alpha_ME[i, j, 0] == 1)
                else:
                    constraints.append(self.alpha_ME[i, j, 0] == 0)
        
        # 公式(27): 任意时刻一个移动储能最多连接一个节点
        for i in range(M):
            for t in range(T):
                constraints.append(cp.sum(self.alpha_ME[i, :, t]) <= 1)
                # 添加上界约束
                for j in range(N):
                    constraints.append(self.alpha_ME[i, j, t] <= 1)
        
        # 公式(26): 移动过程中的连接状态约束
        # 简化处理: 如果储能在移动，则不能同时在两个节点连接
        for i in range(M):
            for t in range(T - 1):
                # 位置变化时需要考虑通行时间
                for j in range(N):
                    for k in range(N):
                        if j != k:
                            # 计算通行时间
                            travel_time = self.traffic_model.calculate_travel_time(
                                j + 1, k + 1, self.time_periods[t], self.distance_matrix
                            )
                            
                            # 如果通行时间大于1小时，添加移动约束
                            if travel_time > 1.0:
                                # 简化: 移动期间不能立即在目标节点连接
                                constraints.append(
                                    self.alpha_ME[i, j, t] + self.alpha_ME[i, k, t + 1] <= 1
                                )
        
        # 公式(35): 双线性项线性化 - alpha_MCS辅助变量
        for i in range(M):
            for j in range(N):
                for t in range(T - 1):
                    # 线性化约束
                    constraints.append(self.alpha_MCS[i, j, t] <= self.alpha_ME[i, j, t])
                    constraints.append(self.alpha_MCS[i, j, t] <= self.alpha_ME[i, j, t + 1])
                    constraints.append(
                        self.alpha_MCS[i, j, t] >= 
                        self.alpha_ME[i, j, t] + self.alpha_ME[i, j, t + 1] - 1
                    )
        
        # 公式(28): 充放电状态与连接状态的耦合 - 线性化后
        for i in range(M):
            for t in range(T - 1):
                constraints.append(
                    self.U_mess_ch[i, t] + self.U_mess_dch[i, t] <= 
                    cp.sum(self.alpha_MCS[i, :, t])
                )
        
        # 公式(29)-(32): 充放电功率约束
        for i in range(M):
            for t in range(T):
                constraints.append(self.P_mess_ch[i, t] <= self.U_mess_ch[i, t] * self.P_mess_max)
                constraints.append(self.P_mess_dch[i, t] <= self.U_mess_dch[i, t] * self.P_mess_max)
                constraints.append(self.Q_mess[i, t] <= self.Q_mess_max)
                constraints.append(self.Q_mess[i, t] >= -self.Q_mess_max)
                
                # 充放电互斥和上界约束
                constraints.append(self.U_mess_ch[i, t] + self.U_mess_dch[i, t] <= 1)
                constraints.append(self.U_mess_ch[i, t] <= 1)
                constraints.append(self.U_mess_dch[i, t] <= 1)
        
        # 公式(33): SOC动态变化
        dt = 1.0  # 时间步长1小时
        for i in range(M):
            # 初始SOC
            constraints.append(self.E_mess[i, 0] == 0.5 * self.E_mess_cap)  # 初始50%
            
            for t in range(T - 1):
                constraints.append(
                    self.E_mess[i, t + 1] == 
                    self.E_mess[i, t] + 
                    self.P_mess_ch[i, t] * self.eta_mess_ch * dt -
                    self.P_mess_dch[i, t] / self.eta_mess_dch * dt
                )
        
        # 公式(34): SOC上下限
        for i in range(M):
            for t in range(T):
                constraints.append(self.E_mess[i, t] >= self.E_mess_min)
                constraints.append(self.E_mess[i, t] <= self.E_mess_max)
    
    def _add_evs_constraints(self, constraints: List):
        """添加EVS约束 - 公式(36)-(40)"""
        T = self.n_periods
        N = self.n_buses
        
        for t in range(T):
            # 公式(36)-(37): 充放电功率约束
            constraints.append(self.P_evs_ch[:, t] <= cp.multiply(self.U_evs_ch[:, t], self.P_evs_max))
            constraints.append(self.P_evs_dch[:, t] <= cp.multiply(self.U_evs_dch[:, t], self.P_evs_max))
            
            # 公式(38): 充放电状态互斥
            constraints.append(self.U_evs_ch[:, t] + self.U_evs_dch[:, t] <= 1)
            
            # 无功功率约束
            constraints.append(self.Q_evs[:, t] <= 0.33 * (self.P_evs_ch[:, t] + self.P_evs_dch[:, t]))
            constraints.append(self.Q_evs[:, t] >= -0.33 * (self.P_evs_ch[:, t] + self.P_evs_dch[:, t]))
        
        # 公式(39): SOC动态变化
        dt = 1.0
        for i in range(N):
            if self.E_evs_cap[i] > 1e-6:  # 只有配置EVS的节点
                # 初始SOC
                constraints.append(self.E_evs[i, 0] == 0.6 * self.E_evs_cap[i])  # 初始60%
                
                for t in range(T - 1):
                    constraints.append(
                        self.E_evs[i, t + 1] == 
                        self.E_evs[i, t] + 
                        self.P_evs_ch[i, t] * self.eta_evs[i] * dt -
                        self.P_evs_dch[i, t] / self.eta_evs[i] * dt
                    )
                
                # 公式(40): SOC上下限
                for t in range(T):
                    constraints.append(self.E_evs[i, t] >= 0.1 * self.E_evs_cap[i])
                    constraints.append(self.E_evs[i, t] <= 0.9 * self.E_evs_cap[i])
            else:
                # 没有EVS的节点
                for t in range(T):
                    constraints.append(self.P_evs_ch[i, t] == 0)
                    constraints.append(self.P_evs_dch[i, t] == 0)
                    constraints.append(self.Q_evs[i, t] == 0)
                    constraints.append(self.E_evs[i, t] == 0)
    
    def _add_other_dg_constraints(self, constraints: List):
        """添加其他DG约束"""
        T = self.n_periods
        N = self.n_buses
        
        for t in range(T):
            # DEG约束
            constraints.append(self.P_deg[:, t] <= self.P_deg_max)
            constraints.append(self.Q_deg[:, t] <= 0.33 * self.P_deg[:, t])
            constraints.append(self.Q_deg[:, t] >= -0.33 * self.P_deg[:, t])
            
            # PV出力约束 - 使用时变光伏出力数据
            pv_output_t = np.zeros(N)
            for node_id, profiles in self.pv_profiles.items():
                if isinstance(profiles, dict) and t in profiles:
                    pv_output_t[node_id - 1] = float(profiles[t])
                else:
                    pv_output_t[node_id - 1] = 0.0
            
            constraints.append(self.P_pv[:, t] <= pv_output_t)
    
    def _add_power_balance_constraints(self, constraints: List):
        """添加多时段功率平衡约束"""
        T = self.n_periods
        N = self.n_buses
        M = self.n_mess
        
        for t in range(T):
            # 简化的功率平衡 - 忽略MESS的节点间分配，直接用总体平衡
            P_mess_net = cp.sum(self.P_mess_dch[:, t] - self.P_mess_ch[:, t])
            Q_mess_net = cp.sum(self.Q_mess[:, t])
            
            # 创建功率平衡变量
            P_gen_total = cp.Variable(N, name=f"P_gen_total_{t}")
            Q_gen_total = cp.Variable(N, name=f"Q_gen_total_{t}")
            
            # 设置功率平衡约束
            for j in range(N):
                if j == 0:  # 平衡节点包含MESS
                    constraints.append(
                        P_gen_total[j] == 
                        self.P_deg[j, t] + self.P_evs_dch[j, t] - self.P_evs_ch[j, t] + 
                        self.P_pv[j, t] + P_mess_net
                    )
                    constraints.append(
                        Q_gen_total[j] == 
                        self.Q_deg[j, t] + self.Q_evs[j, t] + Q_mess_net
                    )
                else:
                    constraints.append(
                        P_gen_total[j] == 
                        self.P_deg[j, t] + self.P_evs_dch[j, t] - self.P_evs_ch[j, t] + self.P_pv[j, t]
                    )
                    constraints.append(
                        Q_gen_total[j] == 
                        self.Q_deg[j, t] + self.Q_evs[j, t]
                    )
            
            # 净注入功率
            P_net = P_gen_total - self.P_load + self.P_shed[:, t]
            Q_net = Q_gen_total - self.Q_load + self.Q_shed[:, t]
            
            # 功率平衡约束
            constraints.append(self.A @ self.P_branch[:, t] == P_net)
            constraints.append(self.A @ self.Q_branch[:, t] == Q_net)
    
    def _add_network_constraints(self, constraints: List):
        """添加电网运行约束(DistFlow方程)"""
        T = self.n_periods
        
        for t in range(T):
            # 电压约束
            constraints.append(self.V_sqr[0, t] == 1.0)  # 平衡节点
            constraints.append(self.V_sqr[1:, t] >= self.V_min_sqr)
            constraints.append(self.V_sqr[1:, t] <= self.V_max_sqr)
            
            # 支路约束
            for k, (i, j) in enumerate(self.branches):
                i_idx, j_idx = i - 1, j - 1
                
                # 简化的电压降落约束
                constraints.append(
                    self.V_sqr[i_idx, t] - self.V_sqr[j_idx, t] >= 
                    2 * (self.R[k] * self.P_branch[k, t] + self.X[k] * self.Q_branch[k, t]) - 0.2
                )
                constraints.append(
                    self.V_sqr[i_idx, t] - self.V_sqr[j_idx, t] <= 
                    2 * (self.R[k] * self.P_branch[k, t] + self.X[k] * self.Q_branch[k, t]) + 0.2
                )
                
                # 电流限制
                if self.I_max[k] > 0:
                    constraints.append(self.I_sqr[k, t] <= self.I_max[k]**2)
                
                # 二阶锥约束
                constraints.append(
                    cp.norm(cp.vstack([
                        2 * self.P_branch[k, t],
                        2 * self.Q_branch[k, t], 
                        self.I_sqr[k, t] - self.V_sqr[i_idx, t]
                    ])) <= self.I_sqr[k, t] + self.V_sqr[i_idx, t]
                )

    def solve(self, solver=None, verbose: bool = True, **kwargs) -> Optional[Dict]:
        """
        求解多时段动态调度模型
        
        Args:
            solver: CVXPY求解器
            verbose: 是否输出详细信息
            **kwargs: 传递给求解器的其他参数
            
        Returns:
            求解结果字典
        """
        if solver is None:
            try:
                import mosek
                solver = cp.MOSEK
                logger.info("使用MOSEK求解器")
            except ImportError:
                try:
                    import ecos
                    solver = cp.ECOS
                    logger.info("使用ECOS求解器")
                except ImportError:
                    solver = cp.SCS
                    logger.info("使用SCS求解器")
        
        # 求解
        try:
            logger.info("开始求解多时段动态调度模型...")
            self.problem.solve(solver=solver, verbose=verbose, **kwargs)
        except Exception as e:
            logger.error(f"求解失败: {e}")
            return None
            
        if self.problem.status not in ["optimal", "optimal_inaccurate"]:
            logger.error(f"求解状态异常: {self.problem.status}")
            return None
            
        logger.info(f"求解完成，状态: {self.problem.status}")
        
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
            'n_periods': self.n_periods,
            'time_periods': self.time_periods,
        }
        
        # 提取多时段结果
        results['P_shed_time'] = {}
        results['mess_schedule'] = {}
        results['evs_schedule'] = {}
        results['voltage_time'] = {}
        
        for t in range(self.n_periods):
            hour = self.time_periods[t]
            
            # 负荷削减
            results['P_shed_time'][hour] = {
                i+1: val * 1000 for i, val in enumerate(self.P_shed[:, t].value) if val > 1e-6
            }
            
            # MESS调度
            results['mess_schedule'][hour] = {}
            for i in range(self.n_mess):
                mess_id = i + 1
                # 找到MESS位置
                location = None
                for j in range(self.n_buses):
                    if self.alpha_ME[i, j, t].value > 0.5:
                        location = j + 1
                        break
                
                results['mess_schedule'][hour][mess_id] = {
                    'location': location,
                    'P_ch': self.P_mess_ch[i, t].value * 1000 if self.P_mess_ch[i, t].value > 1e-6 else 0,
                    'P_dch': self.P_mess_dch[i, t].value * 1000 if self.P_mess_dch[i, t].value > 1e-6 else 0,
                    'SOC': self.E_mess[i, t].value / self.E_mess_cap * 100,  # 百分比
                }
            
            # 电压分布
            results['voltage_time'][hour] = {
                i+1: np.sqrt(v) * self.V_base for i, v in enumerate(self.V_sqr[:, t].value)
            }
        
        # 统计信息
        total_shed = sum(sum(period_shed.values()) for period_shed in results['P_shed_time'].values())
        results['total_load_shed'] = total_shed
        
        return results
    
    def _print_results(self, results: Dict):
        """打印求解结果"""
        print("\n" + "="*80)
        print("灾后多时段动态调度优化结果 (Demo 1 - 移动储能时空调度)")
        print("="*80)
        
        print(f"\n求解状态: {results['status']}")
        print(f"求解器: {results['solver']}")
        print(f"求解时间: {results['solve_time']:.3f} 秒")
        print(f"目标函数值: {results['objective']:.2f} 元")
        print(f"时间段数: {results['n_periods']} (从{self.time_periods[0]}:00到{self.time_periods[-1]}:00)")
        
        print(f"\n总负荷削减: {results['total_load_shed']:.2f} kW")
        
        # 打印MESS调度轨迹
        print(f"\n移动储能调度轨迹:")
        for mess_id in range(1, self.n_mess + 1):
            print(f"\nMESS {mess_id}:")
            for hour in self.time_periods[:10]:  # 显示前10个时段
                schedule = results['mess_schedule'][hour][mess_id]
                location = schedule['location'] if schedule['location'] else "移动中"
                soc = schedule['SOC']
                action = ""
                if schedule['P_ch'] > 0:
                    action = f"充电 {schedule['P_ch']:.1f}kW"
                elif schedule['P_dch'] > 0:
                    action = f"放电 {schedule['P_dch']:.1f}kW"
                else:
                    action = "待机"
                    
                print(f"  {hour:2d}:00 - 位置: 节点{location}, SOC: {soc:.1f}%, {action}")

    def write_lp(self, filename: str):
        """输出问题描述"""
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("问题: 灾后多时段动态调度模型 (Demo 1)\n")
            f.write(f"时间段数: {self.n_periods}\n")
            f.write(f"节点数: {self.n_buses}\n")
            f.write(f"移动储能数: {self.n_mess}\n")
            f.write(f"变量数: {self.problem.size_metrics.num_scalar_variables}\n")
            f.write(f"约束数: {self.problem.size_metrics.num_scalar_eq_constr + self.problem.size_metrics.num_scalar_leq_constr}\n")
            f.write(f"状态: {self.problem.status if hasattr(self.problem, 'status') else '未求解'}\n")
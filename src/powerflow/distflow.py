"""
DistFlow配电网潮流计算
DistFlow Power Flow Calculation for Distribution Networks

基于论文B中式(13)-(20)的实现
Implementation based on equations (13)-(20) from Paper B
"""

import numpy as np
import cvxpy as cp
from typing import Dict, List, Tuple, Optional
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class DistFlowSolver:
    """DistFlow潮流求解器"""
    
    def __init__(self, branch_data: np.ndarray, load_data: np.ndarray, 
                 v_base: float = 12.66, v_min: float = 11.39, v_max: float = 13.92):
        """
        初始化求解器
        
        Args:
            branch_data: 支路数据 [from_bus, to_bus, r, x]
            load_data: 负荷数据 [bus_id, p_load, q_load]
            v_base: 基准电压 (kV)
            v_min: 最小电压 (kV)
            v_max: 最大电压 (kV)
        """
        self.branch_data = branch_data
        self.load_data = load_data
        self.v_base = v_base
        self.v_min = v_min
        self.v_max = v_max
        
        # 构建网络拓扑
        self.n_buses = int(max(branch_data[:, 1]))
        self.n_branches = len(branch_data)
        self._build_topology()
        
        logger.info(f"初始化完成: {self.n_buses}个节点, {self.n_branches}条支路")
        
    def _build_topology(self):
        """构建网络拓扑结构"""
        self.children = {i: [] for i in range(1, self.n_buses + 1)}
        self.parent = {i: None for i in range(1, self.n_buses + 1)}
        
        for from_bus, to_bus, _, _ in self.branch_data:
            from_bus, to_bus = int(from_bus), int(to_bus)
            self.children[from_bus].append(to_bus)
            self.parent[to_bus] = from_bus
    
    def solve_linear(self) -> Optional[Dict]:
        """
        求解线性化的DistFlow（用于测试）
        
        Returns:
            求解结果字典，包含电压、损耗等信息
        """
        logger.info("开始线性化DistFlow求解...")
        
        # 创建优化变量
        V = cp.Variable(self.n_buses + 1, pos=True)  # 电压幅值
        P = cp.Variable((self.n_buses + 1, self.n_buses + 1))  # 有功功率
        Q = cp.Variable((self.n_buses + 1, self.n_buses + 1))  # 无功功率
        
        constraints = []
        
        # 根节点电压约束
        constraints.append(V[1] == self.v_base)
        
        # 电压限制
        for i in range(2, self.n_buses + 1):
            constraints.append(V[i] >= self.v_min)
            constraints.append(V[i] <= self.v_max)
        
        # 线性化的电压降约束
        for from_bus, to_bus, r, x in self.branch_data:
            i, j = int(from_bus), int(to_bus)
            # 线性化: V_j ≈ V_i - (rP + xQ) / V_base
            constraints.append(
                V[j] == V[i] - (r * P[i,j] + x * Q[i,j]) / self.v_base
            )
        
        # 功率平衡约束
        for i in range(1, self.n_buses + 1):
            # 获取负荷
            load_idx = np.where(self.load_data[:, 0] == i)[0]
            if len(load_idx) > 0:
                p_load = self.load_data[load_idx[0], 1] / 1000  # MW
                q_load = self.load_data[load_idx[0], 2] / 1000  # Mvar
            else:
                p_load = q_load = 0
            
            # 功率平衡
            p_in = sum(P[k, i] for k in range(1, self.n_buses + 1) if self.parent[i] == k)
            q_in = sum(Q[k, i] for k in range(1, self.n_buses + 1) if self.parent[i] == k)
            p_out = sum(P[i, j] for j in self.children[i])
            q_out = sum(Q[i, j] for j in self.children[i])
            
            if i == 1:  # 根节点
                total_p = sum(self.load_data[:, 1]) / 1000
                total_q = sum(self.load_data[:, 2]) / 1000
                constraints.append(p_out == total_p)
                constraints.append(q_out == total_q)
            else:
                constraints.append(p_in == p_out + p_load)
                constraints.append(q_in == q_out + q_load)
        
        # 目标函数：最小化电压偏差
        objective = cp.Minimize(cp.sum_squares(V[2:] - self.v_base))
        
        # 求解
        problem = cp.Problem(objective, constraints)
        try:
            problem.solve(solver=cp.ECOS)
            
            if problem.status == cp.OPTIMAL:
                logger.info("线性化求解成功!")
                return {
                    'status': 'optimal',
                    'voltages': V.value[1:self.n_buses+1],
                    'P': P.value,
                    'Q': Q.value
                }
            else:
                logger.error(f"线性化求解失败: {problem.status}")
                return None
                
        except Exception as e:
            logger.error(f"求解出错: {e}")
            return None
    
    def solve_socp(self, verbose: bool = False) -> Optional[Dict]:
        """
        使用二阶锥松弛求解DistFlow
        
        Args:
            verbose: 是否显示求解器输出
            
        Returns:
            求解结果字典
        """
        logger.info("开始SOCP DistFlow求解...")
        
        # 创建优化变量
        V_sqr = cp.Variable(self.n_buses + 1, nonneg=True)  # V²
        P = cp.Variable((self.n_buses + 1, self.n_buses + 1))  # P_ij
        Q = cp.Variable((self.n_buses + 1, self.n_buses + 1))  # Q_ij
        I_sqr = cp.Variable((self.n_buses + 1, self.n_buses + 1), nonneg=True)  # I²_ij
        
        constraints = []
        
        # 1. 根节点电压
        constraints.append(V_sqr[1] == self.v_base**2)
        
        # 2. 电压限制
        for i in range(2, self.n_buses + 1):
            constraints.append(V_sqr[i] >= self.v_min**2)
            constraints.append(V_sqr[i] <= self.v_max**2)
        
        # 3. 支路约束
        branch_indices = {}  # 存储支路索引
        for idx, (from_bus, to_bus, r, x) in enumerate(self.branch_data):
            i, j = int(from_bus), int(to_bus)
            branch_indices[(i, j)] = idx
            
            # 电压降方程
            constraints.append(
                V_sqr[j] == V_sqr[i] - 2*(r*P[i,j] + x*Q[i,j]) + (r**2 + x**2)*I_sqr[i,j]
            )
            
            # 二阶锥约束
            constraints.append(
                cp.norm(cp.vstack([2*P[i,j], 2*Q[i,j], I_sqr[i,j] - V_sqr[i]])) 
                <= I_sqr[i,j] + V_sqr[i]
            )
        
        # 4. 功率平衡
        for i in range(1, self.n_buses + 1):
            # 负荷
            load_idx = np.where(self.load_data[:, 0] == i)[0]
            if len(load_idx) > 0:
                p_load = self.load_data[load_idx[0], 1] / 1000  # MW
                q_load = self.load_data[load_idx[0], 2] / 1000  # Mvar
            else:
                p_load = q_load = 0
            
            # 流入功率（考虑损耗）
            p_in = 0
            q_in = 0
            if self.parent[i] is not None:
                parent = self.parent[i]
                if (parent, i) in branch_indices:
                    idx = branch_indices[(parent, i)]
                    r = self.branch_data[idx, 2]
                    x = self.branch_data[idx, 3]
                    p_in = P[parent, i] - r * I_sqr[parent, i]
                    q_in = Q[parent, i] - x * I_sqr[parent, i]
            
            # 流出功率
            p_out = sum(P[i, j] for j in self.children[i])
            q_out = sum(Q[i, j] for j in self.children[i])
            
            # 功率平衡约束
            if i == 1:  # 根节点
                # 考虑损耗的总功率
                total_p = sum(self.load_data[:, 1]) / 1000
                total_q = sum(self.load_data[:, 2]) / 1000
                # 添加损耗估计（约5%）
                constraints.append(p_out >= total_p)
                constraints.append(p_out <= 1.1 * total_p)
                constraints.append(q_out >= total_q)
                constraints.append(q_out <= 1.2 * total_q)
            else:
                constraints.append(p_in == p_out + p_load)
                constraints.append(q_in == q_out + q_load)
        
        # 目标函数：最小化损耗
        losses = 0
        for (i, j), idx in branch_indices.items():
            r = self.branch_data[idx, 2]
            losses += r * I_sqr[i, j]
        
        objective = cp.Minimize(losses)
        problem = cp.Problem(objective, constraints)
        
        # 求解
        try:
            problem.solve(solver=cp.SCS, verbose=verbose)
            
            if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                logger.info(f"SOCP求解成功! 状态: {problem.status}")
                
                # 提取结果
                voltages = np.sqrt(V_sqr.value[1:self.n_buses+1])
                losses_kw = problem.value * 1000
                
                logger.info(f"总损耗: {losses_kw:.2f} kW")
                logger.info(f"电压范围: [{voltages.min():.3f}, {voltages.max():.3f}] kV")
                
                return {
                    'status': problem.status,
                    'voltages': voltages,
                    'losses': losses_kw,
                    'V_sqr': V_sqr.value,
                    'P': P.value * 1000,  # 转换为kW
                    'Q': Q.value * 1000,  # 转换为kvar
                    'I_sqr': I_sqr.value
                }
            else:
                logger.error(f"SOCP求解失败: {problem.status}")
                return None
                
        except Exception as e:
            logger.error(f"求解出错: {e}")
            import traceback
            traceback.print_exc()
            return None


def load_ieee33_data():
    """加载IEEE 33节点测试系统数据"""
    # 支路数据
    branch_data = np.array([
        [1, 2, 0.0922, 0.0470],
        [2, 3, 0.4930, 0.2511],
        [3, 4, 0.3660, 0.1864],
        [4, 5, 0.3811, 0.1941],
        [5, 6, 0.8190, 0.7070],
        [6, 7, 0.1872, 0.6188],
        [7, 8, 0.7114, 0.2351],
        [8, 9, 1.0300, 0.7400],
        [9, 10, 1.0440, 0.7400],
        [10, 11, 0.1966, 0.0650],
        [11, 12, 0.3744, 0.1238],
        [12, 13, 1.4680, 1.1550],
        [13, 14, 0.5416, 0.7129],
        [14, 15, 0.5910, 0.5260],
        [15, 16, 0.7463, 0.5450],
        [16, 17, 1.2890, 1.7210],
        [17, 18, 0.7320, 0.5740],
        [2, 19, 0.1640, 0.1565],
        [19, 20, 1.5042, 1.3554],
        [20, 21, 0.4095, 0.4784],
        [21, 22, 0.7089, 0.9373],
        [3, 23, 0.4512, 0.3083],
        [23, 24, 0.8980, 0.7091],
        [24, 25, 0.8960, 0.7011],
        [6, 26, 0.2030, 0.1034],
        [26, 27, 0.2842, 0.1447],
        [27, 28, 1.0590, 0.9337],
        [28, 29, 0.8042, 0.7006],
        [29, 30, 0.5075, 0.2585],
        [30, 31, 0.9744, 0.9630],
        [31, 32, 0.3105, 0.3619],
        [32, 33, 0.3410, 0.5302]
    ])
    
    # 负荷数据
    load_data = np.array([
        [1, 0, 0], [2, 100, 60], [3, 90, 40], [4, 120, 80], [5, 60, 30],
        [6, 60, 20], [7, 200, 100], [8, 200, 100], [9, 60, 20], [10, 60, 20],
        [11, 45, 30], [12, 60, 35], [13, 60, 35], [14, 120, 80], [15, 60, 10],
        [16, 60, 20], [17, 60, 20], [18, 90, 40], [19, 90, 40], [20, 90, 40],
        [21, 90, 40], [22, 90, 40], [23, 90, 50], [24, 420, 200], [25, 420, 200],
        [26, 60, 25], [27, 60, 25], [28, 40, 20], [29, 120, 70], [30, 200, 600],
        [31, 150, 70], [32, 210, 100], [33, 60, 40]
    ])
    
    return branch_data, load_data


def main():
    """主函数"""
    # 加载数据
    branch_data, load_data = load_ieee33_data()
    
    # 系统参数
    v_base = 12.66  # kV
    v_min = 11.39   # kV
    v_max = 13.92   # kV
    
    # 创建求解器
    solver = DistFlowSolver(branch_data, load_data, v_base, v_min, v_max)
    
    # 1. 先尝试线性化求解
    print("\n" + "="*50)
    print("1. 线性化DistFlow求解")
    print("="*50)
    result_linear = solver.solve_linear()
    
    if result_linear:
        print(f"最低电压: {result_linear['voltages'].min():.3f} kV")
        print(f"最高电压: {result_linear['voltages'].max():.3f} kV")
    
    # 2. 尝试SOCP求解
    print("\n" + "="*50)
    print("2. SOCP DistFlow求解")
    print("="*50)
    result_socp = solver.solve_socp(verbose=True)
    
    if result_socp:
        print(f"\n求解成功!")
        print(f"总损耗: {result_socp['losses']:.2f} kW")
        print(f"损耗率: {result_socp['losses']/sum(load_data[:, 1])*100:.2f}%")
        print(f"最低电压: {result_socp['voltages'].min():.3f} kV")
        print(f"最高电压: {result_socp['voltages'].max():.3f} kV")
        
        # 显示部分节点电压
        print("\n部分节点电压:")
        for i in [1, 18, 25, 30, 33]:
            print(f"  节点 {i:2d}: {result_socp['voltages'][i-1]:.3f} kV")


if __name__ == "__main__":
    main()
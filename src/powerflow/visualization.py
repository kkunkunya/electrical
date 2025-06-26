"""
潮流计算结果可视化
Power Flow Result Visualization
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from typing import Dict, Optional

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 黑体
plt.rcParams['axes.unicode_minus'] = False


def plot_voltage_profile(result: Dict, v_base: float = 12.66, 
                        v_min: float = 11.39, v_max: float = 13.92,
                        save_path: Optional[str] = None):
    """
    绘制电压分布图
    
    Args:
        result: 潮流计算结果
        v_base: 基准电压
        v_min: 最小电压
        v_max: 最大电压
        save_path: 保存路径
    """
    voltages = result['voltages']
    n_buses = len(voltages)
    bus_numbers = np.arange(1, n_buses + 1)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # 1. 电压幅值
    ax1.plot(bus_numbers, voltages, 'b.-', linewidth=2, markersize=8, label='节点电压')
    ax1.axhline(y=v_base, color='g', linestyle='-', label=f'额定电压 {v_base} kV', alpha=0.7)
    ax1.axhline(y=v_min, color='r', linestyle='--', label=f'下限 {v_min} kV', alpha=0.7)
    ax1.axhline(y=v_max, color='r', linestyle='--', label=f'上限 {v_max} kV', alpha=0.7)
    ax1.fill_between(bus_numbers, v_min, v_max, alpha=0.2, color='green')
    
    ax1.set_xlabel('节点编号')
    ax1.set_ylabel('电压 (kV)')
    ax1.set_title('节点电压分布')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, n_buses + 1)
    
    # 2. 电压偏差
    voltage_deviation = (voltages - v_base) / v_base * 100
    colors = ['red' if abs(v) > 10 else 'blue' for v in voltage_deviation]
    ax2.bar(bus_numbers, voltage_deviation, color=colors, alpha=0.7)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.axhline(y=10, color='red', linestyle='--', alpha=0.5, label='±10% 限值')
    ax2.axhline(y=-10, color='red', linestyle='--', alpha=0.5)
    
    ax2.set_xlabel('节点编号')
    ax2.set_ylabel('电压偏差 (%)')
    ax2.set_title('电压偏差百分比')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, n_buses + 1)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_power_flow(result: Dict, branch_data: np.ndarray, 
                   save_path: Optional[str] = None):
    """
    绘制功率流分布
    
    Args:
        result: 潮流计算结果
        branch_data: 支路数据
        save_path: 保存路径
    """
    P = result['P']  # kW
    Q = result['Q']  # kvar
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 提取支路功率
    branch_powers = []
    branch_labels = []
    for i, (from_bus, to_bus, _, _) in enumerate(branch_data):
        from_bus, to_bus = int(from_bus), int(to_bus)
        p_flow = P[from_bus, to_bus] if isinstance(P, np.ndarray) else 0
        q_flow = Q[from_bus, to_bus] if isinstance(Q, np.ndarray) else 0
        s_flow = np.sqrt(p_flow**2 + q_flow**2)
        
        branch_powers.append(s_flow)
        branch_labels.append(f"{from_bus}-{to_bus}")
    
    # 1. 支路视在功率
    branch_indices = np.arange(len(branch_powers))
    ax1.bar(branch_indices, branch_powers, alpha=0.7)
    ax1.set_xlabel('支路')
    ax1.set_ylabel('视在功率 (kVA)')
    ax1.set_title('支路功率流')
    ax1.set_xticks(branch_indices[::4])
    ax1.set_xticklabels(branch_labels[::4], rotation=45)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 2. 损耗分布
    losses = []
    loss_labels = []
    for i, (from_bus, to_bus, r, x) in enumerate(branch_data):
        from_bus, to_bus = int(from_bus), int(to_bus)
        if 'I_sqr' in result:
            i_sqr = result['I_sqr'][from_bus, to_bus]
            loss = r * i_sqr * 1000  # kW
            if loss > 0.1:  # 只显示大于0.1kW的损耗
                losses.append(loss)
                loss_labels.append(f"{from_bus}-{to_bus}")
    
    # 按损耗排序
    if losses:
        sorted_indices = np.argsort(losses)[::-1][:15]  # 前15个
        sorted_losses = [losses[i] for i in sorted_indices]
        sorted_labels = [loss_labels[i] for i in sorted_indices]
        
        ax2.barh(range(len(sorted_losses)), sorted_losses, alpha=0.7)
        ax2.set_yticks(range(len(sorted_losses)))
        ax2.set_yticklabels(sorted_labels)
        ax2.set_xlabel('功率损耗 (kW)')
        ax2.set_title('前15条支路损耗')
        ax2.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def print_summary(result: Dict, load_data: np.ndarray):
    """
    打印结果摘要
    
    Args:
        result: 潮流计算结果
        load_data: 负荷数据
    """
    voltages = result['voltages']
    losses = result['losses']
    total_load = sum(load_data[:, 1])
    
    print("\n" + "="*60)
    print("潮流计算结果摘要")
    print("="*60)
    
    print(f"\n系统概况:")
    print(f"  节点数: {len(voltages)}")
    print(f"  总负荷: {total_load:.0f} kW")
    print(f"  总损耗: {losses:.2f} kW")
    print(f"  损耗率: {losses/total_load*100:.2f}%")
    
    print(f"\n电压统计:")
    print(f"  最低电压: {voltages.min():.3f} kV (节点 {np.argmin(voltages)+1})")
    print(f"  最高电压: {voltages.max():.3f} kV (节点 {np.argmax(voltages)+1})")
    print(f"  平均电压: {voltages.mean():.3f} kV")
    print(f"  电压标准差: {voltages.std():.3f} kV")
    
    # 找出电压最低的5个节点
    lowest_indices = np.argsort(voltages)[:5]
    print(f"\n电压最低的5个节点:")
    for idx in lowest_indices:
        bus = idx + 1
        v = voltages[idx]
        v_pu = v / 12.66
        print(f"  节点 {bus:2d}: {v:.3f} kV ({v_pu:.4f} p.u.)")
    
    # 功率平衡验证
    if 'P' in result:
        P = result['P']
        # 根节点注入功率（假设从节点1流出）
        p_gen = 0
        for j in range(2, len(voltages)+1):
            if isinstance(P, np.ndarray) and P[1, j] != 0:
                p_gen += P[1, j]
        
        print(f"\n功率平衡:")
        print(f"  总发电: {p_gen:.2f} kW")
        print(f"  总负荷: {total_load:.2f} kW")
        print(f"  总损耗: {losses:.2f} kW")
        print(f"  平衡误差: {abs(p_gen - total_load - losses):.3f} kW")


def create_report(result: Dict, branch_data: np.ndarray, load_data: np.ndarray,
                 output_dir: str = "output"):
    """
    创建完整的分析报告
    
    Args:
        result: 潮流计算结果
        branch_data: 支路数据
        load_data: 负荷数据
        output_dir: 输出目录
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # 打印摘要
    print_summary(result, load_data)
    
    # 生成图表
    plot_voltage_profile(result, save_path=f"{output_dir}/voltage_profile.png")
    plot_power_flow(result, branch_data, save_path=f"{output_dir}/power_flow.png")
    
    # 保存详细结果到CSV
    import pandas as pd
    
    # 节点电压
    voltage_df = pd.DataFrame({
        'Bus': range(1, len(result['voltages'])+1),
        'Voltage_kV': result['voltages'],
        'Voltage_pu': result['voltages'] / 12.66,
        'Deviation_%': (result['voltages'] - 12.66) / 12.66 * 100
    })
    voltage_df.to_csv(f"{output_dir}/bus_voltages.csv", index=False)
    
    print(f"\n结果已保存到 {output_dir} 目录")


if __name__ == "__main__":
    # 测试可视化
    from distflow import load_ieee33_data, DistFlowSolver
    
    branch_data, load_data = load_ieee33_data()
    solver = DistFlowSolver(branch_data, load_data)
    result = solver.solve_socp(verbose=False)
    
    if result:
        create_report(result, branch_data, load_data)
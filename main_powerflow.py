"""
主程序：运行DistFlow潮流计算并生成报告
Main program: Run DistFlow power flow and generate report
"""

import os
import sys
import logging
from datetime import datetime

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.powerflow.distflow import DistFlowSolver, load_ieee33_data
from src.powerflow.visualization import create_report, print_summary

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def main():
    """主函数"""
    logger.info("="*60)
    logger.info("IEEE 33节点配电网DistFlow潮流计算")
    logger.info("="*60)
    
    # 1. 加载数据
    logger.info("\n1. 加载系统数据...")
    branch_data, load_data = load_ieee33_data()
    
    # 系统参数
    v_base = 12.66  # kV
    v_min = 11.39   # kV  
    v_max = 13.92   # kV
    
    logger.info(f"  节点数: 33")
    logger.info(f"  支路数: {len(branch_data)}")
    logger.info(f"  总负荷: {sum(load_data[:, 1]):.0f} kW + j{sum(load_data[:, 2]):.0f} kvar")
    
    # 2. 创建求解器
    logger.info("\n2. 初始化求解器...")
    solver = DistFlowSolver(branch_data, load_data, v_base, v_min, v_max)
    
    # 3. 求解潮流
    logger.info("\n3. 求解DistFlow潮流...")
    result = solver.solve_socp(verbose=False)
    
    if result:
        logger.info("潮流求解成功!")
        
        # 4. 生成报告
        logger.info("\n4. 生成分析报告...")
        
        # 创建输出目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"output/powerflow_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        
        # 打印摘要
        print_summary(result, load_data)
        
        # 保存结果
        import numpy as np
        np.savez(f"{output_dir}/powerflow_result.npz",
                 voltages=result['voltages'],
                 losses=result['losses'],
                 V_sqr=result['V_sqr'],
                 P=result['P'],
                 Q=result['Q'],
                 I_sqr=result['I_sqr'])
        
        logger.info(f"\n结果已保存到: {output_dir}")
        
        # 验证结果
        logger.info("\n5. 验证结果...")
        voltages = result['voltages']
        
        # 电压约束检查
        voltage_violations = []
        for i, v in enumerate(voltages):
            if v < v_min or v > v_max:
                voltage_violations.append((i+1, v))
        
        if not voltage_violations:
            logger.info("  ✓ 所有节点电压满足约束")
        else:
            logger.warning(f"  ✗ {len(voltage_violations)}个节点电压越限")
        
        # 功率平衡检查
        total_gen = result['P'][1, 2]  # 从根节点流出
        total_load = sum(load_data[:, 1])
        total_loss = result['losses']
        balance_error = abs(total_gen - total_load - total_loss)
        
        logger.info(f"  功率平衡误差: {balance_error:.3f} kW")
        if balance_error < 1.0:
            logger.info("  ✓ 功率平衡验证通过")
        else:
            logger.warning("  ✗ 功率平衡误差较大")
        
        # 生成可视化（如果需要）
        try:
            from src.powerflow.visualization import plot_voltage_profile, plot_power_flow
            logger.info("\n6. 生成可视化图表...")
            plot_voltage_profile(result, v_base, v_min, v_max, 
                               save_path=f"{output_dir}/voltage_profile.png")
            plot_power_flow(result, branch_data, 
                           save_path=f"{output_dir}/power_flow.png")
            logger.info("  图表已保存")
        except ImportError:
            logger.warning("  无法导入matplotlib，跳过可视化")
        
        return result
    else:
        logger.error("潮流求解失败!")
        return None


if __name__ == "__main__":
    result = main()
    
    if result:
        print("\n" + "="*60)
        print("程序执行成功!")
        print("="*60)
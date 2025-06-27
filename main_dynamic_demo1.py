#!/usr/bin/env python
"""
Demo 1 主程序 - 多时段动态调度模型测试
复现文章B的移动储能时空动态调度模型
"""

import sys
import os
import logging
from pathlib import Path
from datetime import datetime

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.models.post_disaster_dynamic_multi_period import PostDisasterDynamicModel
from src.datasets.loader import load_system_data


def setup_logging(output_dir: Path) -> logging.Logger:
    """设置日志系统"""
    log_file = output_dir / "demo1_execution.log"
    
    # 配置日志格式
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("="*80)
    logger.info("启动 Demo 1: 多时段动态调度模型")
    logger.info("="*80)
    
    return logger


def create_output_directory() -> Path:
    """创建输出目录"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = project_root / "output" / f"demo1_dynamic_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def load_data(logger: logging.Logger) -> tuple:
    """加载系统数据"""
    logger.info("加载系统数据...")
    
    try:
        # 数据文件路径
        data_dir = project_root / "data"
        
        # 检查数据文件是否存在
        required_files = ["loads.yaml", "branches.yaml", "generators.yaml"]
        for file in required_files:
            file_path = data_dir / file
            if not file_path.exists():
                logger.error(f"数据文件不存在: {file_path}")
                return None, None, None
        
        # 加载系统数据
        system_data = load_system_data(str(data_dir))
        
        # 时变数据路径
        traffic_file = str(data_dir / "traffic_profile.csv")
        pv_file = str(data_dir / "pv_profile.csv")
        
        logger.info("系统数据加载成功")
        logger.info(f"  节点数: {len(system_data.loads) + 1}")
        logger.info(f"  支路数: {len(system_data.branches)}")
        logger.info(f"  发电机数: {len(system_data.generators)}")
        
        return system_data, traffic_file, pv_file
        
    except Exception as e:
        logger.error(f"数据加载失败: {e}")
        return None, None, None


def run_demo1(logger: logging.Logger, output_dir: Path):
    """运行Demo 1主程序"""
    logger.info("开始运行Demo 1: 多时段动态调度")
    
    # 1. 加载数据
    system_data, traffic_file, pv_file = load_data(logger)
    if system_data is None:
        logger.error("数据加载失败，程序退出")
        return False
    
    # 2. 创建动态调度模型
    logger.info("创建多时段动态调度模型...")
    try:
        model = PostDisasterDynamicModel(
            system_data=system_data,
            n_periods=21,  # 21个时段 (3:00-23:00)  
            start_hour=3,
            traffic_profile_path=traffic_file,
            pv_profile_path=pv_file
        )
        logger.info("模型创建成功")
        
        # 打印模型规模信息
        logger.info(f"模型规模:")
        logger.info(f"  时间段数: {model.n_periods}")
        logger.info(f"  节点数: {model.n_buses}")
        logger.info(f"  移动储能数: {model.n_mess}")
        
    except Exception as e:
        logger.error(f"模型创建失败: {e}")
        return False
    
    # 3. 求解模型
    logger.info("开始求解优化模型...")
    try:
        # 设置求解器参数
        solver_params = {
            'max_iters': 10000,
            'verbose': True
        }
        
        results = model.solve(solver=None, **solver_params)
        
        if results is None:
            logger.error("模型求解失败")
            return False
            
        logger.info("模型求解成功")
        
    except Exception as e:
        logger.error(f"求解过程出错: {e}")
        return False
    
    # 4. 保存结果
    logger.info("保存结果...")
    try:
        # 保存详细结果到文件
        import json
        results_file = output_dir / "demo1_results.json"
        
        # 处理numpy数组，使其可以JSON序列化
        serializable_results = {}
        for key, value in results.items():
            if isinstance(value, (list, dict, str, int, float, bool)):
                serializable_results[key] = value
            else:
                serializable_results[key] = str(value)
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, ensure_ascii=False, indent=2)
        
        # 保存模型描述
        lp_file = output_dir / "demo1_model.lp"
        model.write_lp(str(lp_file))
        
        logger.info(f"结果已保存到: {output_dir}")
        
    except Exception as e:
        logger.error(f"结果保存失败: {e}")
        return False
    
    # 5. 生成报告
    generate_report(results, output_dir, logger)
    
    logger.info("Demo 1 执行完成")
    return True


def generate_report(results: dict, output_dir: Path, logger: logging.Logger):
    """生成结果报告"""
    logger.info("生成结果报告...")
    
    try:
        report_file = output_dir / "demo1_report.txt"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("Demo 1: 多时段动态调度模型 - 结果报告\n")
            f.write("="*80 + "\n\n")
            
            # 基本信息
            f.write(f"求解状态: {results.get('status', 'unknown')}\n")
            f.write(f"求解器: {results.get('solver', 'unknown')}\n")
            f.write(f"求解时间: {results.get('solve_time', 0):.3f} 秒\n")
            f.write(f"目标函数值: {results.get('objective', 0):.2f} 元\n\n")
            
            # 时间维度信息
            f.write(f"仿真时间段: {results.get('n_periods', 0)} 小时\n")
            f.write(f"时间范围: {results.get('time_periods', [])}\n\n")
            
            # 总体统计
            f.write(f"总负荷削减: {results.get('total_load_shed', 0):.2f} kW\n\n")
            
            # 移动储能调度轨迹摘要
            f.write("移动储能调度摘要:\n")
            mess_schedule = results.get('mess_schedule', {})
            if mess_schedule:
                for hour in sorted(list(mess_schedule.keys())[:5]):  # 显示前5个时段
                    f.write(f"  {hour}:00 时段:\n")
                    for mess_id, schedule in mess_schedule[hour].items():
                        location = schedule.get('location', '未知')
                        soc = schedule.get('SOC', 0)
                        f.write(f"    MESS{mess_id}: 位置=节点{location}, SOC={soc:.1f}%\n")
            
            f.write("\n报告生成完成。\n")
            f.write(f"详细结果请查看: demo1_results.json\n")
        
        logger.info(f"报告已生成: {report_file}")
        
    except Exception as e:
        logger.error(f"报告生成失败: {e}")


def main():
    """主函数"""
    print("="*80)
    print("电力系统灾后多时段动态调度优化 - Demo 1")
    print("实现移动储能时空动态调度")
    print("="*80)
    
    # 创建输出目录
    output_dir = create_output_directory()
    
    # 设置日志
    logger = setup_logging(output_dir)
    
    try:
        # 运行Demo 1
        success = run_demo1(logger, output_dir)
        
        if success:
            print(f"\n✅ Demo 1 成功完成!")
            print(f"📁 结果保存在: {output_dir}")
            print("\n主要输出文件:")
            print(f"  - demo1_results.json     # 详细结果数据")
            print(f"  - demo1_report.txt       # 结果摘要报告")
            print(f"  - demo1_model.lp         # 模型描述文件")
            print(f"  - demo1_execution.log    # 执行日志")
        else:
            print(f"\n❌ Demo 1 执行失败")
            print(f"📁 查看日志: {output_dir / 'demo1_execution.log'}")
            
    except KeyboardInterrupt:
        logger.info("用户中断程序执行")
        print("\n程序被用户中断")
    except Exception as e:
        logger.error(f"程序执行出现未预期错误: {e}")
        print(f"\n❌ 程序执行出错: {e}")
    finally:
        logger.info("程序结束")


if __name__ == "__main__":
    main()
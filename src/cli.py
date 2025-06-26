"""
命令行接口
提供项目的统一入口
"""

import argparse
import logging
from pathlib import Path
import sys
import numpy as np

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.datasets.loader import load_system_data
from src.models.post_disaster_static import PostDisasterStaticModel
from src.models.post_disaster_static_v2 import PostDisasterStaticModelV2
from src.powerflow.distflow import DistFlowSolver
from src.powerflow import visualization


def setup_logging(verbose: bool = False):
    """设置日志"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def cmd_distflow(args):
    """运行DistFlow潮流计算"""
    # 加载数据
    data = load_system_data(args.data_dir)
    
    # 创建求解器
    solver = DistFlowSolver(data)
    
    # 求解
    results = solver.solve_socp(verbose=args.verbose)
    
    if results:
        print("潮流计算成功完成!")
        
        # 可视化
        if args.plot:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(exist_ok=True)
            
            # 转换branch数据格式
            branch_array = []
            for b in data.branches:
                branch_array.append([b.from_bus, b.to_bus, b.resistance, b.reactance])
            branch_array = np.array(branch_array)
            
            visualization.plot_voltage_profile(
                results, 
                save_path=str(output_dir / "voltage_profile.png")
            )
            visualization.plot_power_flow(
                results, 
                branch_array,
                save_path=str(output_dir / "power_flow.png")
            )
            print(f"图表已保存到: {output_dir}")
    else:
        print("潮流计算失败!")
        

def cmd_post_static(args):
    """运行灾后静态调度优化"""
    # 加载数据
    if args.case:
        # 从case文件加载（未来实现）
        print(f"从case文件加载: {args.case}")
        data_dir = Path(args.case).parent
    else:
        data_dir = Path(args.data_dir)
        
    data = load_system_data(str(data_dir))
    
    # 根据版本选择模型
    if args.version == 2:
        print("使用Demo 2模型（含固定MESS和DEG）")
        model = PostDisasterStaticModelV2(data)
    else:
        print("使用Demo 1模型（基础版本）")
        model = PostDisasterStaticModel(data)
    
    # 求解
    results = model.solve(verbose=args.verbose)
    
    if results:
        print(f"\n求解成功!")
        print(f"Objective = {results['objective']:.2f} 元")
        print(f"Max V = {results['max_voltage']:.3f} kV")
        
        # 输出LP文件（CVXPY版本）
        if args.out:
            model.write_lp(args.out)
            print(f"\n问题描述已保存到: {args.out}")
    else:
        print("求解失败!")
        

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="配电网灾害优化项目CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='显示详细信息'
    )
    
    # 子命令
    subparsers = parser.add_subparsers(
        title='子命令',
        dest='command',
        help='可用的子命令'
    )
    
    # distflow子命令
    parser_distflow = subparsers.add_parser(
        'distflow',
        help='运行DistFlow潮流计算'
    )
    parser_distflow.add_argument(
        '--data-dir',
        default='data',
        help='数据目录路径 (默认: data)'
    )
    parser_distflow.add_argument(
        '--output-dir',
        default='output',
        help='输出目录路径 (默认: output)'
    )
    parser_distflow.add_argument(
        '--plot',
        action='store_true',
        help='生成可视化图表'
    )
    parser_distflow.set_defaults(func=cmd_distflow)
    
    # post-static子命令
    parser_post_static = subparsers.add_parser(
        'post-static',
        help='运行灾后静态调度优化'
    )
    parser_post_static.add_argument(
        '--case',
        help='case文件路径 (例如: data/ieee33.yaml)'
    )
    parser_post_static.add_argument(
        '--data-dir',
        default='data',
        help='数据目录路径 (默认: data)'
    )
    parser_post_static.add_argument(
        '--out',
        help='输出LP文件路径 (例如: snapshot.lp)'
    )
    parser_post_static.add_argument(
        '--version',
        type=int,
        choices=[1, 2],
        default=1,
        help='模型版本 (1: Demo1基础版, 2: Demo2含MESS和DEG)'
    )
    parser_post_static.set_defaults(func=cmd_post_static)
    
    # 解析参数
    args = parser.parse_args()
    
    # 设置日志
    setup_logging(args.verbose)
    
    # 执行命令
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
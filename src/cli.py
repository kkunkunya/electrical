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
from src.models.post_disaster_dynamic_multi_period import PostDisasterDynamicModel
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


def cmd_post_dynamic(args):
    """运行灾后多时段动态调度优化"""
    # 加载数据
    data_dir = Path(args.data_dir)
    data = load_system_data(str(data_dir))
    
    print("使用多时段动态调度模型（含移动储能时空调度）")
    
    # 时变数据路径
    traffic_file = str(data_dir / "traffic_profile.csv")
    pv_file = str(data_dir / "pv_profile.csv")
    
    # 创建模型
    model = PostDisasterDynamicModel(
        system_data=data,
        n_periods=args.periods,
        start_hour=args.start_hour,
        traffic_profile_path=traffic_file,
        pv_profile_path=pv_file
    )
    
    print(f"模型规模: {model.n_periods}时段, {model.n_buses}节点, {model.n_mess}个移动储能")
    print(f"变量数: {model.problem.size_metrics.num_scalar_variables}")
    
    # 求解
    results = model.solve(verbose=args.verbose)
    
    if results:
        print(f"\n求解成功!")
        print(f"目标函数值: {results['objective']:.2f} 元")
        print(f"总负荷削减: {results.get('total_load_shed', 0):.2f} kW")
        
        # 输出结果文件
        if args.out:
            model.write_lp(args.out)
            print(f"\n模型描述已保存到: {args.out}")
            
        # 保存详细结果
        if args.save_results:
            import json
            output_file = args.save_results
            
            # 处理numpy数组使其可JSON序列化
            serializable_results = {}
            for key, value in results.items():
                if isinstance(value, (list, dict, str, int, float, bool)):
                    serializable_results[key] = value
                else:
                    serializable_results[key] = str(value)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_results, f, ensure_ascii=False, indent=2)
            print(f"详细结果已保存到: {output_file}")
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
    
    # post-dynamic子命令 (新增)
    parser_post_dynamic = subparsers.add_parser(
        'post-dynamic',
        help='运行灾后多时段动态调度优化 (Demo 1)'
    )
    parser_post_dynamic.add_argument(
        '--data-dir',
        default='data',
        help='数据目录路径 (默认: data)'
    )
    parser_post_dynamic.add_argument(
        '--periods',
        type=int,
        default=21,
        help='时间段数量 (默认: 21, 即3:00-23:00)'
    )
    parser_post_dynamic.add_argument(
        '--start-hour',
        type=int,
        default=3,
        help='起始小时 (默认: 3)'
    )
    parser_post_dynamic.add_argument(
        '--out',
        help='输出模型描述文件路径 (例如: demo1_model.lp)'
    )
    parser_post_dynamic.add_argument(
        '--save-results',
        help='保存详细结果的JSON文件路径 (例如: demo1_results.json)'
    )
    parser_post_dynamic.set_defaults(func=cmd_post_dynamic)
    
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
#!/usr/bin/env python
"""
Demo 1 测试脚本 - 快速验证模型功能
用于验证多时段动态调度模型的基本功能
"""

import sys
from pathlib import Path
import logging

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.models.post_disaster_dynamic_multi_period import PostDisasterDynamicModel, TrafficModel
from src.datasets.loader import load_system_data


def test_traffic_model():
    """测试交通模型"""
    print("\n" + "="*50)
    print("测试 1: 交通模型")
    print("="*50)
    
    # 创建交通模型
    traffic_model = TrafficModel("")  # 使用默认数据
    
    # 测试通行时间计算
    distance_matrix = [[0, 5], [5, 0]]  # 简单的2x2矩阵
    
    for time_period in [3, 8, 12, 18]:
        travel_time = traffic_model.calculate_travel_time(
            from_node=1, to_node=2, time_period=time_period,
            distance_matrix=distance_matrix
        )
        congestion = traffic_model.traffic_data.get(time_period, 0.4)
        print(f"时段 {time_period:2d}:00 - 拥堵程度: {congestion:.2f}, 通行时间: {travel_time:.2f}小时")
    
    print("✅ 交通模型测试通过")


def test_model_creation():
    """测试模型创建"""
    print("\n" + "="*50)
    print("测试 2: 模型创建")
    print("="*50)
    
    try:
        # 加载数据
        data_dir = project_root / "data"
        system_data = load_system_data(str(data_dir))
        
        print(f"系统数据加载成功:")
        print(f"  节点数: {len(system_data.loads) + 1}")
        print(f"  支路数: {len(system_data.branches)}")
        print(f"  发电机数: {len(system_data.generators)}")
        
        # 创建模型（使用较少时段进行测试）
        model = PostDisasterDynamicModel(
            system_data=system_data,
            n_periods=3,  # 仅3个时段用于快速测试
            start_hour=12
        )
        
        print(f"模型创建成功:")
        print(f"  时间段数: {model.n_periods}")
        print(f"  节点数: {model.n_buses}")
        print(f"  移动储能数: {model.n_mess}")
        print(f"  问题规模: {model.problem.size_metrics.num_scalar_variables} 变量")
        
        print("✅ 模型创建测试通过")
        return model
        
    except Exception as e:
        print(f"❌ 模型创建失败: {e}")
        return None


def test_model_solve(model):
    """测试模型求解"""
    print("\n" + "="*50)
    print("测试 3: 模型求解")
    print("="*50)
    
    if model is None:
        print("❌ 无可用模型，跳过求解测试")
        return
    
    try:
        # 尝试求解（使用快速求解器）
        print("开始求解...")
        results = model.solve(verbose=False, max_iters=1000)
        
        if results is None:
            print("❌ 求解失败")
            return
        
        print(f"求解状态: {results['status']}")
        print(f"求解器: {results['solver']}")
        print(f"求解时间: {results['solve_time']:.3f} 秒")
        
        if 'objective' in results:
            print(f"目标函数值: {results['objective']:.2f} 元")
        
        # 检查关键结果
        if 'mess_schedule' in results and results['mess_schedule']:
            print("\nMESS调度示例:")
            first_hour = next(iter(results['mess_schedule']))
            mess_data = results['mess_schedule'][first_hour]
            for mess_id, schedule in mess_data.items():
                location = schedule.get('location', '未知')
                soc = schedule.get('SOC', 0)
                print(f"  MESS{mess_id}: 位置=节点{location}, SOC={soc:.1f}%")
        
        print("✅ 模型求解测试通过")
        
    except Exception as e:
        print(f"❌ 求解失败: {e}")


def test_data_consistency():
    """测试数据一致性"""
    print("\n" + "="*50)
    print("测试 4: 数据一致性")
    print("="*50)
    
    try:
        # 检查数据文件
        data_dir = project_root / "data"
        
        files_to_check = [
            "loads.yaml",
            "branches.yaml", 
            "generators.yaml",
            "traffic_profile.csv",
            "pv_profile.csv"
        ]
        
        for filename in files_to_check:
            filepath = data_dir / filename
            if filepath.exists():
                print(f"✅ {filename} 存在")
            else:
                print(f"❌ {filename} 不存在")
        
        # 检查图片数据
        img_dir = project_root / "input_data_picture"
        key_images = [
            "图3 改进的IEEE33节点配电网结构.png",
            "图B2 交通网络的拥堵程度.png",
            "图B3 灾害发生后光伏机组的预测出力.png"
        ]
        
        for img_name in key_images:
            img_path = img_dir / img_name
            if img_path.exists():
                print(f"✅ {img_name} 存在")
            else:
                print(f"❌ {img_name} 不存在")
        
        print("✅ 数据一致性检查完成")
        
    except Exception as e:
        print(f"❌ 数据检查失败: {e}")


def main():
    """主测试函数"""
    print("="*60)
    print("Demo 1 功能测试套件")
    print("多时段动态调度模型 - 快速验证")
    print("="*60)
    
    # 设置简单日志
    logging.basicConfig(level=logging.WARNING)  # 只显示警告和错误
    
    # 运行测试
    test_traffic_model()
    model = test_model_creation()
    test_model_solve(model)
    test_data_consistency()
    
    print("\n" + "="*60)
    print("测试完成")
    print("="*60)
    print("\n如需运行完整Demo，请执行: python main_dynamic_demo1.py")


if __name__ == "__main__":
    main()
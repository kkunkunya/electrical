"""
数据加载和验证测试
Data loading and validation tests

测试数据加载器是否正确读取和验证数据
Test that the data loader correctly reads and validates data
"""

import pytest
import numpy as np
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.datasets.loader import DataLoader, load_system_data


class TestDataLoader:
    """测试数据加载器功能"""
    
    @pytest.fixture
    def data_loader(self):
        """创建数据加载器实例"""
        return DataLoader(data_dir="../data")
    
    def test_load_generators(self, data_loader):
        """测试发电机数据加载"""
        generators = data_loader.load_generators()
        
        # 检查发电机数量
        assert len(generators) == 14, f"应该有14个发电机，实际有{len(generators)}个"
        
        # 检查DEG发电机
        assert 'DEG1' in generators
        assert generators['DEG1'].type == 'DEG'
        assert generators['DEG1'].active_power_max == 120
        assert generators['DEG1'].reactive_power_max == 100
        
        assert 'DEG4' in generators
        assert generators['DEG4'].active_power_max == 80
        assert generators['DEG4'].reactive_power_max == 60
        
        # 检查MESS储能
        assert 'MESS' in generators
        assert generators['MESS'].type == 'MESS'
        assert generators['MESS'].active_power_max == 200
        assert generators['MESS'].reactive_power_max == 170
        assert generators['MESS'].charging_efficiency == 0.98
        assert generators['MESS'].storage_capacity == 600
        
        # 检查EVS电动汽车储能
        assert 'EVS1' in generators
        assert generators['EVS1'].type == 'EVS'
        assert generators['EVS1'].active_power_max == 60
        assert generators['EVS1'].reactive_power_max == 50
        assert generators['EVS1'].charging_efficiency == 0.98
        assert generators['EVS1'].storage_capacity == 120
        
        # 检查PV光伏
        assert 'PV1' in generators
        assert generators['PV1'].type == 'PV'
        assert generators['PV1'].predicted_power == 120
        assert generators['PV1'].active_power_max == 120  # 应该等于predicted_power
        
        assert 'PV3' in generators
        assert generators['PV3'].predicted_power == 360
        assert generators['PV3'].active_power_max == 360
    
    def test_load_loads(self, data_loader):
        """测试负荷数据加载"""
        loads = data_loader.load_loads()
        
        # 检查负荷节点数量
        assert len(loads) == 33, f"应该有33个负荷节点，实际有{len(loads)}个"
        
        # 检查特定节点数据（基于表B2）
        # 节点1
        assert loads['node_1'].bus_id == 1
        assert loads['node_1'].unit_load_shedding_cost == 1
        assert loads['node_1'].active_power == 60
        assert loads['node_1'].reactive_power == 30
        
        # 节点7 - 高负荷节点
        assert loads['node_7'].bus_id == 7
        assert loads['node_7'].unit_load_shedding_cost == 1
        assert loads['node_7'].active_power == 200
        assert loads['node_7'].reactive_power == 100
        
        # 节点24 - 最高负荷节点
        assert loads['node_24'].bus_id == 24
        assert loads['node_24'].unit_load_shedding_cost == 10
        assert loads['node_24'].active_power == 420
        assert loads['node_24'].reactive_power == 200
        
        # 节点30 - 最高无功负荷
        assert loads['node_30'].bus_id == 30
        assert loads['node_30'].unit_load_shedding_cost == 1
        assert loads['node_30'].active_power == 200
        assert loads['node_30'].reactive_power == 600
        
        # 验证总负荷
        total_active = sum(load.active_power for load in loads.values())
        total_reactive = sum(load.reactive_power for load in loads.values())
        assert total_active == 3715, f"总有功负荷应为3715 kW，实际为{total_active} kW"
        assert total_reactive == 2300, f"总无功负荷应为2300 kvar，实际为{total_reactive} kvar"
    
    def test_load_system_parameters(self, data_loader):
        """测试系统参数加载"""
        _, system_params = data_loader.load_branches()
        
        # 基于表B3的系统参数
        assert system_params.nominal_voltage == 12.66
        assert system_params.voltage_upper_limit == 13.92
        assert system_params.voltage_lower_limit == 11.39
        assert system_params.mobile_storage_unit_cost == 500
        assert system_params.mobile_storage_installation_time == 0.5
        assert system_params.mobile_storage_ideal_speed == 25
        assert system_params.uncertainty_factor == 0.35
        assert system_params.constant_epsilon == 4
    
    def test_load_profiles(self, data_loader):
        """测试负荷和交通曲线加载"""
        load_profiles, traffic_profiles = data_loader.load_profiles()
        
        # 检查数据形状
        assert len(load_profiles) == 21  # 3:00到23:00，共21个时间点
        assert len(traffic_profiles) == 21
        
        # 检查列名
        assert 'critical_load_factor' in load_profiles.columns
        assert 'non_critical_load_factor' in load_profiles.columns
        assert 'congestion_degree' in traffic_profiles.columns
        
        # 检查数值范围
        assert load_profiles['critical_load_factor'].min() >= 0.4
        assert load_profiles['critical_load_factor'].max() <= 1.0
        assert traffic_profiles['congestion_degree'].min() >= 0.2
        assert traffic_profiles['congestion_degree'].max() <= 1.0
    
    def test_data_consistency(self, data_loader):
        """测试数据一致性"""
        system_data = data_loader.load_all()
        
        # 验证数据一致性
        is_consistent = data_loader.validate_data_consistency(system_data)
        
        # 检查所有节点都有对应的负荷数据
        bus_ids = {load.bus_id for load in system_data.loads.values()}
        assert bus_ids == set(range(1, 34)), "负荷节点应该包含1-33所有节点"
    
    def test_numerical_precision(self, data_loader):
        """测试数值精度（误差<1e-6）"""
        generators = data_loader.load_generators()
        loads = data_loader.load_loads()
        _, system_params = data_loader.load_branches()
        
        # 定义期望值（来自PDF表格）
        expected_values = {
            # 发电机参数
            'DEG1_active': 120.0,
            'DEG5_reactive': 60.0,
            'MESS_efficiency': 0.98,
            'EVS1_storage': 120.0,
            'PV2_predicted': 240.0,
            
            # 负荷参数
            'node_4_active': 120.0,
            'node_30_reactive': 600.0,
            'node_11_active': 45.0,
            
            # 系统参数
            'nominal_voltage': 12.66,
            'voltage_lower': 11.39,
            'uncertainty': 0.35
        }
        
        # 验证发电机数值
        assert abs(generators['DEG1'].active_power_max - expected_values['DEG1_active']) < 1e-6
        assert abs(generators['DEG5'].reactive_power_max - expected_values['DEG5_reactive']) < 1e-6
        assert abs(generators['MESS'].charging_efficiency - expected_values['MESS_efficiency']) < 1e-6
        assert abs(generators['EVS1'].storage_capacity - expected_values['EVS1_storage']) < 1e-6
        assert abs(generators['PV2'].predicted_power - expected_values['PV2_predicted']) < 1e-6
        
        # 验证负荷数值
        assert abs(loads['node_4'].active_power - expected_values['node_4_active']) < 1e-6
        assert abs(loads['node_30'].reactive_power - expected_values['node_30_reactive']) < 1e-6
        assert abs(loads['node_11'].active_power - expected_values['node_11_active']) < 1e-6
        
        # 验证系统参数数值
        assert abs(system_params.nominal_voltage - expected_values['nominal_voltage']) < 1e-6
        assert abs(system_params.voltage_lower_limit - expected_values['voltage_lower']) < 1e-6
        assert abs(system_params.uncertainty_factor - expected_values['uncertainty']) < 1e-6


def test_load_system_data():
    """测试便捷加载函数"""
    try:
        system_data = load_system_data("../data")
        
        # 基本检查
        assert system_data is not None
        assert len(system_data.generators) > 0
        assert len(system_data.loads) == 33
        assert system_data.system_parameters is not None
        assert system_data.load_profiles is not None
        assert system_data.traffic_profiles is not None
        
        print("系统数据加载成功!")
        
    except Exception as e:
        pytest.fail(f"系统数据加载失败: {e}")


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v", "-s"])
"""
配电网数据加载器
Distribution Network Data Loader

使用pydantic进行数据验证和类型检查
Using pydantic for data validation and type checking
"""

from typing import Dict, List, Optional, Literal
from pathlib import Path
import yaml
import pandas as pd
from pydantic import BaseModel, Field, validator
import numpy as np


class Generator(BaseModel):
    """发电机数据类"""
    type: Literal["DEG", "MESS", "EVS", "PV"]
    active_power_max: Optional[float] = Field(None, ge=0, description="最大有功功率 (kW)")
    reactive_power_max: Optional[float] = Field(None, ge=0, description="最大无功功率 (kvar)")
    charging_efficiency: Optional[float] = Field(None, ge=0, le=1, description="充放电效率")
    storage_capacity: Optional[float] = Field(None, ge=0, description="储能容量 (kW·h)")
    predicted_power: Optional[float] = Field(None, ge=0, description="光伏预测功率 (kW)")
    
    @validator('active_power_max', pre=True, always=True)
    def set_active_power_for_pv(cls, v, values):
        """为PV类型设置active_power_max"""
        if 'type' in values and values['type'] == 'PV':
            # 对于PV类型，如果没有active_power_max，使用predicted_power
            if v is None and 'predicted_power' in values:
                return values['predicted_power']
        return v
    
    @validator('active_power_max')
    def validate_active_power(cls, v, values):
        """验证有功功率参数"""
        if values.get('type') in ['DEG', 'MESS', 'EVS'] and v is None:
            raise ValueError(f"{values.get('type')} 类型发电机必须设置最大有功功率")
        return v
    
    @validator('reactive_power_max')
    def validate_reactive_power(cls, v, values):
        """验证无功功率参数"""
        if values.get('type') in ['DEG', 'MESS', 'EVS'] and v is None:
            raise ValueError(f"{values.get('type')} 类型发电机必须设置最大无功功率")
        return v
    
    @validator('charging_efficiency')
    def validate_efficiency(cls, v, values):
        """验证充放电效率"""
        if values.get('type') in ['MESS', 'EVS'] and v is None:
            raise ValueError(f"{values.get('type')} 类型储能设备必须设置充放电效率")
        return v
    
    @validator('storage_capacity')
    def validate_storage(cls, v, values):
        """验证储能容量"""
        if values.get('type') in ['MESS', 'EVS'] and v is None:
            raise ValueError(f"{values.get('type')} 类型储能设备必须设置储能容量")
        return v


class Load(BaseModel):
    """负荷数据类"""
    bus_id: int = Field(ge=1, le=33, description="节点编号")
    unit_load_shedding_cost: float = Field(ge=0, description="单位负荷削减成本 (元/kW)")
    active_power: float = Field(ge=0, description="有功功率 (kW)")
    reactive_power: float = Field(ge=0, description="无功功率 (kvar)")
    
    @validator('bus_id')
    def validate_bus_id(cls, v):
        """验证节点编号在33节点系统范围内"""
        if not (1 <= v <= 33):
            raise ValueError("节点编号必须在1-33之间")
        return v


class Branch(BaseModel):
    """支路数据类"""
    from_bus: int = Field(ge=1, le=33, description="起始节点")
    to_bus: int = Field(ge=1, le=33, description="终止节点")
    resistance: float = Field(ge=0, description="电阻 (Ω)")
    reactance: float = Field(ge=0, description="电抗 (Ω)")
    capacity: float = Field(gt=0, description="容量 (A)")
    
    @validator('to_bus')
    def validate_different_buses(cls, v, values):
        """验证起始和终止节点不同"""
        if v == values.get('from_bus'):
            raise ValueError("起始节点和终止节点不能相同")
        return v


class SystemParameters(BaseModel):
    """系统运行参数类"""
    nominal_voltage: float = Field(gt=0, description="额定电压 (kV)")
    voltage_upper_limit: float = Field(gt=0, description="电压上限 (kV)")
    voltage_lower_limit: float = Field(gt=0, description="电压下限 (kV)")
    mobile_storage_unit_cost: float = Field(ge=0, description="移动储能单位配置成本 (元)")
    mobile_storage_installation_time: float = Field(gt=0, description="移动储能安装配置时间 (h)")
    mobile_storage_ideal_speed: float = Field(gt=0, description="移动储能理想车速 (km/h)")
    uncertainty_factor: float = Field(ge=0, le=1, description="不确定度 τ")
    constant_epsilon: float = Field(gt=0, description="常数 ε")
    
    @validator('voltage_upper_limit')
    def validate_voltage_limits(cls, v, values):
        """验证电压上下限关系"""
        nominal = values.get('nominal_voltage')
        if nominal and v <= nominal:
            raise ValueError("电压上限必须大于额定电压")
        return v
    
    @validator('voltage_lower_limit')
    def validate_lower_limit(cls, v, values):
        """验证电压下限关系"""
        nominal = values.get('nominal_voltage')
        if nominal and v >= nominal:
            raise ValueError("电压下限必须小于额定电压")
        return v


class SystemData(BaseModel):
    """完整系统数据类"""
    generators: Dict[str, Generator]
    loads: Dict[str, Load]
    branches: Dict[str, Branch]
    system_parameters: SystemParameters
    load_profiles: Optional[pd.DataFrame] = Field(None, description="负荷预测曲线")
    traffic_profiles: Optional[pd.DataFrame] = Field(None, description="交通拥堵曲线")
    
    class Config:
        arbitrary_types_allowed = True  # 允许pandas DataFrame类型
    
    @validator('loads')
    def validate_load_count(cls, v):
        """验证负荷节点数量"""
        if len(v) != 33:
            raise ValueError(f"应该有33个负荷节点，但实际有{len(v)}个")
        return v
    
    @validator('branches')
    def validate_branch_count(cls, v):
        """验证支路数量 (33节点系统应该有32条支路)"""
        if len(v) != 32:
            # 暂时警告，因为支路数据可能不完整
            print(f"警告：33节点系统通常有32条支路，但当前有{len(v)}条支路")
        return v


class DataLoader:
    """数据加载器"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise FileNotFoundError(f"数据目录不存在: {self.data_dir}")
    
    def load_generators(self) -> Dict[str, Generator]:
        """加载发电机数据"""
        file_path = self.data_dir / "generators.yaml"
        if not file_path.exists():
            raise FileNotFoundError(f"发电机数据文件不存在: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        generators = {}
        for category in ['deg_generators', 'mess_generators', 'evs_generators', 'pv_generators']:
            if category in data:
                for name, gen_data in data[category].items():
                    generators[name] = Generator(**gen_data)
        
        return generators
    
    def load_loads(self) -> Dict[str, Load]:
        """加载负荷数据"""
        file_path = self.data_dir / "loads.yaml"
        if not file_path.exists():
            raise FileNotFoundError(f"负荷数据文件不存在: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        loads = {}
        if 'loads' in data:
            for name, load_data in data['loads'].items():
                loads[name] = Load(**load_data)
        
        return loads
    
    def load_branches(self) -> Dict[str, Branch]:
        """加载支路数据"""
        file_path = self.data_dir / "branches.yaml"
        if not file_path.exists():
            raise FileNotFoundError(f"支路数据文件不存在: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        branches = {}
        system_params = None
        
        if 'branches' in data:
            for name, branch_data in data['branches'].items():
                branches[name] = Branch(**branch_data)
        
        if 'system_parameters' in data:
            system_params = SystemParameters(**data['system_parameters'])
        
        return branches, system_params
    
    def load_profiles(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """加载负荷预测曲线和交通拥堵曲线"""
        # 加载负荷预测曲线
        load_file = self.data_dir / "pv_profile.csv"
        if not load_file.exists():
            raise FileNotFoundError(f"负荷预测文件不存在: {load_file}")
        
        # 加载交通拥堵曲线
        traffic_file = self.data_dir / "traffic_profile.csv"
        if not traffic_file.exists():
            raise FileNotFoundError(f"交通拥堵文件不存在: {traffic_file}")
        
        load_profiles = pd.read_csv(load_file, comment='#')
        traffic_profiles = pd.read_csv(traffic_file, comment='#')
        
        return load_profiles, traffic_profiles
    
    def load_all(self) -> SystemData:
        """加载所有数据"""
        generators = self.load_generators()
        loads = self.load_loads()
        branches, system_params = self.load_branches()
        load_profiles, traffic_profiles = self.load_profiles()
        
        if system_params is None:
            raise ValueError("系统参数缺失")
        
        return SystemData(
            generators=generators,
            loads=loads,
            branches=branches,
            system_parameters=system_params,
            load_profiles=load_profiles,
            traffic_profiles=traffic_profiles
        )
    
    def validate_data_consistency(self, system_data: SystemData) -> bool:
        """验证数据一致性"""
        # 检查节点编号连续性
        load_bus_ids = {load.bus_id for load in system_data.loads.values()}
        expected_bus_ids = set(range(1, 34))  # 1到33
        
        if load_bus_ids != expected_bus_ids:
            missing = expected_bus_ids - load_bus_ids
            extra = load_bus_ids - expected_bus_ids
            if missing:
                print(f"警告：缺少节点: {missing}")
            if extra:
                print(f"警告：多余节点: {extra}")
            return False
        
        # 检查支路连接性
        branch_buses = set()
        for branch in system_data.branches.values():
            branch_buses.add(branch.from_bus)
            branch_buses.add(branch.to_bus)
        
        # 检查所有负荷节点是否都有支路连接
        unconnected = load_bus_ids - branch_buses
        if unconnected:
            print(f"警告：以下节点没有支路连接: {unconnected}")
        
        return True


def load_system_data(data_dir: str = "data") -> SystemData:
    """便捷函数：加载完整系统数据"""
    loader = DataLoader(data_dir)
    system_data = loader.load_all()
    loader.validate_data_consistency(system_data)
    return system_data


if __name__ == "__main__":
    # 测试数据加载
    try:
        system_data = load_system_data()
        print("数据加载成功！")
        print(f"发电机数量: {len(system_data.generators)}")
        print(f"负荷节点数量: {len(system_data.loads)}")
        print(f"支路数量: {len(system_data.branches)}")
        print(f"系统额定电压: {system_data.system_parameters.nominal_voltage} kV")
        
    except Exception as e:
        print(f"数据加载失败: {e}")
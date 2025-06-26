"""
IEEE 33节点标准配电网数据
IEEE 33-bus standard distribution network data
"""

import yaml
from pathlib import Path


def get_ieee33_branch_data():
    """
    获取IEEE 33节点系统标准支路数据
    数据来源：IEEE标准测试系统
    """
    # IEEE 33节点系统支路数据
    # 格式：(from_bus, to_bus, resistance(Ω), reactance(Ω), capacity(A))
    branches_data = [
        (1, 2, 0.0922, 0.0470, 400),
        (2, 3, 0.4930, 0.2511, 400),
        (3, 4, 0.3660, 0.1864, 400),
        (4, 5, 0.3811, 0.1941, 400),
        (5, 6, 0.8190, 0.7070, 400),
        (6, 7, 0.1872, 0.6188, 400),
        (7, 8, 0.7114, 0.2351, 400),
        (8, 9, 1.0300, 0.7400, 400),
        (9, 10, 1.0440, 0.7400, 400),
        (10, 11, 0.1966, 0.0650, 400),
        (11, 12, 0.3744, 0.1238, 400),
        (12, 13, 1.4680, 1.1550, 400),
        (13, 14, 0.5416, 0.7129, 400),
        (14, 15, 0.5910, 0.5260, 400),
        (15, 16, 0.7463, 0.5450, 400),
        (16, 17, 1.2890, 1.7210, 400),
        (17, 18, 0.3200, 0.5740, 400),
        (2, 19, 0.1640, 0.1565, 400),
        (19, 20, 1.5042, 1.3554, 400),
        (20, 21, 0.4095, 0.4784, 400),
        (21, 22, 0.7089, 0.9373, 400),
        (3, 23, 0.4512, 0.3083, 400),
        (23, 24, 0.8980, 0.7091, 400),
        (24, 25, 0.8960, 0.7011, 400),
        (6, 26, 0.2030, 0.1034, 400),
        (26, 27, 0.2842, 0.1447, 400),
        (27, 28, 1.0590, 0.9337, 400),
        (28, 29, 0.8042, 0.7006, 400),
        (29, 30, 0.5075, 0.2585, 400),
        (30, 31, 0.9744, 0.9630, 400),
        (31, 32, 0.3105, 0.3619, 400),
        (32, 33, 0.3410, 0.5302, 400),
    ]
    
    return branches_data


def update_branches_yaml(output_path: str = "data/branches.yaml"):
    """
    更新branches.yaml文件为完整的IEEE 33节点数据
    
    Args:
        output_path: 输出文件路径
    """
    branches_data = get_ieee33_branch_data()
    
    # 构建YAML数据结构
    yaml_data = {
        'branches': {},
        'system_parameters': {
            'nominal_voltage': 12.66,
            'base_voltage': 12.66,
            'voltage_upper_limit': 13.92,
            'voltage_lower_limit': 11.39,
            'min_voltage': 11.39,
            'max_voltage': 13.92,
            'mobile_storage_unit_cost': 500,
            'mobile_storage_installation_time': 0.5,
            'mobile_storage_ideal_speed': 25,
            'uncertainty_factor': 0.35,
            'constant_epsilon': 4
        }
    }
    
    # 添加所有支路数据
    for i, (from_bus, to_bus, resistance, reactance, capacity) in enumerate(branches_data):
        branch_name = f"branch_{from_bus}_{to_bus}"
        yaml_data['branches'][branch_name] = {
            'from_bus': from_bus,
            'to_bus': to_bus,
            'resistance': resistance,
            'reactance': reactance,
            'capacity': capacity,
            'max_current': capacity  # 添加max_current字段
        }
    
    # 写入文件
    output_path = Path(output_path)
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(yaml_data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    
    print(f"已更新branches.yaml文件，包含{len(branches_data)}条支路数据")
    return len(branches_data)


if __name__ == "__main__":
    # 更新branches.yaml文件
    update_branches_yaml()
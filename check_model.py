#!/usr/bin/env python3
"""
检查已保存模型的内容
"""

import torch
import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

def check_model_file(model_path: str):
    """检查模型文件内容"""
    print(f"检查模型文件: {model_path}")
    
    try:
        # 加载模型文件
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        
        print("✅ 模型文件加载成功")
        print(f"检查点键: {list(checkpoint.keys())}")
        
        for key, value in checkpoint.items():
            print(f"\n🔍 {key}:")
            if key == 'model_state_dict':
                print(f"  - 类型: {type(value)}")
                print(f"  - 参数数量: {len(value) if isinstance(value, dict) else 'N/A'}")
                if isinstance(value, dict):
                    print(f"  - 参数键示例: {list(value.keys())[:5]}")
            elif key == 'model_config':
                print(f"  - 类型: {type(value)}")
                print(f"  - 内容: {value}")
            else:
                print(f"  - 类型: {type(value)}")
                print(f"  - 值: {value}")
                
        return True
        
    except Exception as e:
        print(f"❌ 检查失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    model_path = "/mnt/c/Users/sxk27/OneDrive - MSFT/Project/electrical/output/demo4_g2milp/training/g2milp_training_20250706_220830/final_model.pth"
    
    success = check_model_file(model_path)
    
    if success:
        print("\n✅ 模型文件检查完成")
    else:
        print("\n❌ 模型文件检查失败")

if __name__ == "__main__":
    main()
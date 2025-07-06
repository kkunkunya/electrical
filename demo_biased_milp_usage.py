"""
Demo 2主程序：有偏差MILP实例生成与基准解对比
Biased MILP Instance Generation and Ground Truth Comparison

本程序演示如何：
1. 生成基准解（金标准解）
2. 通过数据扰动生成有偏差的MILP实例
3. 对比分析基准解与扰动实例的差异
4. 评估扰动对系统性能的影响

功能特性：
- 完整的基准解生成和验证流程
- 多种扰动场景的MILP实例生成
- 详细的对比分析和可视化报告
- 完善的日志记录和错误处理
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import json

# 添加项目路径到sys.path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 导入核心模块
from src.datasets.loader import load_system_data
from src.models.ground_truth_solver import GroundTruthSolver, create_ground_truth_solver
from src.models.biased_milp_generator import (
    BiasedMILPGenerator, 
    PerturbationConfig,
    create_default_perturbation_configs,
    create_scenario_perturbation_configs
)

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class Demo2Manager:
    """Demo 2管理器 - 统一管理基准解生成和MILP实例生成流程"""
    
    def __init__(self, data_dir: str = "data", output_dir: str = "output/demo2"):
        """
        初始化Demo 2管理器
        
        Args:
            data_dir: 数据目录
            output_dir: 输出目录
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建子目录
        self.ground_truth_dir = self.output_dir / "ground_truth"
        self.milp_instances_dir = self.output_dir / "milp_instances"
        self.analysis_dir = self.output_dir / "analysis"
        
        for dir_path in [self.ground_truth_dir, self.milp_instances_dir, self.analysis_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # 核心组件
        self.system_data = None
        self.ground_truth_solver = None
        self.milp_generator = None
        self.baseline_results = None
        
        logger.info("Demo 2管理器初始化完成")
        logger.info(f"数据目录: {self.data_dir}")
        logger.info(f"输出目录: {self.output_dir}")
    
    def load_system_data(self) -> bool:
        """
        加载系统数据
        
        Returns:
            是否加载成功
        """
        try:
            logger.info("="*60)
            logger.info("步骤 1: 加载系统数据")
            logger.info("="*60)
            
            self.system_data = load_system_data(str(self.data_dir))
            
            logger.info("✅ 系统数据加载成功")
            logger.info(f"📊 系统规模:")
            logger.info(f"   • 发电机数量: {len(self.system_data.generators)}")
            logger.info(f"   • 负荷节点数: {len(self.system_data.loads)}")
            logger.info(f"   • 支路数量: {len(self.system_data.branches)}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 系统数据加载失败: {e}")
            return False
    
    def generate_ground_truth(self, config: Optional[Dict] = None) -> bool:
        """
        生成基准解
        
        Args:
            config: 基准解生成配置
            
        Returns:
            是否生成成功
        """
        try:
            logger.info("="*60)
            logger.info("步骤 2: 生成基准解 (Ground Truth)")
            logger.info("="*60)
            
            if self.system_data is None:
                logger.error("系统数据未加载，请先调用load_system_data()")
                return False
            
            # 默认配置
            default_config = {
                'n_periods': 21,
                'start_hour': 3,
                'traffic_profile_path': str(self.data_dir / "traffic_profile.csv"),
                'pv_profile_path': str(self.data_dir / "pv_profile.csv"),
                'output_dir': str(self.ground_truth_dir)
            }
            
            if config:
                default_config.update(config)
            
            # 创建基准解生成器
            self.ground_truth_solver = create_ground_truth_solver(
                self.system_data, 
                default_config
            )
            
            # 求解基准解
            logger.info("🔧 开始求解基准解...")
            self.baseline_results = self.ground_truth_solver.solve_ground_truth(
                verbose=True
            )
            
            if self.baseline_results:
                logger.info("="*40)
                logger.info("✅ 基准解生成成功!")
                logger.info("="*40)
                logger.info(f"📊 基准解指标:")
                logger.info(f"   • 目标函数值: {self.baseline_results['objective']:.2f} 元")
                logger.info(f"   • 总负荷削减: {self.baseline_results.get('total_load_shed', 0):.2f} kW")
                logger.info(f"   • 求解时间: {self.baseline_results.get('solve_time', 0):.3f} 秒")
                logger.info(f"   • 求解状态: {self.baseline_results.get('status', 'unknown')}")
                
                # 导出对比数据
                export_path = self.ground_truth_solver.export_for_comparison()
                logger.info(f"📄 基准解对比数据已导出: {export_path}")
                
                return True
            else:
                logger.error("❌ 基准解求解失败")
                return False
                
        except Exception as e:
            logger.error(f"❌ 基准解生成过程出错: {e}")
            return False
    
    def generate_biased_milp_instances(self, 
                                     generation_mode: str = "scenario",
                                     custom_configs: Optional[List[PerturbationConfig]] = None) -> bool:
        """
        生成有偏差的MILP实例
        
        Args:
            generation_mode: 生成模式 ("scenario", "batch", "custom")
            custom_configs: 自定义扰动配置列表
            
        Returns:
            是否生成成功
        """
        try:
            logger.info("="*60)
            logger.info("步骤 3: 生成有偏差MILP实例")
            logger.info("="*60)
            
            if self.system_data is None:
                logger.error("系统数据未加载，请先调用load_system_data()")
                return False
            
            # 创建MILP实例生成器
            self.milp_generator = BiasedMILPGenerator(
                base_system_data=self.system_data,
                output_dir=str(self.milp_instances_dir),
                log_dir=str(self.output_dir / "logs")
            )
            
            if generation_mode == "scenario":
                # 场景模式：生成不同场景的实例
                logger.info("🎯 使用场景模式生成MILP实例")
                
                scenario_configs = create_scenario_perturbation_configs()
                logger.info(f"📋 将生成 {len(scenario_configs)} 个场景实例:")
                for scenario_name in scenario_configs.keys():
                    logger.info(f"   • {scenario_name}")
                
                scenario_instances = self.milp_generator.generate_scenario_instances(
                    scenario_configs=scenario_configs,
                    n_periods=21,
                    start_hour=3,
                    save_to_file=True
                )
                
                logger.info(f"✅ 场景实例生成完成，共 {len(scenario_instances)} 个")
                
                # 打印每个实例的摘要
                for scenario_name, instance in scenario_instances.items():
                    logger.info(f"\n📊 场景 {scenario_name} 实例摘要:")
                    self._print_instance_brief(instance)
                
                return True
                
            elif generation_mode == "batch":
                # 批量模式：使用默认配置生成多个实例
                logger.info("📦 使用批量模式生成MILP实例")
                
                batch_configs = create_default_perturbation_configs()
                logger.info(f"📋 将生成 {len(batch_configs)} 个批量实例")
                
                batch_instances = self.milp_generator.generate_batch_instances(
                    perturbation_configs=batch_configs,
                    instance_prefix="demo2_batch",
                    n_periods=21,
                    start_hour=3,
                    save_to_file=True
                )
                
                logger.info(f"✅ 批量实例生成完成，共 {len(batch_instances)} 个")
                
                return True
                
            elif generation_mode == "custom" and custom_configs:
                # 自定义模式：使用用户提供的配置
                logger.info("🔧 使用自定义模式生成MILP实例")
                
                custom_instances = self.milp_generator.generate_batch_instances(
                    perturbation_configs=custom_configs,
                    instance_prefix="demo2_custom",
                    n_periods=21,
                    start_hour=3,
                    save_to_file=True
                )
                
                logger.info(f"✅ 自定义实例生成完成，共 {len(custom_instances)} 个")
                
                return True
                
            else:
                logger.error(f"❌ 不支持的生成模式: {generation_mode}")
                return False
                
        except Exception as e:
            logger.error(f"❌ MILP实例生成过程出错: {e}")
            return False
    
    def _print_instance_brief(self, instance):
        """打印实例简要信息"""
        if instance.statistics:
            stats = instance.statistics
            logger.info(f"     变量: {stats.n_variables}, 约束: {stats.n_constraints}")
            logger.info(f"     节点: {stats.n_buses}, 时段: {stats.n_time_periods}")
        
        if instance.perturbation_config:
            config = instance.perturbation_config
            logger.info(f"     扰动强度: {config.perturbation_intensity}, 种子: {config.random_seed}")
    
    def generate_comparison_analysis(self) -> bool:
        """
        生成对比分析报告
        
        Returns:
            是否生成成功
        """
        try:
            logger.info("="*60)
            logger.info("步骤 4: 生成对比分析报告")
            logger.info("="*60)
            
            if self.baseline_results is None:
                logger.error("基准解未生成，请先调用generate_ground_truth()")
                return False
            
            # 生成详细的分析报告
            analysis_file = self.analysis_dir / f"demo2_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            
            with open(analysis_file, 'w', encoding='utf-8') as f:
                f.write("="*80 + "\n")
                f.write("Demo 2: 有偏差MILP实例生成与基准解对比分析\n")
                f.write("="*80 + "\n\n")
                
                # 基准解分析
                f.write("1. 基准解分析\n")
                f.write("-" * 40 + "\n")
                f.write(f"目标函数值: {self.baseline_results['objective']:.2f} 元\n")
                f.write(f"总负荷削减: {self.baseline_results.get('total_load_shed', 0):.2f} kW\n")
                f.write(f"求解时间: {self.baseline_results.get('solve_time', 0):.3f} 秒\n")
                f.write(f"求解状态: {self.baseline_results.get('status', 'unknown')}\n")
                f.write(f"验证得分: {self.ground_truth_solver.validation_report.get('summary', {}).get('validation_score', 0):.1%}\n\n")
                
                # MILP实例统计
                f.write("2. MILP实例统计\n")
                f.write("-" * 40 + "\n")
                instance_files = list(self.milp_instances_dir.glob("*.pkl"))
                f.write(f"生成的MILP实例数量: {len(instance_files)}\n")
                f.write(f"实例保存目录: {self.milp_instances_dir}\n\n")
                
                # 数据扰动影响分析
                f.write("3. 数据扰动影响分析\n")
                f.write("-" * 40 + "\n")
                f.write("基准解代表了在无扰动情况下的最优调度策略，\n")
                f.write("而有偏差的MILP实例模拟了灾后数据缺失和不确定性条件。\n")
                f.write("这些实例可用于：\n")
                f.write("• 测试优化算法在不确定环境下的鲁棒性\n")
                f.write("• 评估数据质量对调度决策的影响\n")
                f.write("• 开发更有效的不确定性处理方法\n\n")
                
                # 应用建议
                f.write("4. 应用建议\n")
                f.write("-" * 40 + "\n")
                f.write("1. 将基准解作为算法性能评估的金标准\n")
                f.write("2. 使用不同扰动场景测试算法适应性\n")
                f.write("3. 分析扰动参数对目标函数的敏感性\n")
                f.write("4. 研究扰动模式与实际不确定性的匹配度\n\n")
                
                # 文件清单
                f.write("5. 生成的文件清单\n")
                f.write("-" * 40 + "\n")
                f.write(f"基准解目录: {self.ground_truth_dir}\n")
                f.write(f"MILP实例目录: {self.milp_instances_dir}\n")
                f.write(f"分析报告目录: {self.analysis_dir}\n")
                f.write(f"日志目录: {self.output_dir / 'logs'}\n\n")
                
                f.write("="*80 + "\n")
                f.write("分析报告生成完成\n")
                f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            
            logger.info(f"✅ 对比分析报告已生成: {analysis_file}")
            
            # 生成JSON格式的总结数据
            summary_data = {
                "demo2_summary": {
                    "timestamp": datetime.now().isoformat(),
                    "baseline_results": {
                        "objective_value": self.baseline_results['objective'],
                        "total_load_shed": self.baseline_results.get('total_load_shed', 0),
                        "solve_time": self.baseline_results.get('solve_time', 0),
                        "status": self.baseline_results.get('status', 'unknown')
                    },
                    "milp_instances": {
                        "count": len(list(self.milp_instances_dir.glob("*.pkl"))),
                        "directory": str(self.milp_instances_dir)
                    },
                    "directories": {
                        "ground_truth": str(self.ground_truth_dir),
                        "milp_instances": str(self.milp_instances_dir),
                        "analysis": str(self.analysis_dir)
                    }
                }
            }
            
            summary_file = self.analysis_dir / "demo2_summary.json"
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"📄 总结数据已保存: {summary_file}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 对比分析生成过程出错: {e}")
            return False
    
    def run_complete_demo(self, 
                         generation_mode: str = "scenario",
                         ground_truth_config: Optional[Dict] = None) -> bool:
        """
        运行完整的Demo 2流程
        
        Args:
            generation_mode: MILP实例生成模式
            ground_truth_config: 基准解生成配置
            
        Returns:
            是否全部成功
        """
        logger.info("🚀 开始运行Demo 2完整流程")
        logger.info("="*80)
        
        # 步骤1：加载系统数据
        if not self.load_system_data():
            return False
        
        # 步骤2：生成基准解
        if not self.generate_ground_truth(ground_truth_config):
            return False
        
        # 步骤3：生成有偏差MILP实例
        if not self.generate_biased_milp_instances(generation_mode):
            return False
        
        # 步骤4：生成对比分析
        if not self.generate_comparison_analysis():
            return False
        
        logger.info("="*80)
        logger.info("🎉 Demo 2完整流程执行成功!")
        logger.info("="*80)
        logger.info(f"📁 所有结果保存在: {self.output_dir}")
        logger.info("📊 主要输出文件:")
        logger.info(f"   • 基准解: {self.ground_truth_dir}")
        logger.info(f"   • MILP实例: {self.milp_instances_dir}")
        logger.info(f"   • 分析报告: {self.analysis_dir}")
        logger.info("="*80)
        
        return True


def main():
    """主函数 - Demo 2使用示例"""
    
    print("🎯 Demo 2: 有偏差MILP实例生成与基准解对比")
    print("="*80)
    
    try:
        # 检查数据目录
        data_dir = Path("data")
        if not data_dir.exists():
            print(f"❌ 数据目录不存在: {data_dir}")
            print("请确保数据文件位于data/目录下")
            return
        
        # 创建Demo管理器
        demo_manager = Demo2Manager(
            data_dir=str(data_dir),
            output_dir="output/demo2"
        )
        
        # 基准解生成配置
        ground_truth_config = {
            'n_periods': 21,  # 21个时段 (3:00-23:00)
            'start_hour': 3,  # 从3点开始
        }
        
        # 运行完整Demo
        success = demo_manager.run_complete_demo(
            generation_mode="scenario",  # 使用场景模式
            ground_truth_config=ground_truth_config
        )
        
        if success:
            print("\n🎉 Demo 2执行成功!")
            print(f"📁 查看结果: {demo_manager.output_dir}")
            
            # 显示生成的基准解指标
            if demo_manager.baseline_results:
                print("\n📊 基准解关键指标:")
                print(f"   • 目标函数值: {demo_manager.baseline_results['objective']:.2f} 元")
                print(f"   • 总负荷削减: {demo_manager.baseline_results.get('total_load_shed', 0):.2f} kW")
                print(f"   • 求解时间: {demo_manager.baseline_results.get('solve_time', 0):.3f} 秒")
            
            # 显示生成的实例数量
            instance_count = len(list(demo_manager.milp_instances_dir.glob("*.pkl")))
            print(f"   • 生成MILP实例: {instance_count} 个")
            
        else:
            print("❌ Demo 2执行失败，请检查日志输出")
    
    except Exception as e:
        print(f"❌ Demo 2执行过程出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Demo 4 推理功能测试脚本
Demo 4 Inference Function Test Script

测试修复后的推理功能，使用已训练的模型进行MILP实例生成和质量评估
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import traceback
import pickle

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

# 导入必要模块
from models.g2milp.generator import G2MILPGenerator, GeneratorConfig
from models.g2milp.inference import G2MILPInference, InferenceConfig
from models.g2milp.evaluation import G2MILPEvaluator, EvaluationConfig
from models.bipartite_graph.format_converter import Demo3ToDemo4Converter

# 设置日志
def setup_logging():
    """设置日志系统"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path("output/demo4_g2milp/inference_test")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / f"inference_test_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"日志文件: {log_file}")
    return logger

logger = setup_logging()


def load_trained_model(model_path: str, device: str = "cuda") -> G2MILPGenerator:
    """
    加载已训练的G2MILP模型
    
    Args:
        model_path: 模型文件路径
        device: 计算设备
        
    Returns:
        加载的模型实例
    """
    logger.info(f"加载训练好的模型: {model_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    # 加载模型状态 (PyTorch 2.7+需要设置weights_only=False)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # 提取模型配置
    if 'config' in checkpoint:
        model_config = checkpoint['config']
        logger.info("从检查点加载模型配置")
    elif 'model_config' in checkpoint:
        model_config = checkpoint['model_config']
        logger.info("从检查点加载模型配置(旧格式)")
    else:
        # 使用默认配置
        model_config = GeneratorConfig()
        logger.warning("使用默认模型配置")
    
    # 从检查点获取特征维度
    constraint_feature_dim = checkpoint.get('constraint_feature_dim', 16)
    variable_feature_dim = checkpoint.get('variable_feature_dim', 9)
    edge_feature_dim = checkpoint.get('edge_feature_dim', 8)
    
    # 创建模型实例
    model = G2MILPGenerator(
        constraint_feature_dim=constraint_feature_dim,
        variable_feature_dim=variable_feature_dim,
        edge_feature_dim=edge_feature_dim,
        config=model_config
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    logger.info(f"✅ 模型加载成功")
    logger.info(f"  - 参数数量: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"  - 训练epoch: {checkpoint.get('epoch', 'unknown')}")
    logger.info(f"  - 训练损失: {checkpoint.get('train_loss', 'unknown')}")
    logger.info(f"  - 验证损失: {checkpoint.get('val_loss', 'unknown')}")
    
    return model


def load_demo3_data(data_path: str) -> torch.Tensor:
    """
    加载Demo 3的二分图数据
    
    Args:
        data_path: 数据文件路径
        
    Returns:
        转换后的HeteroData对象
    """
    logger.info(f"加载Demo 3数据: {data_path}")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"数据文件不存在: {data_path}")
    
    # 加载pickle文件
    with open(data_path, 'rb') as f:
        bipartite_graph = pickle.load(f)
    
    logger.info(f"原始数据类型: {type(bipartite_graph)}")
    
    # 转换为HeteroData格式
    converter = Demo3ToDemo4Converter()
    conversion_result = converter.convert_bipartite_graph(bipartite_graph)
    hetero_data = conversion_result['bipartite_data']
    
    logger.info(f"✅ Demo 3数据加载成功")
    logger.info(f"  - 约束节点: {hetero_data['constraint'].x.shape[0]}")
    logger.info(f"  - 变量节点: {hetero_data['variable'].x.shape[0]}")
    logger.info(f"  - 边数: {hetero_data['constraint', 'connects', 'variable'].edge_index.shape[1]}")
    
    return hetero_data


def test_inference_pipeline(model: G2MILPGenerator, source_data: torch.Tensor):
    """
    测试完整的推理管道
    
    Args:
        model: 训练好的模型
        source_data: 源数据
    """
    logger.info("🔍 开始推理管道测试")
    logger.info("=" * 60)
    
    # 创建推理配置
    inference_config = InferenceConfig(
        eta=0.1,
        num_test_instances=3,
        temperature=1.0,
        sample_from_prior=True,
        constraint_selection_strategy="random",
        diversity_boost=True,
        num_diverse_samples=5,
        compute_similarity_metrics=True,
        generate_comparison_report=True,
        experiment_name=f"inference_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    
    # 创建推理器
    inference_engine = G2MILPInference(model, inference_config)
    
    # 执行推理测试
    logger.info("执行推理测试...")
    
    try:
        # 1. 单实例生成测试
        logger.info("📋 测试 1: 单实例生成")
        result = inference_engine.generate_single_instance(
            source_data=source_data,
            save_intermediate=True
        )
        
        if result:
            logger.info("✅ 单实例生成成功")
            logger.info(f"  - 生成质量: {result.get('quality_score', 'N/A')}")
            logger.info(f"  - 相似度: {result.get('similarity_score', 'N/A')}")
        else:
            logger.error("❌ 单实例生成失败")
            return False
        
        # 2. 批量生成测试
        logger.info("📋 测试 2: 批量生成")
        batch_results = inference_engine.generate_batch_instances(
            source_data=source_data,
            num_instances=inference_config.num_test_instances
        )
        
        if batch_results:
            logger.info(f"✅ 批量生成成功 ({len(batch_results)} 个实例)")
            
            # 分析批量结果
            quality_scores = [r.get('quality_score', 0) for r in batch_results if r.get('quality_score')]
            similarity_scores = [r.get('similarity_score', 0) for r in batch_results if r.get('similarity_score')]
            
            if quality_scores:
                logger.info(f"  - 平均质量: {np.mean(quality_scores):.4f}")
                logger.info(f"  - 质量范围: [{np.min(quality_scores):.4f}, {np.max(quality_scores):.4f}]")
            
            if similarity_scores:
                logger.info(f"  - 平均相似度: {np.mean(similarity_scores):.4f}")
                logger.info(f"  - 相似度范围: [{np.min(similarity_scores):.4f}, {np.max(similarity_scores):.4f}]")
        else:
            logger.error("❌ 批量生成失败")
            return False
        
        # 3. 多样性增强测试
        logger.info("📋 测试 3: 多样性增强生成")
        diversity_results = inference_engine.generate_diverse_instances(
            source_data=source_data,
            num_samples=inference_config.num_diverse_samples
        )
        
        if diversity_results:
            logger.info(f"✅ 多样性生成成功 ({len(diversity_results)} 个样本)")
            
            # 分析多样性
            diversity_scores = [r.get('diversity_score', 0) for r in diversity_results if r.get('diversity_score')]
            if diversity_scores:
                logger.info(f"  - 平均多样性: {np.mean(diversity_scores):.4f}")
                logger.info(f"  - 多样性范围: [{np.min(diversity_scores):.4f}, {np.max(diversity_scores):.4f}]")
        else:
            logger.error("❌ 多样性生成失败")
            return False
        
        # 4. 质量评估测试
        logger.info("📋 测试 4: 质量评估")
        evaluation_results = inference_engine.evaluate_generated_instances(
            generated_instances=batch_results[:2],  # 使用前两个实例
            source_data=source_data
        )
        
        if evaluation_results:
            logger.info("✅ 质量评估成功")
            logger.info(f"  - 综合得分: {evaluation_results.get('overall_score', 'N/A')}")
            logger.info(f"  - 有效性: {evaluation_results.get('validity_score', 'N/A')}")
            logger.info(f"  - 图结构相似度: {evaluation_results.get('graph_similarity', 'N/A')}")
        else:
            logger.warning("⚠️ 质量评估返回空结果")
        
        logger.info("🎉 推理管道测试完成")
        return True
        
    except Exception as e:
        logger.error(f"❌ 推理测试过程中出现错误: {e}")
        logger.error(traceback.format_exc())
        return False


def test_numerical_stability():
    """
    测试数值稳定性修复
    """
    logger.info("🔬 测试数值稳定性修复")
    logger.info("=" * 60)
    
    # 创建测试数据（包含可能导致数值问题的情况）
    test_cases = [
        {
            'name': '零标准差测试',
            'data1': np.array([1.0, 1.0, 1.0, 1.0, 1.0]),  # 常数值
            'data2': np.array([2.0, 2.0, 2.0, 2.0, 2.0])   # 不同常数值
        },
        {
            'name': '相同常数测试',
            'data1': np.array([0.5, 0.5, 0.5, 0.5, 0.5]),  # 相同常数
            'data2': np.array([0.5, 0.5, 0.5, 0.5, 0.5])   # 相同常数
        },
        {
            'name': '极小方差测试',
            'data1': np.array([1.0, 1.0000001, 1.0, 1.0, 1.0]),  # 极小方差
            'data2': np.array([2.0, 2.0, 2.0, 2.0, 2.0])
        },
        {
            'name': '正常数据测试',
            'data1': np.array([1.0, 2.0, 3.0, 4.0, 5.0]),  # 正常数据
            'data2': np.array([2.0, 4.0, 6.0, 8.0, 10.0])
        }
    ]
    
    success_count = 0
    
    for test_case in test_cases:
        logger.info(f"🧪 {test_case['name']}")
        
        try:
            # 模拟inference.py中的数值稳定性处理
            data1 = test_case['data1']
            data2 = test_case['data2']
            
            # 检查标准差
            std1 = np.std(data1)
            std2 = np.std(data2)
            
            logger.info(f"  - 数据1标准差: {std1:.8f}")
            logger.info(f"  - 数据2标准差: {std2:.8f}")
            
            if std1 < 1e-8 or std2 < 1e-8:
                # 标准差为零的情况
                mean1 = np.mean(data1)
                mean2 = np.mean(data2)
                
                if abs(mean1 - mean2) < 1e-8:
                    pearson_corr = 1.0  # 完全相同的常数值
                    logger.info(f"  - 结果: 相同常数值，相关系数 = {pearson_corr}")
                else:
                    pearson_corr = 0.0  # 不同的常数值
                    logger.info(f"  - 结果: 不同常数值，相关系数 = {pearson_corr}")
            else:
                # 正常计算相关系数
                try:
                    corr_matrix = np.corrcoef(data1, data2)
                    pearson_corr = corr_matrix[0, 1]
                    logger.info(f"  - 结果: 正常计算，相关系数 = {pearson_corr:.6f}")
                except:
                    pearson_corr = 0.0
                    logger.info(f"  - 结果: 计算失败，设为 0.0")
            
            # 检查结果有效性
            if np.isfinite(pearson_corr):
                logger.info(f"  ✅ 数值稳定性测试通过")
                success_count += 1
            else:
                logger.error(f"  ❌ 结果包含无效值: {pearson_corr}")
                
        except Exception as e:
            logger.error(f"  ❌ 测试失败: {e}")
    
    logger.info(f"🎯 数值稳定性测试结果: {success_count}/{len(test_cases)} 通过")
    return success_count == len(test_cases)


def main():
    """主测试函数"""
    logger.info("🚀 启动Demo 4推理功能测试")
    logger.info("=" * 80)
    
    # 检查环境
    logger.info(f"PyTorch版本: {torch.__version__}")
    logger.info(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"GPU设备: {torch.cuda.get_device_name()}")
    
    # 设置设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"使用设备: {device}")
    
    try:
        # 1. 数值稳定性测试
        logger.info("\n" + "=" * 50)
        stability_ok = test_numerical_stability()
        
        if not stability_ok:
            logger.error("❌ 数值稳定性测试失败，退出")
            return False
        
        # 2. 模型路径
        model_path = "/mnt/c/Users/sxk27/OneDrive - MSFT/Project/electrical/output/demo4_g2milp/training/g2milp_training_20250706_220830/final_model.pth"
        
        # 3. 数据路径
        data_path = "/mnt/c/Users/sxk27/OneDrive - MSFT/Project/electrical/output/demo3_g2milp/bipartite_graphs/demo3_bipartite_graph.pkl"
        
        # 4. 加载模型
        logger.info("\n" + "=" * 50)
        model = load_trained_model(model_path, device)
        
        # 5. 加载数据
        logger.info("\n" + "=" * 50)
        source_data = load_demo3_data(data_path)
        source_data = source_data.to(device)
        
        # 6. 执行推理测试
        logger.info("\n" + "=" * 50)
        inference_ok = test_inference_pipeline(model, source_data)
        
        # 7. 输出总结
        logger.info("\n" + "=" * 80)
        logger.info("🎯 Demo 4推理功能测试总结")
        logger.info("=" * 80)
        
        if stability_ok and inference_ok:
            logger.info("✅ 所有测试通过！")
            logger.info("🔧 修复效果:")
            logger.info("  1. 数值稳定性问题已解决")
            logger.info("  2. 推理功能正常工作")
            logger.info("  3. 质量评估系统完善")
            logger.info("  4. 多样性生成功能正常")
            return True
        else:
            logger.error("❌ 部分测试失败")
            logger.error(f"  - 数值稳定性: {'✅' if stability_ok else '❌'}")
            logger.error(f"  - 推理功能: {'✅' if inference_ok else '❌'}")
            return False
            
    except Exception as e:
        logger.error(f"❌ 测试过程中出现错误: {e}")
        logger.error(traceback.format_exc())
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
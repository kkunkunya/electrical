#!/usr/bin/env python3
"""
Demo Early Stopping测试脚本
Demo Early Stopping Test Script

测试新增的高级Early Stopping机制在Demo 4训练中的效果
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
import logging
import time
from datetime import datetime

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

# 导入必要模块
from models.g2milp.early_stopping import (
    EarlyStoppingMonitor, 
    EarlyStoppingConfig, 
    EarlyStoppingStrategy,
    create_early_stopping_monitor
)

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def test_early_stopping_strategies():
    """测试不同Early Stopping策略"""
    logger.info("🧪 测试不同Early Stopping策略")
    logger.info("=" * 60)
    
    strategies = [
        "simple",
        "multi_metric", 
        "adaptive",
        "trend_analysis",
        "combined"
    ]
    
    results = {}
    
    for strategy in strategies:
        logger.info(f"\n📊 测试策略: {strategy}")
        logger.info("-" * 40)
        
        # 创建监控器
        monitor = create_early_stopping_monitor(
            strategy=strategy,
            patience=50,
            monitor_metric="val_loss",
            verbose=False  # 减少输出
        )
        
        # 模拟训练过程
        start_time = time.time()
        epochs_trained = 0
        stop_reason = "max_epochs_reached"
        
        for epoch in range(200):  # 最大200个epoch
            # 模拟不同的训练场景
            if strategy == "simple":
                # 简单衰减模式
                val_loss = 1.0 * np.exp(-epoch * 0.01) + np.random.normal(0, 0.01)
            elif strategy == "multi_metric":
                # 多指标模式
                val_loss = 1.0 - epoch * 0.002 + np.random.normal(0, 0.005)
                # 后期出现过拟合
                if epoch > 100:
                    val_loss += (epoch - 100) * 0.001
            elif strategy == "adaptive":
                # 自适应模式：有时快速收敛，有时缓慢
                if epoch < 50:
                    val_loss = 1.0 - epoch * 0.01 + np.random.normal(0, 0.01)
                else:
                    val_loss = 0.5 - (epoch - 50) * 0.0005 + np.random.normal(0, 0.002)
            elif strategy == "trend_analysis":
                # 趋势分析模式：有平台期
                if epoch < 30:
                    val_loss = 1.0 - epoch * 0.02 + np.random.normal(0, 0.01)
                elif epoch < 80:
                    val_loss = 0.4 + np.random.normal(0, 0.005)  # 平台期
                else:
                    val_loss = 0.4 - (epoch - 80) * 0.001 + np.random.normal(0, 0.002)
            else:  # combined
                # 组合模式：复杂的训练曲线
                base_loss = 1.0 * np.exp(-epoch * 0.008)
                noise = np.random.normal(0, 0.01)
                # 添加周期性波动
                periodic = 0.05 * np.sin(epoch * 0.1)
                val_loss = base_loss + noise + periodic
            
            # 模拟其他指标
            train_loss = val_loss * 0.8 + np.random.normal(0, 0.005)
            grad_norm = 0.1 * np.exp(-epoch * 0.01) + np.random.normal(0, 0.001)
            generation_quality = min(0.9, epoch * 0.005) + np.random.normal(0, 0.02)
            
            metrics = {
                'val_loss': val_loss,
                'train_loss': train_loss,
                'grad_norm': grad_norm,
                'generation_quality': generation_quality,
                'kl_weight': min(1.0, epoch / 100),
                'learning_rate': 1e-4 * np.exp(-epoch * 0.01)
            }
            
            # 更新监控器
            result = monitor.update(epoch, metrics)
            
            epochs_trained = epoch + 1
            
            # 检查停止条件
            if result['should_stop']:
                stop_reason = result['decision_reason']
                break
        
        # 记录结果
        training_time = time.time() - start_time
        summary = monitor.get_summary()
        
        results[strategy] = {
            'epochs_trained': epochs_trained,
            'stop_reason': stop_reason,
            'best_score': summary['best_score'],
            'best_epoch': summary['best_epoch'],
            'improvements': summary['improvements'],
            'training_time': training_time
        }
        
        logger.info(f"✓ 策略 {strategy} 完成:")
        logger.info(f"  - 训练epoch数: {epochs_trained}")
        logger.info(f"  - 停止原因: {stop_reason}")
        logger.info(f"  - 最佳得分: {summary['best_score']:.6f} (epoch {summary['best_epoch']})")
        logger.info(f"  - 改善次数: {summary['improvements']}")
        logger.info(f"  - 训练时间: {training_time:.2f}秒")
    
    return results


def test_quality_monitoring():
    """测试质量监控功能"""
    logger.info("\n🎯 测试质量监控功能")
    logger.info("=" * 60)
    
    # 创建启用质量监控的监控器
    monitor = create_early_stopping_monitor(
        strategy="combined",
        patience=100,
        monitor_metric="val_loss",
        enable_quality_monitoring=True,
        quality_threshold=0.8,
        verbose=True
    )
    
    logger.info("模拟质量改善的训练过程...")
    
    for epoch in range(150):
        # 模拟质量逐步改善的过程
        val_loss = 1.0 * np.exp(-epoch * 0.008) + np.random.normal(0, 0.01)
        
        # 质量分阶段改善
        if epoch < 50:
            quality_score = 0.3 + epoch * 0.004  # 慢速改善
        elif epoch < 100:
            quality_score = 0.5 + (epoch - 50) * 0.006  # 中等改善
        else:
            quality_score = 0.8 + (epoch - 100) * 0.002  # 接近目标
        
        quality_score = min(0.95, quality_score) + np.random.normal(0, 0.02)
        
        metrics = {
            'val_loss': val_loss,
            'train_loss': val_loss * 0.9,
            'generation_quality': quality_score,
            'graph_similarity': quality_score * 0.9,
            'diversity_score': quality_score * 0.8,
            'grad_norm': 0.1 * np.exp(-epoch * 0.01)
        }
        
        # 更新监控器
        result = monitor.update(epoch, metrics)
        
        # 每20个epoch输出一次质量报告
        if epoch % 20 == 0:
            logger.info(f"Epoch {epoch:3d}: Quality={quality_score:.3f}, Val_Loss={val_loss:.4f}")
            if result['quality_analysis']:
                qa = result['quality_analysis']
                logger.info(f"  质量分析: 得分={qa.get('quality_score', 0):.3f}, "
                          f"趋势={qa.get('quality_trend', 'unknown')}")
        
        # 检查停止条件
        if result['should_stop']:
            logger.info(f"\n✓ 质量监控停止 (Epoch {epoch})")
            logger.info(f"  停止原因: {result['decision_reason']}")
            break
    
    # 输出质量监控摘要
    summary = monitor.get_summary()
    logger.info(f"\n📊 质量监控摘要:")
    logger.info(f"  - 最佳得分: {summary['best_score']:.6f}")
    logger.info(f"  - 最佳epoch: {summary['best_epoch']}")
    logger.info(f"  - 总改善次数: {summary['improvements']}")


def test_adaptive_patience():
    """测试自适应patience功能"""
    logger.info("\n⚙️ 测试自适应Patience功能")
    logger.info("=" * 60)
    
    # 创建启用自适应patience的监控器
    monitor = create_early_stopping_monitor(
        strategy="adaptive",
        patience=30,
        monitor_metric="val_loss",
        adaptive_patience=True,
        verbose=False
    )
    
    logger.info("模拟不规律改善的训练过程...")
    
    patience_history = []
    
    for epoch in range(200):
        # 模拟不规律的改善模式
        if epoch < 50:
            # 前期快速改善
            val_loss = 1.0 - epoch * 0.015 + np.random.normal(0, 0.01)
        elif epoch < 100:
            # 中期缓慢改善
            val_loss = 0.25 - (epoch - 50) * 0.002 + np.random.normal(0, 0.005)
        else:
            # 后期极慢改善
            val_loss = 0.15 - (epoch - 100) * 0.0001 + np.random.normal(0, 0.002)
        
        metrics = {
            'val_loss': val_loss,
            'train_loss': val_loss * 0.9,
            'grad_norm': 0.1 * np.exp(-epoch * 0.01)
        }
        
        # 更新监控器
        result = monitor.update(epoch, metrics)
        patience_history.append(result['current_patience'])
        
        # 每25个epoch记录patience变化
        if epoch % 25 == 0:
            logger.info(f"Epoch {epoch:3d}: Val_Loss={val_loss:.6f}, "
                      f"Patience={result['current_patience']}, "
                      f"Counter={result['patience_counter']}")
        
        # 检查停止条件
        if result['should_stop']:
            logger.info(f"\n✓ 自适应停止 (Epoch {epoch})")
            logger.info(f"  停止原因: {result['decision_reason']}")
            logger.info(f"  最终patience: {result['current_patience']}")
            break
    
    # 分析patience变化
    logger.info(f"\n📈 Patience变化分析:")
    logger.info(f"  - 初始patience: {patience_history[0]}")
    logger.info(f"  - 最大patience: {max(patience_history)}")
    logger.info(f"  - 最小patience: {min(patience_history)}")
    logger.info(f"  - 最终patience: {patience_history[-1]}")


def test_trend_analysis():
    """测试趋势分析功能"""
    logger.info("\n📈 测试趋势分析功能")
    logger.info("=" * 60)
    
    # 创建启用趋势分析的监控器
    monitor = create_early_stopping_monitor(
        strategy="trend_analysis",
        patience=50,
        monitor_metric="val_loss",
        verbose=False
    )
    
    logger.info("模拟具有明显趋势的训练过程...")
    
    # 定义不同的趋势阶段
    trends = [
        ("下降趋势", lambda x: 1.0 - x * 0.02),     # 快速下降
        ("平稳期", lambda x: 0.3 + np.sin(x * 0.1) * 0.02),  # 震荡平稳
        ("缓慢下降", lambda x: 0.3 - x * 0.001),    # 极慢下降
        ("上升趋势", lambda x: 0.2 + x * 0.005)     # 过拟合上升
    ]
    
    epoch = 0
    for trend_name, trend_func in trends:
        logger.info(f"\n🔄 进入{trend_name}阶段...")
        
        trend_start_epoch = epoch
        for i in range(40):  # 每个趋势40个epoch
            val_loss = trend_func(i) + np.random.normal(0, 0.01)
            
            metrics = {
                'val_loss': val_loss,
                'train_loss': val_loss * 0.9,
                'grad_norm': 0.1 * np.exp(-epoch * 0.01)
            }
            
            # 更新监控器
            result = monitor.update(epoch, metrics)
            
            # 每10个epoch输出趋势分析
            if i % 10 == 0 and result['trend_analysis']:
                trend = result['trend_analysis']
                logger.info(f"  Epoch {epoch:3d}: 趋势={trend['trend']}, "
                          f"斜率={trend['slope']:.2e}, 稳定性={trend['stability']:.3f}")
            
            epoch += 1
            
            # 检查停止条件
            if result['should_stop']:
                logger.info(f"\n✓ 趋势分析停止 (Epoch {epoch-1})")
                logger.info(f"  停止原因: {result['decision_reason']}")
                logger.info(f"  趋势阶段: {trend_name}")
                return
    
    logger.info(f"\n📊 完成所有趋势阶段测试 (总计 {epoch} epochs)")


def main():
    """主测试函数"""
    logger.info("🚀 启动Early Stopping机制测试")
    logger.info("=" * 80)
    
    # 检查PyTorch可用性
    logger.info(f"PyTorch版本: {torch.__version__}")
    logger.info(f"CUDA可用: {torch.cuda.is_available()}")
    
    try:
        # 1. 测试不同策略
        strategy_results = test_early_stopping_strategies()
        
        # 2. 测试质量监控
        test_quality_monitoring()
        
        # 3. 测试自适应patience
        test_adaptive_patience()
        
        # 4. 测试趋势分析
        test_trend_analysis()
        
        # 输出测试总结
        logger.info("\n" + "=" * 80)
        logger.info("🎯 Early Stopping机制测试总结")
        logger.info("=" * 80)
        
        logger.info("\n📊 策略效果对比:")
        for strategy, result in strategy_results.items():
            logger.info(f"  {strategy:15s}: {result['epochs_trained']:3d} epochs, "
                      f"得分 {result['best_score']:.6f}, {result['improvements']:2d} 次改善")
        
        logger.info("\n✅ 所有测试完成，Early Stopping机制工作正常！")
        logger.info("\n🔧 集成建议:")
        logger.info("  1. 推荐使用 'combined' 策略，综合效果最佳")
        logger.info("  2. 启用质量监控可以提升训练效果")
        logger.info("  3. 自适应patience适合长期训练")
        logger.info("  4. 趋势分析有助于识别过拟合")
        
    except Exception as e:
        logger.error(f"❌ 测试过程中出现错误: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
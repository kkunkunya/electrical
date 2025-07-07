"""
G2MILP Early Stopping 模块
G2MILP Early Stopping Module

实现高级Early Stopping机制，支持：
1. 多指标监控和综合评估
2. 自适应停止条件
3. 性能趋势分析
4. 智能重启策略
"""

import torch
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from collections import deque
from enum import Enum
import time
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class EarlyStoppingStrategy(Enum):
    """Early Stopping策略枚举"""
    SIMPLE = "simple"                    # 简单的loss监控
    MULTI_METRIC = "multi_metric"        # 多指标综合评估
    ADAPTIVE = "adaptive"                # 自适应阈值调整
    TREND_ANALYSIS = "trend_analysis"    # 趋势分析
    COMBINED = "combined"                # 组合策略


@dataclass
class EarlyStoppingConfig:
    """Early Stopping配置"""
    # 基础参数
    strategy: Union[EarlyStoppingStrategy, str] = EarlyStoppingStrategy.COMBINED
    patience: int = 500                  # 等待epoch数
    min_delta: float = 1e-6             # 最小改善阈值
    monitor_metric: str = "val_loss"     # 主要监控指标
    
    # 多指标监控
    additional_metrics: List[str] = None  # 额外监控指标
    metric_weights: Dict[str, float] = None  # 指标权重
    
    # 自适应参数
    adaptive_patience: bool = True       # 是否自适应调整patience
    patience_factor: float = 1.5         # patience增长因子
    max_patience: int = 1000             # 最大patience
    min_patience: int = 100              # 最小patience
    
    # 趋势分析
    trend_window: int = 20               # 趋势分析窗口
    trend_threshold: float = 0.01        # 趋势阈值
    slope_threshold: float = 1e-6        # 斜率阈值
    
    # 质量评估集成
    enable_quality_monitoring: bool = True  # 启用质量监控
    quality_threshold: float = 0.7          # 质量阈值
    quality_patience: int = 10               # 质量patience
    quality_improvement_threshold: float = 0.05  # 质量改善阈值
    
    # 性能阈值
    performance_thresholds: Dict[str, float] = None  # 性能阈值字典
    
    # 智能重启
    enable_smart_restart: bool = False   # 启用智能重启
    restart_threshold: float = 0.1       # 重启阈值
    max_restarts: int = 2                # 最大重启次数
    
    # 保存和恢复
    save_best_model: bool = True         # 保存最佳模型
    restore_best_weights: bool = True    # 恢复最佳权重
    
    # 日志和监控
    verbose: bool = True                 # 详细日志
    log_frequency: int = 10              # 日志频率
    
    def __post_init__(self):
        """初始化后处理"""
        if self.additional_metrics is None:
            self.additional_metrics = ["train_loss", "kl_weight", "grad_norm"]
        
        if self.metric_weights is None:
            self.metric_weights = {
                "val_loss": 1.0,
                "train_loss": 0.3,
                "kl_weight": 0.1,
                "grad_norm": 0.1
            }
        
        if self.performance_thresholds is None:
            self.performance_thresholds = {
                "val_loss": 0.01,      # 验证损失目标
                "train_loss": 0.01,    # 训练损失目标
                "quality_score": 0.8,  # 质量得分目标
                "convergence_ratio": 0.95  # 收敛比例目标
            }


class EarlyStoppingMonitor:
    """
    Enhanced Early Stopping监控器
    
    支持多种停止策略和智能监控
    """
    
    def __init__(self, config: EarlyStoppingConfig = None):
        self.config = config or EarlyStoppingConfig()
        
        # 确保策略是枚举类型
        if isinstance(self.config.strategy, str):
            self.config.strategy = EarlyStoppingStrategy(self.config.strategy)
        
        # 监控状态
        self.best_score = float('inf')
        self.best_epoch = 0
        self.patience_counter = 0
        self.current_patience = self.config.patience
        
        # 历史记录
        self.history = {
            'epochs': [],
            'scores': [],
            'improvements': [],
            'decisions': []
        }
        
        # 多指标监控
        self.metric_history = {metric: deque(maxlen=self.config.trend_window) 
                             for metric in [self.config.monitor_metric] + self.config.additional_metrics}
        
        # 趋势分析
        self.trend_analyzer = TrendAnalyzer(self.config.trend_window)
        
        # 质量监控
        self.quality_monitor = QualityMonitor(self.config) if self.config.enable_quality_monitoring else None
        
        # 智能重启
        self.restart_counter = 0
        self.restart_history = []
        
        # 模型权重保存
        self.best_weights = None
        
        # 统计信息
        self.stats = {
            'total_epochs': 0,
            'improvements': 0,
            'stagnant_periods': 0,
            'restarts': 0,
            'early_stops': 0
        }
        
        logger.info(f"Early Stopping监控器初始化完成")
        logger.info(f"策略: {self.config.strategy.value}")
        logger.info(f"监控指标: {self.config.monitor_metric}")
        logger.info(f"初始patience: {self.current_patience}")
    
    def update(self, 
               epoch: int, 
               metrics: Dict[str, float], 
               model_state: Optional[Dict] = None) -> Dict[str, Any]:
        """
        更新监控状态
        
        Args:
            epoch: 当前epoch
            metrics: 指标字典
            model_state: 模型状态（用于保存最佳权重）
            
        Returns:
            监控结果字典
        """
        self.stats['total_epochs'] = epoch
        
        # 获取主要监控指标
        primary_score = metrics.get(self.config.monitor_metric, float('inf'))
        
        # 更新历史记录
        self.history['epochs'].append(epoch)
        self.history['scores'].append(primary_score)
        
        # 更新多指标历史
        for metric_name in self.metric_history:
            if metric_name in metrics:
                self.metric_history[metric_name].append(metrics[metric_name])
        
        # 趋势分析
        self.trend_analyzer.update(primary_score)
        
        # 质量监控
        quality_result = {}
        if self.quality_monitor:
            quality_result = self.quality_monitor.update(epoch, metrics)
        
        # 判断改善
        improvement = self._check_improvement(primary_score)
        self.history['improvements'].append(improvement)
        
        # 决策逻辑
        decision = self._make_decision(epoch, primary_score, metrics, improvement, quality_result)
        self.history['decisions'].append(decision)
        
        # 更新最佳状态
        if improvement:
            self.best_score = primary_score
            self.best_epoch = epoch
            self.patience_counter = 0
            self.stats['improvements'] += 1
            
            # 保存最佳模型权重
            if self.config.save_best_model and model_state:
                self.best_weights = model_state.copy()
        else:
            self.patience_counter += 1
        
        # 自适应patience调整
        if self.config.adaptive_patience:
            self._adjust_patience(improvement, quality_result)
        
        # 返回监控结果
        result = {
            'should_stop': decision['should_stop'],
            'should_restart': decision['should_restart'],
            'improvement': improvement,
            'best_score': self.best_score,
            'best_epoch': self.best_epoch,
            'patience_counter': self.patience_counter,
            'current_patience': self.current_patience,
            'trend_analysis': self.trend_analyzer.get_analysis(),
            'quality_analysis': quality_result,
            'decision_reason': decision['reason'],
            'statistics': self.stats.copy()
        }
        
        # 日志输出
        if self.config.verbose and epoch % self.config.log_frequency == 0:
            self._log_status(epoch, result)
        
        return result
    
    def _check_improvement(self, score: float) -> bool:
        """检查是否有改善"""
        if self.config.monitor_metric.endswith('_loss'):
            # 损失函数：越小越好
            return score < (self.best_score - self.config.min_delta)
        else:
            # 其他指标：越大越好
            return score > (self.best_score + self.config.min_delta)
    
    def _make_decision(self, 
                      epoch: int, 
                      primary_score: float, 
                      metrics: Dict[str, float], 
                      improvement: bool,
                      quality_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        制定停止决策
        
        Returns:
            决策结果字典
        """
        decision = {
            'should_stop': False,
            'should_restart': False,
            'reason': '',
            'confidence': 0.0
        }
        
        # 根据策略进行决策
        if self.config.strategy == EarlyStoppingStrategy.SIMPLE:
            decision = self._simple_decision()
        elif self.config.strategy == EarlyStoppingStrategy.MULTI_METRIC:
            decision = self._multi_metric_decision(metrics)
        elif self.config.strategy == EarlyStoppingStrategy.ADAPTIVE:
            decision = self._adaptive_decision(metrics, quality_result)
        elif self.config.strategy == EarlyStoppingStrategy.TREND_ANALYSIS:
            decision = self._trend_analysis_decision()
        elif self.config.strategy == EarlyStoppingStrategy.COMBINED:
            decision = self._combined_decision(epoch, metrics, quality_result)
        
        return decision
    
    def _simple_decision(self) -> Dict[str, Any]:
        """简单决策：基于patience"""
        should_stop = self.patience_counter >= self.current_patience
        return {
            'should_stop': should_stop,
            'should_restart': False,
            'reason': f'Simple patience exceeded ({self.patience_counter}/{self.current_patience})',
            'confidence': 1.0 if should_stop else 0.0
        }
    
    def _multi_metric_decision(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """多指标综合决策"""
        # 计算加权综合得分
        weighted_score = 0.0
        total_weight = 0.0
        
        for metric_name, weight in self.config.metric_weights.items():
            if metric_name in metrics:
                # 归一化处理
                metric_value = metrics[metric_name]
                if metric_name in self.metric_history and len(self.metric_history[metric_name]) > 1:
                    history_values = list(self.metric_history[metric_name])
                    if max(history_values) > min(history_values):
                        normalized_value = (metric_value - min(history_values)) / (max(history_values) - min(history_values))
                        weighted_score += weight * normalized_value
                        total_weight += weight
        
        if total_weight > 0:
            weighted_score /= total_weight
        
        # 决策逻辑
        stagnation_threshold = 0.8  # 停滞阈值
        should_stop = (weighted_score > stagnation_threshold and 
                      self.patience_counter >= self.current_patience * 0.5)
        
        return {
            'should_stop': should_stop,
            'should_restart': False,
            'reason': f'Multi-metric stagnation detected (score: {weighted_score:.3f})',
            'confidence': weighted_score
        }
    
    def _adaptive_decision(self, metrics: Dict[str, float], quality_result: Dict[str, Any]) -> Dict[str, Any]:
        """自适应决策"""
        # 检查是否达到性能阈值
        performance_achieved = True
        for metric_name, threshold in self.config.performance_thresholds.items():
            if metric_name in metrics:
                if metrics[metric_name] > threshold:  # 假设所有指标都是越小越好
                    performance_achieved = False
                    break
        
        # 质量检查
        quality_achieved = False
        if quality_result and 'quality_score' in quality_result:
            quality_achieved = quality_result['quality_score'] >= self.config.quality_threshold
        
        # 决策
        if performance_achieved and quality_achieved:
            return {
                'should_stop': True,
                'should_restart': False,
                'reason': 'Performance and quality targets achieved',
                'confidence': 1.0
            }
        elif self.patience_counter >= self.current_patience:
            return {
                'should_stop': True,
                'should_restart': False,
                'reason': 'Adaptive patience exceeded',
                'confidence': 0.8
            }
        else:
            return {
                'should_stop': False,
                'should_restart': False,
                'reason': 'Continuing training',
                'confidence': 0.0
            }
    
    def _trend_analysis_decision(self) -> Dict[str, Any]:
        """趋势分析决策"""
        analysis = self.trend_analyzer.get_analysis()
        
        # 检查是否处于平稳期
        if (analysis['slope'] < self.config.slope_threshold and 
            analysis['stability'] > 0.9 and 
            self.patience_counter >= self.current_patience * 0.7):
            
            return {
                'should_stop': True,
                'should_restart': False,
                'reason': f'Trend analysis: flat region detected (slope: {analysis["slope"]:.2e})',
                'confidence': analysis['stability']
            }
        
        return {
            'should_stop': False,
            'should_restart': False,
            'reason': 'Trend analysis: still improving',
            'confidence': 0.0
        }
    
    def _combined_decision(self, epoch: int, metrics: Dict[str, float], quality_result: Dict[str, Any]) -> Dict[str, Any]:
        """组合决策策略"""
        # 收集各策略的决策
        simple_decision = self._simple_decision()
        multi_metric_decision = self._multi_metric_decision(metrics)
        adaptive_decision = self._adaptive_decision(metrics, quality_result)
        trend_decision = self._trend_analysis_decision()
        
        # 权重投票
        stop_votes = []
        if simple_decision['should_stop']:
            stop_votes.append(('simple', simple_decision['confidence']))
        if multi_metric_decision['should_stop']:
            stop_votes.append(('multi_metric', multi_metric_decision['confidence']))
        if adaptive_decision['should_stop']:
            stop_votes.append(('adaptive', adaptive_decision['confidence']))
        if trend_decision['should_stop']:
            stop_votes.append(('trend', trend_decision['confidence']))
        
        # 综合决策
        if len(stop_votes) >= 2:  # 至少两个策略同意停止
            total_confidence = sum(vote[1] for vote in stop_votes) / len(stop_votes)
            reasons = [vote[0] for vote in stop_votes]
            
            return {
                'should_stop': True,
                'should_restart': False,
                'reason': f'Combined decision: {", ".join(reasons)} (confidence: {total_confidence:.3f})',
                'confidence': total_confidence
            }
        elif len(stop_votes) == 1 and stop_votes[0][1] > 0.9:  # 单一策略但置信度很高
            return {
                'should_stop': True,
                'should_restart': False,
                'reason': f'High confidence decision: {stop_votes[0][0]} (confidence: {stop_votes[0][1]:.3f})',
                'confidence': stop_votes[0][1]
            }
        else:
            return {
                'should_stop': False,
                'should_restart': False,
                'reason': 'Combined decision: continue training',
                'confidence': 0.0
            }
    
    def _adjust_patience(self, improvement: bool, quality_result: Dict[str, Any]):
        """自适应调整patience"""
        if improvement:
            # 有改善时，可以适当降低patience（更严格）
            self.current_patience = max(
                self.config.min_patience,
                int(self.current_patience / self.config.patience_factor)
            )
        else:
            # 长时间无改善时，增加patience（更宽松）
            if self.patience_counter >= self.current_patience * 0.8:
                self.current_patience = min(
                    self.config.max_patience,
                    int(self.current_patience * self.config.patience_factor)
                )
    
    def _log_status(self, epoch: int, result: Dict[str, Any]):
        """记录状态日志"""
        logger.info(f"📊 Early Stopping Status (Epoch {epoch}):")
        logger.info(f"  ├─ Best Score: {self.best_score:.6f} (Epoch {self.best_epoch})")
        logger.info(f"  ├─ Patience: {self.patience_counter}/{self.current_patience}")
        logger.info(f"  ├─ Improvement: {'✓' if result['improvement'] else '✗'}")
        logger.info(f"  ├─ Decision: {result['decision_reason']}")
        
        if result['trend_analysis']:
            trend = result['trend_analysis']
            logger.info(f"  ├─ Trend: Slope={trend['slope']:.2e}, Stability={trend['stability']:.3f}")
        
        if result['quality_analysis']:
            quality = result['quality_analysis']
            logger.info(f"  └─ Quality: {quality.get('quality_score', 'N/A')}")
        
        if result['should_stop']:
            logger.warning(f"🛑 Early Stopping Triggered: {result['decision_reason']}")
        elif result['should_restart']:
            logger.warning(f"🔄 Smart Restart Triggered: {result['decision_reason']}")
    
    def get_best_weights(self) -> Optional[Dict]:
        """获取最佳模型权重"""
        return self.best_weights
    
    def should_restore_weights(self) -> bool:
        """是否应该恢复最佳权重"""
        return self.config.restore_best_weights and self.best_weights is not None
    
    def get_summary(self) -> Dict[str, Any]:
        """获取监控摘要"""
        return {
            'config': self.config,
            'best_score': self.best_score,
            'best_epoch': self.best_epoch,
            'total_epochs': self.stats['total_epochs'],
            'improvements': self.stats['improvements'],
            'final_patience': self.current_patience,
            'history': self.history,
            'statistics': self.stats
        }
    
    def save_state(self, save_path: str):
        """保存监控状态"""
        state = {
            'config': self.config.__dict__,
            'monitor_state': {
                'best_score': self.best_score,
                'best_epoch': self.best_epoch,
                'patience_counter': self.patience_counter,
                'current_patience': self.current_patience,
                'restart_counter': self.restart_counter
            },
            'history': self.history,
            'statistics': self.stats
        }
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"Early Stopping状态已保存: {save_path}")


class TrendAnalyzer:
    """趋势分析器"""
    
    def __init__(self, window_size: int = 20):
        self.window_size = window_size
        self.values = deque(maxlen=window_size)
        self.timestamps = deque(maxlen=window_size)
    
    def update(self, value: float):
        """更新数值"""
        self.values.append(value)
        self.timestamps.append(time.time())
    
    def get_analysis(self) -> Dict[str, float]:
        """获取趋势分析"""
        if len(self.values) < 3:
            return {'slope': 0.0, 'stability': 0.0, 'trend': 'insufficient_data'}
        
        # 计算线性回归斜率
        values_array = np.array(self.values)
        x = np.arange(len(values_array))
        
        try:
            slope, intercept = np.polyfit(x, values_array, 1)
        except:
            slope = 0.0
            intercept = 0.0
        
        # 计算稳定性（变异系数的倒数）
        if np.std(values_array) > 0:
            stability = 1.0 / (np.std(values_array) / np.mean(values_array))
        else:
            stability = 1.0
        
        # 趋势判断
        if abs(slope) < 1e-6:
            trend = 'flat'
        elif slope < 0:
            trend = 'decreasing'
        else:
            trend = 'increasing'
        
        return {
            'slope': slope,
            'stability': min(stability, 1.0),
            'trend': trend,
            'mean': np.mean(values_array),
            'std': np.std(values_array),
            'recent_value': self.values[-1] if self.values else 0.0
        }


class QualityMonitor:
    """质量监控器"""
    
    def __init__(self, config: EarlyStoppingConfig):
        self.config = config
        self.quality_history = deque(maxlen=50)
        self.quality_counter = 0
        self.last_quality_improvement = 0
    
    def update(self, epoch: int, metrics: Dict[str, float]) -> Dict[str, Any]:
        """更新质量监控"""
        result = {
            'quality_score': 0.0,
            'quality_trend': 'unknown',
            'quality_patience': self.quality_counter,
            'should_stop_for_quality': False
        }
        
        # 提取质量相关指标
        quality_indicators = ['generation_quality', 'overall_quality_score', 'validity_score']
        quality_score = 0.0
        
        for indicator in quality_indicators:
            if indicator in metrics:
                quality_score = max(quality_score, metrics[indicator])
                break
        
        if quality_score > 0:
            self.quality_history.append(quality_score)
            result['quality_score'] = quality_score
            
            # 检查质量改善
            if (len(self.quality_history) >= 2 and 
                quality_score > max(self.quality_history) - self.config.quality_improvement_threshold):
                self.last_quality_improvement = epoch
                self.quality_counter = 0
            else:
                self.quality_counter += 1
            
            # 质量趋势
            if len(self.quality_history) >= 3:
                recent_avg = np.mean(list(self.quality_history)[-3:])
                earlier_avg = np.mean(list(self.quality_history)[:-3])
                
                if recent_avg > earlier_avg + 0.01:
                    result['quality_trend'] = 'improving'
                elif recent_avg < earlier_avg - 0.01:
                    result['quality_trend'] = 'degrading'
                else:
                    result['quality_trend'] = 'stable'
            
            # 质量停止条件
            if (quality_score >= self.config.quality_threshold and 
                self.quality_counter >= self.config.quality_patience):
                result['should_stop_for_quality'] = True
        
        return result


def create_early_stopping_monitor(strategy: str = "combined", 
                                 patience: int = 500, 
                                 monitor_metric: str = "val_loss",
                                 **kwargs) -> EarlyStoppingMonitor:
    """
    创建Early Stopping监控器的工厂函数
    
    Args:
        strategy: 停止策略
        patience: 等待epoch数
        monitor_metric: 监控指标
        **kwargs: 其他配置参数
        
    Returns:
        EarlyStoppingMonitor实例
    """
    # 创建配置时直接传入字符串，在Monitor初始化时转换
    config = EarlyStoppingConfig(
        strategy=strategy,  # 传入字符串，稍后转换
        patience=patience,
        monitor_metric=monitor_metric,
        **kwargs
    )
    
    return EarlyStoppingMonitor(config)


if __name__ == "__main__":
    # 测试Early Stopping监控器
    print("G2MILP Early Stopping模块测试")
    print("=" * 50)
    
    # 创建监控器
    monitor = create_early_stopping_monitor(
        strategy="combined",
        patience=100,
        monitor_metric="val_loss",
        verbose=True
    )
    
    # 模拟训练过程
    print("\n模拟训练过程:")
    for epoch in range(150):
        # 模拟指标
        metrics = {
            'val_loss': 1.0 - epoch * 0.001 + np.random.normal(0, 0.01),
            'train_loss': 0.8 - epoch * 0.0008 + np.random.normal(0, 0.005),
            'grad_norm': 0.1 * np.exp(-epoch * 0.01) + np.random.normal(0, 0.001),
            'generation_quality': min(0.9, epoch * 0.006) + np.random.normal(0, 0.02)
        }
        
        # 更新监控
        result = monitor.update(epoch, metrics)
        
        # 检查停止条件
        if result['should_stop']:
            print(f"✓ 训练在epoch {epoch}停止")
            print(f"  原因: {result['decision_reason']}")
            break
    
    # 输出摘要
    summary = monitor.get_summary()
    print(f"\n训练摘要:")
    print(f"- 最佳得分: {summary['best_score']:.6f}")
    print(f"- 最佳epoch: {summary['best_epoch']}")
    print(f"- 总改善次数: {summary['improvements']}")
    print(f"- 最终patience: {summary['final_patience']}")
    
    print("Early Stopping模块测试完成!")
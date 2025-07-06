"""
G2MILP 实例生成模块
G2MILP Instance Generation Module

本模块实现了G2MILP论文中的核心生成算法，包括：
1. Encoder-Decoder架构的图神经网络
2. Masked VAE范式的MILP实例生成
3. 基于约束遮盖的生成流程
4. 可控的相似度参数η

主要组件:
- encoder: GNN编码器，将二分图编码为潜向量
- decoder: 解码器，包含4个预测器
- masking: 约束遮盖过程
- generator: 主生成器
- training: 训练逻辑
- inference: 推理逻辑
"""

from .encoder import G2MILPEncoder, EncoderConfig, create_encoder
from .decoder import G2MILPDecoder, DecoderConfig, BiasPredictor, DegreePredictor, LogitsPredictor, WeightsPredictor, create_decoder
from .masking import ConstraintMasker, MaskingConfig, create_masker
from .generator import G2MILPGenerator, GeneratorConfig, create_generator
from .training import G2MILPTrainer, TrainingConfig, create_trainer
from .inference import G2MILPInference, InferenceConfig, create_inference_engine

__version__ = "1.0.0"
__author__ = "Demo 4 Development Team"

__all__ = [
    "G2MILPEncoder", "EncoderConfig", "create_encoder",
    "G2MILPDecoder", "DecoderConfig", "create_decoder",
    "BiasPredictor", "DegreePredictor", "LogitsPredictor", "WeightsPredictor",
    "ConstraintMasker", "MaskingConfig", "create_masker",
    "G2MILPGenerator", "GeneratorConfig", "create_generator",
    "G2MILPTrainer", "TrainingConfig", "create_trainer",
    "G2MILPInference", "InferenceConfig", "create_inference_engine"
]
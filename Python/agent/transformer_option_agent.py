"""
利用stable_baselines3中的PPO、TD3和SAC算法进行改进，使之能够使用transformer进行训练
"""
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.td3.policies import TD3Policy
from stable_baselines3.sac.policies import SACPolicy


class TransformerFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=128, nhead=4, num_layers=2):
        """
        使用Transformer编码器作为特征提取器
        
        参数:
        observation_space: 观察空间
        features_dim: 特征维度
        nhead: 多头注意力头数
        num_layers: Transformer层数
        """
        super().__init__(observation_space, features_dim)
        try:
            # 获取观察空间维度
            obs_dim = observation_space.shape[0]
            
            # Transformer需要将输入映射到相同的维度
            self.input_fc = nn.Linear(obs_dim, features_dim)
            
            # Transformer编码器层
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=features_dim, 
                nhead=nhead,
                dim_feedforward=features_dim * 4,
                dropout=0.1,
                batch_first=True
            )
            
            # Transformer编码器
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            
            # 输出层
            self.output_layer = nn.Sequential(
                nn.Linear(features_dim, features_dim),
                nn.ReLU()
            )
            
            print(f"初始化Transformer特征提取器成功: 输入维度={obs_dim}, 特征维度={features_dim}, 头数={nhead}, 层数={num_layers}")
        
        except Exception as e:
            print(f"初始化Transformer特征提取器时出错: {e}")
            raise

    def forward(self, observations):
        """
        前向传播
        
        参数:
        observations: 观察值 [batch_size, obs_dim]
        
        返回:
        特征向量 [batch_size, features_dim]
        """
        try:
            # 映射到特征维度
            x = self.input_fc(observations)
            
            # 添加批次维度 [batch_size, features_dim] -> [batch_size, 1, features_dim]
            if len(x.shape) == 2:
                x = x.unsqueeze(1)
            
            # 通过Transformer编码器
            x = self.transformer(x)
            
            # 取最后一个位置的输出
            x = x.squeeze(1) if x.shape[1] == 1 else x[:, 0, :]
            
            # 通过输出层
            x = self.output_layer(x)
            
            return x
            
        except Exception as e:
            print(f"Transformer前向传播错误: {e}")
            # 出错时返回零张量
            return torch.zeros((observations.shape[0], self.features_dim), 
                               device=observations.device)


class CustomTransformerPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        """PPO算法的自定义Transformer策略"""
        super().__init__(*args,
                         features_extractor_class=TransformerFeatureExtractor,
                         **kwargs)


class CustomTransformerTD3Policy(TD3Policy):
    def __init__(self, *args, **kwargs):
        """TD3算法的自定义Transformer策略"""
        super().__init__(*args,
                         features_extractor_class=TransformerFeatureExtractor,
                         **kwargs)


class CustomTransformerSACPolicy(SACPolicy):
    def __init__(self, *args, **kwargs):
        """SAC算法的自定义Transformer策略"""
        super().__init__(*args,
                         features_extractor_class=TransformerFeatureExtractor,
                         **kwargs)
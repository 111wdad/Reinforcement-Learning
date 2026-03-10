"""
main函数，用于训练模型并保存
"""
import os
import sys
import time
import yaml
import torch
import traceback
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import glob
from envs.train_env import TrainEnv
from utils import custom_model
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

def load_or_create_model(env, config_path, model_dir):
    """
    检测本地是否存在模型，存在则加载，不存在则创建新模型
    
    参数:
    env: 训练环境
    config_path: 配置文件路径
    model_dir: 模型保存目录
    
    返回:
    model: 加载的或新创建的模型
    is_new: 是否是新创建的模型
    """
    # 查找最新的模型文件
    model_files = glob.glob(os.path.join(model_dir, "model_*.zip"))
    final_model_path = os.path.join(model_dir, "final_model.zip")
    
    if os.path.exists(final_model_path):
        model_files.append(final_model_path)
    
    if model_files:
        # 按修改时间排序，获取最新的模型文件
        latest_model = max(model_files, key=os.path.getmtime)
        print(f"找到本地模型: {latest_model}，正在加载...")
        
        try:
            # 加载模型
            model = custom_model.load_model(model_path=latest_model, env=env, config_path=config_path)
            print(f"成功加载模型，继续训练...")
            return model, False
        except Exception as e:
            print(f"加载模型失败: {e}")
            print("创建新模型...")
    else:
        print("未找到本地模型，创建新模型...")
    
    # 创建新模型
    model = custom_model.create_model(config_path=config_path, env=env)
    print("新模型创建成功")
    return model, True

def main():
    try:
        print("正在初始化训练环境...")
        # 创建日志目录
        log_dir = "./logs/"
        model_dir = "./model/"
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)
        
        # 设置日志
        logger = configure(log_dir, ["stdout", "csv", "tensorboard"])
        
        # 设置检查点回调
        checkpoint_callback = CheckpointCallback(
            save_freq=10000,  # 每10000步保存一次
            save_path=model_dir,
            name_prefix="model",
            save_replay_buffer=True,
            save_vecnormalize=True,
        )
        
        # 检查CUDA是否可用
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"使用GPU训练: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device("cpu")
            print("GPU不可用，使用CPU训练")
        
        # 加载配置
        with open('./config/algs.yaml', 'r', encoding='utf-8') as f:
            algs_config = yaml.safe_load(f)
            
        # 更新设备配置
        algs_config['training']['device'] = str(device)
            
        # 创建训练环境
        print("正在创建训练环境...")
        try:
            base_env = TrainEnv(config_path='./config/envs.yaml')
            print("训练环境创建成功")
        except Exception as env_error:
            print(f"创建训练环境失败: {env_error}")
            traceback.print_exc()
            return
            
        print("创建Monitor包装环境...")
        env = Monitor(base_env)
        
        # 加载或创建模型
        print("正在检查本地模型...")
        try:
            model, is_new_model = load_or_create_model(
                env=env, 
                config_path='./config/algs.yaml', 
                model_dir=model_dir
            )
        except Exception as model_error:
            print(f"模型加载/创建失败: {model_error}")
            traceback.print_exc()
            return
            
        model.set_logger(logger)
        
        # 开始训练
        print("开始训练过程...")
        total_timesteps = 500000  # 总训练步数
        # 可以视情况分段训练
        try:
            model.learn(
                total_timesteps=total_timesteps,
                progress_bar=True,
                reset_num_timesteps=is_new_model,  # 如果是新模型则重置步数
                log_interval=1,
                callback=checkpoint_callback
            )
        except Exception as train_error:
            print(f"训练过程出错: {train_error}")
            traceback.print_exc()
            return
        
        # 保存最终模型
        final_model_path = os.path.join(model_dir, "final_model")
        model.save(final_model_path)
        print(f"训练完成，最终模型已保存至 {final_model_path}")
        
        # 可视化训练结果
        print("正在生成训练结果可视化...")
        try:
            df = pd.read_csv(os.path.join(log_dir, "progress.csv"))
            # 绘制奖励曲线
            plt.figure(figsize=(10, 6))
            plt.plot(df.index, df['rollout/ep_rew_mean'])
            plt.title('平均回合奖励')
            plt.xlabel('迭代次数')
            plt.ylabel('奖励')
            plt.savefig(os.path.join(log_dir, 'reward_curve.png'))
            plt.close()
        except Exception as vis_error:
            print(f"可视化生成失败: {vis_error}")
    
    except Exception as e:
        print(f"训练过程出错: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
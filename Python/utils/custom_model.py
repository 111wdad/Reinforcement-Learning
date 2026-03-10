from agent.transformer_option_agent import (
    CustomTransformerPolicy,
    CustomTransformerTD3Policy,
    CustomTransformerSACPolicy
)
import yaml
import torch
from stable_baselines3 import PPO, TD3, SAC
from stable_baselines3.ppo import MlpPolicy as PPOMlpPolicy
from stable_baselines3.td3 import MlpPolicy as TD3MlpPolicy
from stable_baselines3.sac import MlpPolicy as SACMlpPolicy


def create_model(config_path, env):
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        agent_cfg = config['agent']
        train_cfg = config['training']
        algorithm = train_cfg['algorithm'].lower()
        
        # 检查CUDA是否可用
        if 'device' in train_cfg and train_cfg['device'] == 'cuda':
            if torch.cuda.is_available():
                device = torch.device("cuda")
                print(f"使用CUDA: {torch.cuda.get_device_name(0)}")
            else:
                device = torch.device("cpu")
                print("CUDA不可用，使用CPU")
                train_cfg['device'] = 'cpu'
        else:
            device = torch.device("cpu")
            print("使用CPU")
            train_cfg['device'] = 'cpu'
            
        # 策略类映射表
        policy_map = {
            'ppo': (PPO, {'transformer': CustomTransformerPolicy, 'mlp': PPOMlpPolicy}),
            'td3': (TD3, {'transformer': CustomTransformerTD3Policy, 'mlp': TD3MlpPolicy}),
            'sac': (SAC, {'transformer': CustomTransformerSACPolicy, 'mlp': SACMlpPolicy})
        }

        if algorithm not in policy_map:
            raise ValueError(f"不支持的算法: {algorithm}, 请从 {list(policy_map.keys())} 中选择")

        model_class, policy_dict = policy_map[algorithm]
        policy_type = agent_cfg.get('policy_type', 'mlp')  # 默认为mlp

        # 获取策略类
        policy = policy_dict.get(policy_type)
        if not policy:
            raise ValueError(f"不支持的策略类型: {policy_type} (算法: {algorithm})")

        # 配置策略参数
        policy_kwargs = {}
        if policy_type == 'transformer':
            policy_kwargs = {
                'features_extractor_kwargs': {
                    'features_dim': agent_cfg.get('features', 256),
                    'nhead': agent_cfg.get('nheads', 8),
                    'num_layers': agent_cfg.get('num_layers', 3)
                }
            }
        else:
            # 配置MLP网络结构
            if algorithm == 'sac' or algorithm == 'td3':
                policy_kwargs = {
                    'net_arch': {
                        'pi': agent_cfg.get('pi_arch', [256, 256]),
                        'qf': agent_cfg.get('qf_arch', [256, 256])
                    }
                }
            else:  # PPO
                policy_kwargs = {
                    'net_arch': {
                        'pi': agent_cfg.get('pi_arch', [256, 256]),
                        'vf': agent_cfg.get('qf_arch', [256, 256])
                    }
                }

        # 算法特定参数
        alg_specific_params = {}
        if algorithm == 'sac':
            alg_specific_params = {
                'tau': agent_cfg.get('tau', 0.005),
                'ent_coef': agent_cfg.get('ent_coef', 'auto'),
                'target_entropy': 'auto',
                'learning_rate': train_cfg.get('learning_rate', 0.0003),
                'gamma': train_cfg.get('gamma', 0.99),
                'batch_size': train_cfg.get('batch_size', 256),
                'buffer_size': train_cfg.get('buffer_size', 1000000),
                'learning_starts': train_cfg.get('learning_starts', 1000),
                'train_freq': train_cfg.get('train_freq', 1),
                'gradient_steps': train_cfg.get('gradient_steps', 1),
                'target_update_interval': train_cfg.get('target_update_interval', 1)
            }
        elif algorithm == 'td3':
            alg_specific_params = {
                'learning_rate': train_cfg.get('learning_rate', 0.0003),
                'gamma': train_cfg.get('gamma', 0.99),
                'batch_size': train_cfg.get('batch_size', 100),
                'buffer_size': train_cfg.get('buffer_size', 1000000)
            }
        else:  # PPO
            alg_specific_params = {
                'learning_rate': train_cfg.get('learning_rate', 0.0003),
                'gamma': train_cfg.get('gamma', 0.99),
                'n_steps': train_cfg.get('n_steps', 2048),
                'batch_size': train_cfg.get('batch_size', 64)
            }
        
        print(f"使用算法: {algorithm}, 策略类型: {policy_type}")
        print(f"网络结构: {policy_kwargs}")
            
        model = model_class(
            policy,
            env,
            verbose=1,
            policy_kwargs=policy_kwargs,
            device=train_cfg['device'],
            **alg_specific_params
        )
        
        print("模型创建成功")
        return model
        
    except Exception as e:
        print(f"创建模型时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        raise
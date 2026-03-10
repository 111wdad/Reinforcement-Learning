import numpy as np


def marshal_action(action):
    """
    将Agent输出的动作转换为实际控制量
    
    控制量:
    1. 油门 throttle [0, 1]
    2. 俯仰角 pitch [-1, 1]
    3. 翻滚角 roll [-1, 1]
    4. 偏航角 yaw [-1, 1]
    
    参数:
    action: Agent输出的动作，范围[-1, 1]
    
    返回:
    控制量，范围与参数相同
    """
    try:
        # 确保输入为numpy数组
        if not isinstance(action, np.ndarray):
            action = np.array(action)
        
        # 确保动作在范围内
        action = np.clip(action, -1.0, 1.0)
        
        # 控制处理
        # 油门范围需要从[-1,1]映射到[0,1]
        throttle = (action[0] + 1.0) / 2.0  # 映射到[0,1]
        
        # 其他控制量保持原范围
        pitch = action[1]  # 俯仰角
        roll = action[2]   # 翻滚角
        yaw = action[3]    # 偏航角
        
        # 组合控制量
        real_action = np.array([throttle, pitch, roll, yaw], dtype=np.float64)
        
        return real_action
        
    except Exception as e:
        print(f"动作处理错误: {str(e)}")
        # 出错时返回随机动作（用于测试）
        return np.array([np.random.uniform(0,1), np.random.uniform(-1,1), np.random.uniform(-1,1), np.random.uniform(-1,1)], dtype=np.float64)
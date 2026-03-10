import numpy as np


# This is the observation processing function. Remember to modify the declarations in trainenv.py correspondingly.
def marshal_observation(my_state, enemy_state):
    """
    处理观察状态，将原始状态转换为Agent可用的归一化状态
    
    参数:
    my_state: 我方战机状态 [x, y, z, vx, vy, vz, phi, theta, psi, p, q, r]
    enemy_state: 敌方靶机状态 [x, y, z, vx, vy, vz, phi, theta, psi, p, q, r]
    
    返回:
    agent_state: 归一化后的状态，包含15个关键特征
    """
    try:
        # 初始化观察空间
        agent_state = np.zeros(shape=[15], dtype=np.float64)
        
        # 1-3: 计算相对位置
        rel_pos = enemy_state[0:3] - my_state[0:3]
        
        # 4: 计算距离
        distance = np.linalg.norm(rel_pos)
        
        # 5-7: 计算相对位置的单位向量（方向）
        direction = rel_pos / (distance + 1e-8)  # 避免除零错误
        
        # 8-10: 我方战机速度
        velocity = my_state[3:6]
        velocity_mag = np.linalg.norm(velocity)
        
        # 11-13: 我方战机姿态角
        attitude = my_state[6:9]
        
        # 14: 我方速度大小
        speed = velocity_mag
        
        # 15: 位置高度
        height = my_state[2]
        
        # 归一化处理
        # 相对位置归一化 (除以较大值来归一化)
        normalized_rel_pos = rel_pos / 150.0  # 假设场景大小为150x150x150
        agent_state[0:3] = normalized_rel_pos
        
        # 距离归一化
        agent_state[3] = distance / 200.0  # 归一化距离
        
        # 方向已经是单位向量，不需要归一化
        agent_state[4:7] = direction
        
        # 速度归一化
        agent_state[7:10] = velocity / 20.0  # 假设最大速度为20
        
        # 姿态角归一化 (姿态角原本就在[-pi, pi]范围内，除以pi归一化到[-1,1])
        agent_state[10:13] = attitude / np.pi
        
        # 速度大小归一化
        agent_state[13] = speed / 20.0
        
        # 高度归一化
        agent_state[14] = height / 50.0  # 假设最大高度为50
        
        # 确保所有数值都在合理范围内
        agent_state = np.clip(agent_state, -1.0, 1.0)
        
        return agent_state
        
    except Exception as e:
        print(f"观察状态处理错误: {str(e)}")
        # 出错时返回零数组
        return np.zeros(shape=[15], dtype=np.float64)
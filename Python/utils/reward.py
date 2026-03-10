import numpy as np

# This is the reward calculation function. We provide current state and previous state for you.
def calculate_reward(prev_my_state, prev_enemy_state, my_state, enemy_state):
    """
    计算奖励函数，引导Agent学习击落靶机的策略
    
    奖励组成 (优化版):
    1. 血量差值奖励：血量差增加获得高奖励（主要目标）
    2. 距离奖励：接近目标获得正奖励
    3. 攻击姿态奖励：机头朝向目标获得正奖励
    4. 边界惩罚：超出边界获得负奖励
    5. 碰撞惩罚：与地面碰撞获得负奖励
    6. 速度奖励：保持合适速度获得奖励
    7. 稳定性奖励：减少剧烈动作获得奖励
    
    状态变量结构:
    - 位置坐标: indices 0-2 (x, y, z)
    - 欧拉角: indices 3-5 (φ, θ, ψ) - 翻滚(roll)、俯仰(pitch)、偏航(yaw)
    - 线速度: indices 6-8 (u, v, w)
    - 角速度: indices 9-11 (ω, β, η)
    - 血量: index 12
    """
    try:
        # 计算当前和前一时刻的血量差
        prev_health_diff = prev_my_state[12] - prev_enemy_state[12]
        current_health_diff = my_state[12] - enemy_state[12]
        health_diff_change = current_health_diff - prev_health_diff
        
        # 计算当前和前一时刻与靶机的距离
        prev_distance = np.linalg.norm(prev_enemy_state[0:3] - prev_my_state[0:3])
        current_distance = np.linalg.norm(enemy_state[0:3] - my_state[0:3])
        
        # 初始化奖励
        reward = 0.0
        
        # 1. 血量差值奖励（主要奖励）- 优化版
        health_reward = 0.0
        
        # 基础血量差值奖励：血量差值增加时给予奖励
        if health_diff_change > 0:
            # 血量差增加（我方优势增加或劣势减少）
            health_reward = health_diff_change * 35.0  # 增加权重
            print(f"血量差增加奖励: {health_reward:.3f}, 血量差变化: {health_diff_change:.3f}")
        elif health_diff_change < 0:
            # 血量差减少（我方优势减少或劣势增加）
            health_reward = health_diff_change * 15.0  # 增加惩罚
            print(f"血量差减少惩罚: {health_reward:.3f}, 血量差变化: {health_diff_change:.3f}")
        
        # 放大小血量差的奖励/惩罚，鼓励智能体更关注小的血量变化
        if 0 < abs(health_diff_change) < 0.2:
            health_reward *= 2.0  # 更强的放大效果
            print(f"小血量差放大: 奖励调整为 {health_reward:.3f}")
            
        # 当前血量差为正时给予额外奖励，使用平方根函数提供更平滑的奖励曲线
        if current_health_diff > 0:
            bonus = 3.0 * np.sqrt(current_health_diff)  # 使用平方根函数
            health_reward += bonus
            print(f"当前血量优势奖励: {bonus:.3f}")
        
        # 2. 距离奖励（辅助奖励）- 优化版
        distance_diff = prev_distance - current_distance
        distance_reward = 0.0
        
        # 使用平滑的距离奖励函数
        if distance_diff > 0:  # 距离减小时给予奖励
            distance_reward = distance_diff * 2  # 降低系数，减少波动
            if current_distance < 50:  # 接近敌机时增加奖励
                distance_reward *= 2.5  # 降低倍数，使曲线更平滑
        else:
            distance_reward = distance_diff * 2 # 降低惩罚系数
            if current_distance < 20:  # 接近敌机时减小惩罚
                distance_reward *= 1.5  # 降低倍数
                
        # 添加平滑处理：限制单次奖励的最大值
        distance_reward = np.clip(distance_reward, -1.0, 1.0)
        
        if abs(distance_reward) > 0.1:
            print(f"距离奖励: {distance_reward:.3f}, 距离变化: {distance_diff:.3f}")
        
        # 3. 攻击姿态奖励 - 平滑版
        attack_pose_reward = 0.0
        
        # 计算指向靶机的方向向量
        to_enemy_vector = enemy_state[0:3] - my_state[0:3]
        to_enemy_dir = to_enemy_vector / (np.linalg.norm(to_enemy_vector) + 1e-8)
        
        # 计算机头方向向量
        nose_dir = np.array([
            np.cos(my_state[4]) * np.cos(my_state[5]),
            np.cos(my_state[4]) * np.sin(my_state[5]),
            -np.sin(my_state[4])
        ])
        
        # 计算机头方向与指向敌机方向的夹角余弦值
        nose_alignment = np.dot(nose_dir, to_enemy_dir)
        
        # 当距离越近，攻击姿态奖励越重要，使用非线性函数增强近距离效果
        alignment_weight = 10 * np.exp(-current_distance/100)
        attack_pose_reward = nose_alignment * alignment_weight
        
        # 对准目标时指数增强奖励
        if nose_alignment > 0.7:  # 降低阈值，更早给予奖励
            attack_pose_reward *= (10.0 + nose_alignment)  # 非线性增强
            print(f"攻击姿态奖励: {attack_pose_reward:.3f}, 对准度: {nose_alignment:.3f}")
        
        
    
        # 6. 新增：速度奖励 - 保持合适的速度范围
        speed_reward = 0.0
        current_speed = np.linalg.norm(my_state[6:9])
        optimal_speed = 15.0  # 最佳速度
        
        # 使用高斯函数，在最佳速度附近给予最高奖励
        speed_diff = current_speed - optimal_speed
        speed_reward = 0.5 * np.exp(-(speed_diff**2) / 50)
        
        if abs(speed_reward) > 0.1:
            print(f"速度奖励: {speed_reward:.3f}, 当前速度: {current_speed:.3f}")
            
        # 7. 新增：稳定性奖励 - 减少剧烈动作
        stability_reward = 0.0
        angular_velocity = np.linalg.norm(my_state[9:12])
        
        # 角速度过大时给予惩罚
        if angular_velocity > 1.0:
            stability_reward = -0.3 * (angular_velocity - 1.0)
            print(f"稳定性惩罚: {stability_reward:.3f}, 角速度: {angular_velocity:.3f}")
        
        # 计算总奖励 - 调整各部分权重，确保血量差有最高权重
        reward = health_reward * 2.5 + \
                 distance_reward * 0.4 + \
                 attack_pose_reward * 1.3 + \
                 speed_reward * 0.3 + \
                 stability_reward * 0.2
        
        # 记录总奖励
        if abs(reward) > 1.0:
            print(f"总奖励: {reward:.3f}")
            
        return reward
        
    except Exception as e:
        print(f"奖励计算错误: {str(e)}")
        return 0

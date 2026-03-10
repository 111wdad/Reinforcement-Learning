import numpy as np
import random


def generate_initial_state():
    """
    初始化战机和靶机的位置，确保初始距离不小于1000m（即100个单位）
    
    my_state结构: [x, y, z, vx, vy, vz, phi, theta, psi, p, q, r]
    战机状态变量：位置(x,y,z)、速度(vx,vy,vz)、姿态角(phi,theta,psi)、角速度(p,q,r)
    
    enemy_state结构与my_state相同
    """
    try:
        # 初始化我方战机的位置和状态
        my_initial_state = np.zeros(12, dtype=np.float64)
        # 设置初始位置在原点
        my_initial_state[0:3] = [0.0, 0.0, 20.0]  # x,y,z - 初始高度20单位
        # 设置初始速度
        my_initial_state[3:6] = [0.0, 0.0, 0.0]  # vx,vy,vz - 初始速度沿x方向
        # 设置初始姿态角
        my_initial_state[6:9] = [0.0, 0.0, 0.0]  # phi,theta,psi
        # 设置初始角速度
        my_initial_state[9:12] = [0.0, 0.0, 0.0]  # p,q,r
        
        # 初始化敌方靶机的位置和状态
        enemy_initial_state = np.zeros(12, dtype=np.float64)
        
        # 敌方靶机位置 - 确保距离大于100个单位（即1000米）
        distance = 0
        while distance < 100:
            # 在远处随机放置靶机
            x = random.uniform(100, 150)
            y = random.uniform(-20, 20)
            z = random.uniform(15, 25)
            
            # 计算距离
            distance = np.sqrt((x - my_initial_state[0])**2 + 
                              (y - my_initial_state[1])**2 + 
                              (z - my_initial_state[2])**2)
        
        enemy_initial_state[0:3] = [100, 0, 20]  # x,y,z
        # 靶机是固定的，速度为0
        enemy_initial_state[3:6] = [0.0, 0.0, 0.0]  # vx,vy,vz
        enemy_initial_state[6:9] = [0.0, 0.0, 0.0]  # phi,theta,psi
        enemy_initial_state[9:12] = [0.0, 0.0, 0.0]  # p,q,r
        
        # 合并状态
        initial_state = np.append(my_initial_state, enemy_initial_state)
        print(f"初始化成功: 战机位置 {my_initial_state[0:3]}, 靶机位置 {enemy_initial_state[0:3]}, 距离 {distance*10}米")
        return initial_state
        
    except Exception as e:
        print(f"初始化状态时出错: {str(e)}")
        # 出错时返回默认值
        my_initial_state = np.zeros(12)
        enemy_initial_state = np.ones(12) * 100  # 默认放在远处
        initial_state = np.append(my_initial_state, enemy_initial_state)
        return initial_state
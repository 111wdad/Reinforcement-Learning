
import numpy as np

def check_truncation(my_state, enemy_state):
    """
    检查是否应该截断当前回合
    
    截断条件:
    1. 飞机超出边界
    2. 飞机与地面碰撞
    3. 回合时间过长（在其他地方控制）
    
    参数:
    my_state: 我方战机状态
    enemy_state: 敌方靶机状态
    
    返回:
    truncated: 布尔值，表示是否截断
    """
    try:
        truncated = False
        
        # 检查是否超出边界
        # 假设场景边界为 ±200 单位
      
        
        # 检查是否与地面碰撞
        if my_state[2] < 0:  # z < 0表示撞地
            print("战机与地面碰撞，回合截断")
            truncated = True
        
        # 距离过远也截断
        distance = np.linalg.norm(enemy_state[0:3] - my_state[0:3])
        if distance > 300:  # 距离超过300单位也截断
            print(f"战机远离目标 (距离: {distance})，回合截断")
            truncated = True
        
        return truncated
        
    except Exception as e:
        print(f"截断检查错误: {str(e)}")
        return False
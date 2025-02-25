import numpy as np
from envs.utils.math_utils import reach_goal,cal_distance

class LLM_reward():
    def __init__(self,robots) -> None:
        self.robots = robots
        self.collision_flag = False
    
# INSERT EUREKA REWARD HERE


import numpy as np
from envs.utils.math_utils import reach_goal

class LLM_reward():
    def __init__(self,robots) -> None:
        self.robots = robots
        self.collision_flag = False
    
    def get_reward(self):
        rewards = []
        action_stability_reward = 0
        reward_info = {
            'avoid_obstacle': 0,
            'maintain_formation': 0,
            'reach_destination': 0,
            'action_stability': 0
        }
        for robot in self.robots:
            if robot.collision == True:
                self.collision_flag = True
                break
        for robot in self.robots:
            r_avoid = 0
            r_goal = 0
            r_nav = 0
            robot.goal_flag = False

            #collision
            if robot.collision == True:
                r_avoid = -60
            else:
                if robot.dmin < robot.discomfort_dist * 2:
                    r_avoid = -np.exp(-robot.dmin/3)
            
            #formation
            r_formation = -np.sqrt(robot.for_feature)

            #navigation
            r_nav += (robot.pre_dist2goal - robot.dist2goal) * 5
            if r_nav > 0 and robot.collision:
                r_nav = 0
            
            if reach_goal(robot):
                robot.goal_flag = True
                if robot.v != 0:
                    r_goal += 5
            if self.collision_flag:
                r_goal = 0

            discount_formation = 60
            discount_avoid = 100
            discount_nav = 20
            discount_goal = 100

            reward =  discount_formation * r_formation + discount_avoid * r_avoid + discount_nav * r_nav + discount_goal * r_goal
            if robot.goal_flag:
                reward = 0
            # print(discount_formation * r_formation, discount_nav * r_nav, discount_avoid * r_avoid, reward, robot.id)
            rewards.append([reward])
            reward_info['avoid_obstacle'] += discount_avoid * r_avoid
            reward_info['maintain_formation'] += discount_formation * r_formation
            reward_info['reach_destination'] += discount_nav * r_nav + discount_goal * r_goal
            reward_info['action_stability'] += action_stability_reward
        return rewards,reward_info
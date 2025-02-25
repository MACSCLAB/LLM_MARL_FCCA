import numpy as np
from envs.utils.math_utils import reach_goal,cal_distance

class LLM_reward():
    def __init__(self,robots) -> None:
        self.robots = robots
        self.collision_flag = False
    
    def get_reward(self):
        rewards = []
        reward_info = {
            'total': 0.0,
            'goal_reached': 0.0,
            'distance_moved': 0.0,
            'collision_penalty': 0.0,
            'formation_maintenance': 0.0,
            'velocity_stability': 0.0,
            'time_efficiency': 0.0,
            'proximity_penalty': 0.0
        }

        # Determine if any robot has collided this episode
        any_collision = any(robot.collision for robot in self.robots)

        for robot in self.robots:
            # Reward for reaching the goal (only if no collisions)
            goal_reached_reward = 200.0 if robot.dist2goal <= robot.radius and not any_collision else 0.0
            
            # Penalty for collision
            collision_penalty = -100.0 if robot.collision else 0.0

            # Reward based on progress towards the goal
            distance_moved_reward = (robot.pre_dist2goal - robot.dist2goal) if robot.pre_dist2goal is not None else 0.0

            # Reward for maintaining formation
            formation_maintenance_reward = -robot.for_feature * 10 if robot.for_feature is not None else 0.0

            # Reward for sustaining a stable velocity
            velocity_stability_reward = -abs(robot.v - robot.pre_v) if robot.pre_v is not None else 0.0

            # Reward for efficient time usage (encouraging faster movement)
            time_efficiency_reward = distance_moved_reward * robot.v if robot.pre_v is not None else 0.0

            # Penalty for coming too close to obstacles or other robots
            proximity_penalty = -1.0 / (robot.dmin + 1e-6) if robot.dmin < robot.discomfort_dist else 0.0

            # Sum up all reward components for this robot
            total_reward = (
                goal_reached_reward +
                collision_penalty +
                distance_moved_reward +
                formation_maintenance_reward +
                velocity_stability_reward +
                time_efficiency_reward +
                proximity_penalty
            )

            rewards.append([total_reward])

            # Update the aggregated reward information
            reward_info['total'] += total_reward
            reward_info['goal_reached'] += goal_reached_reward
            reward_info['distance_moved'] += distance_moved_reward
            reward_info['collision_penalty'] += collision_penalty
            reward_info['formation_maintenance'] += formation_maintenance_reward
            reward_info['velocity_stability'] += velocity_stability_reward
            reward_info['time_efficiency'] += time_efficiency_reward
            reward_info['proximity_penalty'] += proximity_penalty

        return rewards, reward_info



import logging
import numpy as np
from envs.utils.human import generate_human
from envs.utils.robot import generate_robot
from envs.utils.info import *
from envs.utils.math_utils import cal_distance,reach_goal,get_weight
from envs.reward.original_reward import LLM_reward
# from envs.reward.LLM_reward import LLM_reward
import time

class EnvCore(object):
    def __init__(self, args):
        self.args = args
        self.time_step = args.time_step
        self.time_limit = args.episode_length * self.time_step
        self.method = args.method
        self.humans = None
        self.robots = None
        self.global_time = None
        self.human_times = None
        self.robot_num = args.num_robots
        self.human_num = args.num_humans
        self.att_agents = args.num_attention_agents
        self.robot_obs_dim = args.robot_obs_dim
        self.human_obs_dim = args.human_obs_dim
        self.obs_dim = max(self.robot_obs_dim, self.human_obs_dim)
        self.vel_action_dim = args.vel_action_dim
        self.dir_action_dim = args.dir_action_dim
        self.discomfort_dist = args.dcf_dist
        self.base_v = args.base_v
        # simulation configuration
        self.config = None
        self.randomize_attributes = args.randomize_attributes
        self.train_val_sim = args.human_action
        self.square_width = args.square_width
        self.circle_radius = args.circle_radius
        # for visualization
        self.total_obs = None

        # for formulate
        self.observation_states = None
        self.attention_weights = None
        self.for_edge = args.for_edge
        self.for_feature = None

        self.collision_flag = None
        self.rewards_component = None

    def reset(self):
        """
        # self.agent_num设定为2个智能体时，返回值为一个list，每个list里面为一个shape = (self.obs_dim, )的观测数据
        # When self.agent_num is set to 2 agents, the return value is a list, each list contains a shape = (self.obs_dim, ) observation data
        """
        self.robots = generate_robot(self.args)
        self.humans = generate_human(self.args, self.robots)
        self.total_obs = []
        self.global_time = 0
        self.collision_flag = False
        self.rewards_component = LLM_reward(self.robots)

        # 正n边型编队
        W = (np.ones((self.robot_num, self.robot_num)) - np.eye(self.robot_num)) * (self.for_edge)**2
        # 一字型编队
        # W = np.array([[0,4,16], 
        #               [4,0,4],
        #               [16,4,0]])

        D = np.diag(sum(W))
        L = D - W
        D_sys = np.diag(pow(sum(W),-1/2))  #sysmmetric normolize
        L_des = D_sys @ L @ D_sys
        for robot in self.robots:
            robot.L_des = L_des
        assert np.round(L_des[0][0]) == 1,'L compute error!'

        #设置时间步长
        for agent in self.robots + self.humans :
            agent.time_step = self.time_step
        
        obs = self.get_obs()
        share_obs = self.get_share_obs()
        self.total_obs.append(share_obs)

        if self.method == 'ppo':
            return obs, share_obs
        elif self.method == 'orca' or self.method == 'apf':
            obs_orca = []
            for agent in self.robots + self.humans:
                obs_orca.append(np.array([agent.px, agent.py, agent.vx, agent.vy, agent.gx, agent.gy]))
            return obs_orca

    def step(self):
        human_obs = []
        human_actions = []
        self.global_time += self.time_step

        # human step
        for human in self.humans:
             # observation for humans is always coordinates
            ob = [other_human.get_observable_state() for other_human in self.humans if other_human != human]
            #这里得到的ob是JointState中的human_states，即其它human的状态
            human_action = human.act(ob)
            # print(human_action[1])
            human_actions.append(human_action)    #human到终点后位置不再更新
            human.theta = np.arctan2(human.vy,human.vx)
            human_obs.append(np.array([human.px, human.py, human.vx, human.vy, human.theta]))
        all_success = [robot.success for robot in self.robots]
        for i, human_action in enumerate(human_actions):
            if not all(all_success):
                self.humans[i].step(human_action)

        obs = self.get_obs()
        share_obs = self.get_share_obs()
        dones = self.get_dones()
        rewards,reward_info = self.rewards_component.get_reward()
        rewards = np.array(rewards)
        info = []
        self.total_obs.append(share_obs)

        if self.method == 'ppo':
            return [obs, share_obs, rewards, dones, reward_info]
        
        elif self.method == 'orca' or self.method == 'apf':
            obs_orca = []
            for agent in self.robots + self.humans:
                obs_orca.append(np.array([agent.px, agent.py, agent.vx, agent.vy, agent.gx, agent.gy]))
            return  [obs_orca]

    def get_obs(self):
        """
        return: np.array([robot_num, 1+human_num, obs_dim])
        """
        W = np.zeros((self.robot_num, self.robot_num))
        for i, robot in enumerate(self.robots):
            robot.vx_formation = 0
            robot.vy_formation = 0
            #formation detect
            for r in self.robots:
                if robot != r:
                    robot.vx_formation -= (robot.px - r.px - (robot.for_std[0] - r.for_std[0]))
                    robot.vy_formation -= (robot.py - r.py - (robot.for_std[1] - r.for_std[1]))

            for j, car in enumerate(self.robots):
                W[i][j] = get_weight(robot.px, robot.py, car.px, car.py)

        assert W[-1][-2] != 0,'W compute error!'
        D = np.diag(sum(W))  #本征不变性
        L = D - W
        D_sys = np.diag(pow(sum(W),-1/2))  #sysmmetric normolize
        L_hat = D_sys @ L @ D_sys #缩放不变性
        for robot in self.robots:
            robot.for_feature = np.trace(np.transpose((L_hat - robot.L_des)) @ (L_hat - robot.L_des))

        #navigation
        for robot in self.robots:
            robot.pre_dist2goal = robot.dist2goal
            robot.dist2goal = cal_distance(robot.px,robot.py,robot.gx,robot.gy)

        # collision
        for robot in self.robots:
            #initialize some flags of robots
            robot.dmin = float('inf')
            robot.collision = False
            for agent in self.robots + self.humans:
                if robot != agent:
                    d = cal_distance(robot.px, robot.py, agent.px, agent.py)
                    d -= robot.radius + agent.radius
                    if d < robot.dmin:
                        robot.dmin = d
                
                # if robot.dmin - robot.radius - agent.radius < 0 or robot.px < 0 or robot.py < 0: # with boundary
                if robot.dmin < 0:   # without boundary
                    robot.collision = True
                    robot.success = False
                    self.collision_flag = True

        # format obs for network
        all_obs = []
        for robot in self.robots:
            obs = np.zeros(((1 + max(self.human_num, self.att_agents)), self.obs_dim))
            obs[0,:self.robot_obs_dim] = np.array([robot.px,
                                                   robot.py,
                                                   robot.gx - robot.px,
                                                   robot.gy - robot.py,
                                                   robot.v,
                                                   robot.theta,
                                                   robot.for_feature,
                                                   robot.vx_formation,
                                                   robot.vy_formation])
            for human in self.humans:
                human.dist2rob = cal_distance(robot.px, robot.py, human.px, human.py)
            sorted_humans = self.humans.copy()
            sorted_humans = sorted(sorted_humans, key=lambda x:x.dist2rob, reverse=True)   #行人距离机器人的距离由远到近进行排序
            assert sorted_humans[-1].dist2rob <= sorted_humans[-2].dist2rob, 'sort error!'
            for j,human in enumerate(sorted_humans[:self.human_num]):
                obs[j + 1,:self.human_obs_dim] = \
                np.array([human.px, human.py, human.vx, human.vy, np.arctan2(human.py,human.px)])
            all_obs.append(obs)
        all_obs = np.array(all_obs)
        return all_obs
    
    def get_share_obs(self):
        """
        return: np.array([robot_num + max(human_num,att_agent), obs_dim])
        """
        share_obs = np.zeros(((self.robot_num + max(self.human_num, self.att_agents)), self.obs_dim))
        for i,robot in enumerate(self.robots):
            share_obs[i, :self.robot_obs_dim] = np.array([robot.px,
                                                   robot.py,
                                                   robot.gx,
                                                   robot.gy,
                                                   robot.v,
                                                   robot.theta,
                                                   robot.for_feature,
                                                   robot.vx_formation,
                                                   robot.vy_formation])
        for j,human in enumerate(self.humans):
            share_obs[j + self.robot_num, :self.human_obs_dim] = np.array([human.px, 
                                                                           human.py, 
                                                                           human.vx, 
                                                                           human.vy, 
                                                                           np.arctan2(human.py,human.px)])
        return share_obs
    
    def get_dones(self):
        dones = []
        for robot in self.robots:
            done = False
            if robot.collision == True: # 只有全部机器人碰撞才reset
            # if self.collision_flag == True: # 单个机器人碰撞就reset
                done = True
            if reach_goal(robot):
                done = True
            if self.global_time >= self.time_limit:
                done = True
            dones.append(done)

        return dones
    
    # def get_info(self,agent):
    #     info = Nothing()
    #     if agent.collision == True:
    #         info = Collision()

    #     if agent.dmin < agent.discomfort_dist:
    #         info = Danger()

    #     if self.global_time >= self.time_limit:
    #         info = Timeout()
        
    #     if reach_goal(agent):
    #         info = ReachGoal()

    #     return info
    
    

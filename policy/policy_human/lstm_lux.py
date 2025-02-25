import logging
import torch
import torch.nn as nn
import itertools
from torch.distributions.categorical import Categorical
from envs.utils.action import *
import numpy as np


import logging
from policy.policy_human.cadrl import mlp
from policy.policy_human.cadrl import CADRL

def build_action_space():
    speeds = [(np.exp((i + 1) / 5) - 1) / (np.e - 1) * 1 for i in range(5)]
    rotations = np.linspace(0, 2*np.pi, 16, endpoint=False)  # 均分
    action_sapce = [ActionXY(0, 0)]
    for rotation, speed in itertools.product(rotations, speeds):
        action_sapce.append(ActionXY(speed * np.cos(rotation), speed * np.sin(rotation)))
    return action_sapce

class ValueNetwork1(nn.Module):
    def __init__(self, input_dim, robot1_state_dim, robot2_state_dim, mlp_dims, lstm_hidden_dim):
        super().__init__()

        self.robot1_state_dim = robot1_state_dim
        self.robot2_state_dim = robot2_state_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.mlp_robot1 = mlp(robot1_state_dim + lstm_hidden_dim, mlp_dims)
        self.mlp_robot2= mlp(robot2_state_dim + lstm_hidden_dim, mlp_dims)
        # batch_first affect output
        self.lstm = nn.LSTM(input_dim, lstm_hidden_dim, batch_first=True)
    def forward(self, state, no):

        size =state.shape
        dim = state.dim()
        if dim == 2:
            state = state.unsqueeze(0) 
            size = state.shape
        if no == 0:
            robot1_state = state[:, 0, :self.robot1_state_dim]
            robot2_state = state[:, 0, 13:13 + self.robot2_state_dim]
            # logging.info('state1:%s.', robot1_state)
            # logging.info('state2:%s.', robot2_state)
            robot1_state_lstm = state[:, :, :13]
            robot2_state_lstm = state[:, :, 13:]

            state_lstm1 = state#robot1以及robot2以及行人全状态
            state_lstm2 = torch.cat([robot2_state_lstm, robot1_state_lstm], dim=2)
            # logging.info('state1:%s.', state_lstm1)
            # logging.info('state2:%s.', state_lstm2)
        else:
            #robot2_state = state[:, 0, :self.robot1_state_dim]
            #robot1_state = state[:, 0, 13:13 + self.robot2_state_dim]
            robot2_state = state[:, 0, :self.robot2_state_dim]
            robot1_state = state[:, 0, 13:13 + self.robot2_state_dim]
            robot2_state_lstm = state[:, :, :13]
            robot1_state_lstm = state[:, :, 13:]
            state_lstm2 = state
            state_lstm1 = torch.cat([robot1_state_lstm, robot2_state_lstm], dim=2)

        h0 = torch.zeros(1, size[0], self.lstm_hidden_dim)
        c0 = torch.zeros(1, size[0], self.lstm_hidden_dim)
        h1 = torch.zeros(1, size[0], self.lstm_hidden_dim)
        c1 = torch.zeros(1, size[0], self.lstm_hidden_dim)       

        output1, (hn1, cn1) = self.lstm(state_lstm1, (h0, c0))# (h0, c0)为LSTM 模型的初始隐藏状态和记忆状态
        output2, (hn2, cn2) = self.lstm(state_lstm2, (h1, c1))
        hn1 = hn1.squeeze(0)
        hn2 = hn2.squeeze(0)
        joint_state_with_robot1 = torch.cat([robot1_state, hn1], dim=1)
        joint_state_with_robot2 = torch.cat([robot2_state, hn2], dim=1)

        # robot1_action
        value_robot1 = self.mlp_robot1(joint_state_with_robot1)#将当前状态和LSTM记忆状态传入感知机得到价值输出

        value_robot2 = self.mlp_robot2(joint_state_with_robot2)

        return value_robot1, value_robot2  


class ValueNetwork2(nn.Module):
    def __init__(self, input_dim, robot1_state_dim, robot2_state_dim, mlp1_dims, mlp_dims, lstm_hidden_dim):
        super().__init__()  # 初始化父类的属性

        self.robot1_state_dim = robot1_state_dim
        self.robot2_state_dim = robot2_state_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.mlp1 = mlp(input_dim, mlp1_dims)
        self.mlp_robot1 = mlp(robot1_state_dim + lstm_hidden_dim, mlp_dims)
        self.mlp_robot2= mlp(robot2_state_dim + lstm_hidden_dim, mlp_dims)
        # batch_first affect output
        self.lstm = nn.LSTM(mlp1_dims[-1], lstm_hidden_dim, batch_first=True) ##!!!

    def forward(self, state):

        size =state.shape
        dim = state.dim()
        if dim == 2:
            state = state.unsqueeze(0) 
            size = state.shape

        robot1_state = state[:, 0, :self.robot1_state_dim]
        robot2_state = state[:, 0, 13:13 + self.robot2_state_dim]

        robot1_state = torch.reshape(robot1_state, (-1, size[2]))
        robot2_state = torch.reshape(robot2_state, (-1, size[2]))##!!!!!

        robot1_mlp1_output = self.mlp1(robot1_state)
        #robot1_mlp1_output = torch.reshape(robot1_mlp1_output, (size[0], size[1], -1))
        robot2_mlp1_output = self.mlp1(robot2_state)
        #robot2_mlp1_output = torch.reshape(robot2_mlp1_output, (size[0], size[1], -1))##!!!!!!


        h0 = torch.zeros(1, size[0], self.lstm_hidden_dim)
        c0 = torch.zeros(1, size[0], self.lstm_hidden_dim)

        output, (hn, cn) = self.lstm(state, (h0, c0))
        hn = hn.squeeze(0)
        joint_state = torch.cat([robot1_state, robot2_state, hn], dim=1)
        joint_state_with_robot1 = torch.cat([robot1_state, hn], dim=1)
        joint_state_with_robot2 = torch.cat([robot2_state, hn], dim=1)

        # robot1_action
        value_robot1 = self.mlp_robot1(joint_state_with_robot1)

        # robot1_value
        value_robot2 = self.mlp_robot2(joint_state_with_robot2)

        return value_robot1, value_robot2


class Lstm_RL_LUX(CADRL):
    def __init__(self):
        super().__init__()
        self.name = 'LSTM-RL'
        self.kinematics = 'holonomic'

        self.with_interaction_module = None
        self.interaction_module_dims = None

        self.robot1_state_dim = 6
        self.human_state_dim = 7
        self.robot2_state_dim = 6
        self.joint_state_dim = self.robot1_state_dim + self.robot2_state_dim + 2 * self.human_state_dim

    def configure(self, config):
        self.set_common_parameters(config)
        mlp_dims = [int(x) for x in config.get('lstm_rl', 'mlp2_dims').split(', ')]
        global_state_dim = config.getint('lstm_rl', 'global_state_dim')#lstm
        self.with_om = config.getboolean('lstm_rl', 'with_om')
        with_interaction_module = config.getboolean('lstm_rl', 'with_interaction_module')

        if with_interaction_module:
            mlp1_dims = [int(x) for x in config.get('lstm_rl', 'mlp1_dims').split(', ')]
            self.model = ValueNetwork2(self.input_dim(), self.robot1_state_dim, self.robot2_state_dim, mlp1_dims, mlp_dims, global_state_dim)
        else:
            self.model = ValueNetwork1(self.input_dim(), self.robot1_state_dim, self.robot2_state_dim, mlp_dims, global_state_dim)

        self.multiagent_training = config.getboolean('lstm_rl', 'multiagent_training')

        self.action_space = build_action_space()

        logging.info('Policy: {}LSTM-RL {} pairwise interaction module'.format(
            'OM-' if self.with_om else '', 'w/' if with_interaction_module else 'w/o'))


    def predict(self, state):
        def dist(human):
        #  sort human order by decreasing distance to the robot
            return np.linalg.norm(np.array(human.position) - np.array(state.robot1_state.position))
        def dist2(human):
        #  sort human order by decreasing distance to the robot
            return np.linalg.norm(np.array(human.position) - np.array(state.robot2_state.position))

        state_robot1 = JointState_2robot(state.robot1_state, state.robot2_state, state.human_states)
        # logging.info('state robot1: %s', state_robot1.self_state)
        
        '''reverse=True ,descending order以Hunman机器人和robot1机器人之间的距离作为依据，对状态进行降序排列'''
        state_robot1.human_states = sorted(state_robot1.human_states, key=dist, reverse=True)

        
        state_robot2 = JointState_2robot(state.robot2_state, state.robot1_state, state.human_states)
        '''reverse=True ,descending order以Hunman机器人和robot1机器人之间的距离作为依据，对状态进行降序排列'''
        state_robot2.human_states = sorted(state_robot2.human_states, key=dist2, reverse=True)  # reverse=True ,descending order

        # # state.human_states = sorted(state.human_states, key=dist,
        #                                    reverse=True)  # reverse=True ,descending order
        predict_robot2 = self.predict_2robot(state_robot2, state_robot1, 1)
        if self.phase == 'train':
            other_robot = self.robot2_state.self_state
        predict_robot1 = self.predict_2robot(state_robot1, state_robot2, 0)
        if self.phase == 'train':
            robot1 = self.robot1_state
        if self.phase == 'train':
            self.last_state = self.transform(robot1, other_robot)

        return predict_robot1, predict_robot2

    def load_model(self, model):
        self.model = model
    def reach_destination(self, state):
        dis = np.linalg.norm((state.self_state.py - state.self_state.gy, state.self_state.px - state.self_state.gx))
        if dis < state.self_state.radius:
            return True
        else:
            return False


    def predict_2robot(self, state, other_robot, no):
        if self.phase is None or self.device is None:
            raise AttributeError('Phase, device attributes have to be set!')
        if self.phase == 'train' and self.epsilon is None:
            raise AttributeError('Epsilon attribute has to be set in training phase')

        if self.reach_destination(state):
            return ActionXY(0, 0) if self.kinematics == 'holonomic' else ActionRot(0, 0)
        if self.action_space is None:
            self.build_action_space(state.self_state.v_pref)

        occupancy_maps = None
        probability = np.random.random()
        if self.phase == 'train' and probability < self.epsilon:
            max_action = self.action_space[np.random.choice(len(self.action_space))]
        else:
            if no == 0:
                self.robot1_action_values = list()
                max_value1 = float('-inf')
            else:
                self.robot2_action_values = list()
                max_value2 = float('-inf')

            max_action = None
            for action in self.action_space:
                next_self_state = self.propagate(state.self_state, action)

                #计算当前动作所产生的价值
                if self.query_env:#self-attention
                    if no == 0:
                        next_human_states, reward, done, info = self.env.onestep_lookahead(action, ActionXY(0, 0))
                    elif no == 1:
                        next_human_states, reward, done, info = self.env.onestep_lookahead(ActionXY(0, 0), action)
                else:
                    '''???'''
                    next_human_states = [self.propagate(human_state, ActionXY(human_state.vx, human_state.vy))
                                       for human_state in state.human_states]
                    reward = self.compute_reward(next_self_state, next_human_states)

                next_human_states = next_human_states.human_states
              
                batch_next_states = torch.cat([torch.Tensor([next_self_state + next_human_state]).to(self.device)
                                               for next_human_state in next_human_states], dim=0)

                batch_next_states_other= torch.cat([torch.Tensor([other_robot.self_state + next_human_state]).to(self.device)
                                               for next_human_state in next_human_states], dim=0)
                rotated_batch_input_robot = self.rotate(batch_next_states).unsqueeze(0)#增加一维用于批次处理
                rotated_batch_input_other = self.rotate(batch_next_states_other).unsqueeze(0)
                rotated_batch_input = torch.cat(([rotated_batch_input_robot, rotated_batch_input_other]), dim=2)
                if self.with_om:
                    if occupancy_maps is None:
                        occupancy_maps = self.build_occupancy_maps(next_human_states).unsqueeze(0)
                    rotated_batch_input = torch.cat([rotated_batch_input, occupancy_maps.to(self.device)], dim=2)
                # VALUE UPDATE
                if no == 0:
                    next_states_value_1, next_states_value_2 = self.model(rotated_batch_input, 0)
                    next_state_value = next_states_value_1.data.item()
                    value_1 = reward[no] + pow(self.gamma, self.time_step * state.self_state.v_pref) * next_state_value
                    # self.robot1_action_values.append(value_1)
                    if value_1 > max_value1:
                        max_value1 = value_1
                        max_action = action
                elif no == 1:
                    next_states_value_1, next_states_value_2 = self.model(rotated_batch_input, 1)
                    next_state_value = next_states_value_2.data.item()
                    value_2 = reward[no] + pow(self.gamma, self.time_step * state.self_state.v_pref) * next_state_value
                    self.robot2_action_values.append(value_2)
                    if value_2 > max_value2:
                        max_value2 = value_2
                        max_action = action
                if max_action is None:
                    raise ValueError('Value network is not well trained. ')

        if self.phase == 'train':
            if no == 0:
                self.robot1_state = state
            elif no == 1:
                self.robot2_state = state
        return max_action

    def fullstate_to_vector(self,full_state):
        vector = [full_state.px, full_state.py, full_state.vx, full_state.vy, full_state.radius, full_state.gx,
                  full_state.gy, full_state.v_pref, full_state.theta]
        return vector

#转换为当前机器人视角下状态信息共12维
    def rotate(self, state):

        # 'px', 'py', 'vx', 'vy', 'radius', 'gx', 'gy', 'v_pref', 'theta'
        # 0     1      2     3      4        5     6      7         8
        # 'px1', 'py1', 'vx1', 'vy1', 'radius1'
        #    9     10      11     12       13

        batch = state.shape[0]

        dx = (state[:, 5] - state[:, 0]).reshape((batch, -1))  
        dy = (state[:, 6] - state[:, 1]).reshape((batch, -1))
        rot = torch.atan2(state[:, 6] - state[:, 1], state[:, 5] - state[:, 0])  # arctan (y/x)

        dg = torch.norm(torch.cat([dx, dy], dim=1), 2, dim=1, keepdim=True)
        v_pref = state[:, 7].reshape((batch, -1))

        #以当前位置与终点间连线作为横坐标重新建系
        vx = (state[:, 2] * torch.cos(rot) + state[:, 3] * torch.sin(rot)).reshape((batch, -1))
        vy = (state[:, 3] * torch.cos(rot) - state[:, 2] * torch.sin(rot)).reshape((batch, -1))

        radius = state[:, 4].reshape((batch, -1))
        if self.kinematics == 'unicycle':
            theta = (state[:, 8] - rot).reshape((batch, -1))
        else:
            # set theta to be zero since it's not used
            theta = torch.zeros_like(v_pref)
        vx1 = (state[:, 11] * torch.cos(rot) + state[:, 12] * torch.sin(rot)).reshape((batch, -1))
        vy1 = (state[:, 12] * torch.cos(rot) - state[:, 11] * torch.sin(rot)).reshape((batch, -1))
        px1 = (state[:, 9] - state[:, 0]) * torch.cos(rot) + (state[:, 10] - state[:, 1]) * torch.sin(rot)
        px1 = px1.reshape((batch, -1))
        py1 = (state[:, 10] - state[:, 1]) * torch.cos(rot) - (state[:, 9] - state[:, 0]) * torch.sin(rot)
        py1 = py1.reshape((batch, -1))
        radius1 = state[:, 13].reshape((batch, -1))
        radius_sum = radius + radius1
        da = torch.norm(torch.cat([(state[:, 0] - state[:, 9]).reshape((batch, -1)), (state[:, 1] - state[:, 10]).
                                  reshape((batch, -1))], dim=1), 2, dim=1, keepdim=True)
        new_state = torch.cat([dg, v_pref, theta, radius, vx, vy, px1, py1, vx1, vy1, radius1, da, radius_sum], dim=1)
        
        # dg, v_pref, theta, radius, vx, vy, px1, py1, vx, vy, radius, da, radius_sum
        #  0    1      2       3      4   5   6    7   8   9     10    11      12
        # N行12列的转换后的当前坐标下的机器人状态信息

        return new_state
    
    def transform(self, state, other_robot):
        def dist(human):
        #  sort human order by decreasing distance to the robot
            return np.linalg.norm(np.array(human.position) - np.array(state.self_state.position))
        def dist2(human):
            return np.linalg.norm(np.array(human.position) - np.array(other_robot.position))
        state.human_states =  sorted(state.human_states, key=dist, reverse=True)
        state_robot1_tensor = torch.cat([torch.Tensor([state.self_state + human_state]).to(self.device)
                                         for human_state in state.human_states], dim=0)    #张量拼接
        state.human_states =  sorted(state.human_states, key=dist2, reverse=True)
        state_robot2_tensor = torch.cat([torch.Tensor([other_robot + human_state]).to(self.device)
                                         for human_state in state.human_states], dim=0)
        state_robot1_tensor = self.rotate(state_robot1_tensor)
        state_robot2_tensor = self.rotate(state_robot2_tensor)

        state_tensor = torch.cat(([state_robot1_tensor, state_robot2_tensor]), dim=1)
        # 1*26
        return state_tensor

class JointState_2robot(object):
    def __init__(self, robot_state, other_robot_state, human_states):
        # assert isinstance(robot1_state, FullState)
        # for human_state in human_states:
        #     assert isinstance(human_state, ObservableState)

        self.self_state = robot_state

        self.human_states = human_states


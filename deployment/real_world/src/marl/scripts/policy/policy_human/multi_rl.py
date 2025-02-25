import logging
import torch
import torch.nn as nn
import itertools
from envs.utils.action import *
import numpy as np
from policy.policy_human.cadrl import mlp
from policy.policy_human.cadrl import CADRL
from envs.utils.state_lux import Trans_OB

def build_action_space():
    speeds = [(np.exp((i + 1) / 5) - 1) / (np.e - 1) * 1 for i in range(5)]
    rotations = np.linspace(0, 2*np.pi, 16, endpoint=False)  # 均分
    action_sapce = [ActionXY(0, 0)]
    for rotation, speed in itertools.product(rotations, speeds):
        action_sapce.append(ActionXY(speed * np.cos(rotation), speed * np.sin(rotation)))
    return action_sapce

class ValueNetwork(nn.Module): 
    def __init__(self, input_dim, robot_self_dim, robots_other_dim, mlp_dims, lstm_hidden_dim):
        super().__init__()
        self.robot_self_dim = robot_self_dim
        self.robots_other_dim = robots_other_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.mlp = mlp(robot_self_dim + lstm_hidden_dim*2 , mlp_dims)
        #self.mlp_formation = mlp(robot_self_dim + lstm_hidden_dim,mlp_dims,True)
        self.lstm_obstacle = nn.LSTM(input_dim, lstm_hidden_dim, batch_first=True)    
        self.lstm_formation = nn.LSTM(7 + lstm_hidden_dim, lstm_hidden_dim, batch_first=True) 
        self.robot_num = 3   
        self.obstacle_num = 3
         
    
    def forward(self, state):
        # dg, v_pref, theta, radius, vx, vy,  px1, py1, vx, vy, radius, da, radius_sum
        #  0    1      2       3      4   5   6    7   8   9     10      11    12  
        size = state.shape
        dim = state.dim
        if dim == 2:
            state = state.unsqueeze(0) 
            size = state.shape
        # logging.info('size :%s',size[0]) 
        #为当前机器人与另一机器人进行转换后的向量
        robots_state = state[:, 0, :self.robot_self_dim]
        #logging.info('robots_state%s',robots_state)
        #1号邻居
        robots_state_other = state[:, 0, self.robot_self_dim:]

        robots_state_adj1 = state[:, 0, self.robot_self_dim:]
        #2号邻居
        robots_state_adj2 = state[:, 1, self.robot_self_dim:]

        state_lstm_obstacle = state[:,self.robot_num-1:self.robot_num-1 + self.obstacle_num,:]
        state_lstm_obstacle_other = state[:,self.robot_num-1 + self.obstacle_num:,:]
        state_lstm_obstacle_adj1 = state[:,self.robot_num-1 + self.obstacle_num:self.robot_num-1 + 2 * self.obstacle_num,:]
        state_lstm_obstacle_adj2 = state[:,self.robot_num-1 + 2 * self.obstacle_num:,:]
        h0 = torch.zeros(1, size[0], self.lstm_hidden_dim).cpu()
        c0 = torch.zeros(1, size[0], self.lstm_hidden_dim).cpu()
        h0_f = torch.zeros(1, size[0], self.lstm_hidden_dim).cpu()
        c0_f = torch.zeros(1, size[0], self.lstm_hidden_dim).cpu()   

        #logging.info("state_lstm_obstacle type%s",state_lstm_obstacle.dtype)
        output, (hn1, cn1) = self.lstm_obstacle(state_lstm_obstacle.cpu(), (h0, c0))
        
        output, (hn1_other, cn1) = self.lstm_obstacle(state_lstm_obstacle_other.cpu(), (h0, c0))
        output, (hn1_adj1, cn1) = self.lstm_obstacle(state_lstm_obstacle_adj1.cpu(), (h0, c0))
        output, (hn1_adj2, cn1) = self.lstm_obstacle(state_lstm_obstacle_adj2.cpu(), (h0, c0))


        hn1 = hn1.squeeze(0)

        state_lstm_formation = torch.cat([robots_state_other.unsqueeze(0),hn1_other],dim = 2).squeeze(0)
        # logging.info('robots_state_other %s',state_lstm_obstacle_other)
        # logging.info('hn1_other %s',hn1_other)
        state_lstm_formation = state_lstm_formation.unsqueeze(1)
        # logging.info('state_lstm_adj1 %s',state_lstm_obstacle_adj1)
        # logging.info('hn1_adj1 %s',hn1_adj1)
        # logging.info('state_lstm_adj2 %s',state_lstm_obstacle_adj2)
        #将邻居信息与LSTM得到的邻居障碍物特征进行拼接
        state_lstm_adj1 = torch.cat([robots_state_adj1.unsqueeze(0),hn1_adj1],dim = 2).squeeze(0)
        state_lstm_adj1 = state_lstm_adj1.unsqueeze(1)
        state_lstm_adj2 = torch.cat([robots_state_adj2.unsqueeze(0),hn1_adj2],dim = 2).squeeze(0)
        state_lstm_adj2 = state_lstm_adj2.unsqueeze(1)
        state_lstm_swarm = torch.cat([state_lstm_adj1,state_lstm_adj2],dim=1)

        output, (hn2_s, cn2_s) = self.lstm_formation(state_lstm_swarm.cpu(), (h0_f, c0_f))

        hn2_s = hn2_s.squeeze(0)     

        joint_state_with_robot = torch.cat([robots_state, hn2_s], dim=1)
        joint_state_with_robot2 = torch.cat([joint_state_with_robot, hn1], dim=1)

        value = self.mlp(joint_state_with_robot2)

        
        # logging.info('value:%svalue_human:%svalue_formation:%s',value,value_human,value_formation)
        return value.cpu() 

        
       

class Multi_RL(CADRL):
    def __init__(self):
        self.name = 'MULTI-RL'
        self.kinematics = 'holonomic'

        self.with_interaction_module = None
        self.interaction_module_dims = None

        self.robot_self_dim = 6
        self.human_state_dim = 7
        self.robots_other_dim= 7  
        self.joint_state_dim = self.robot_self_dim + self.human_state_dim#6+5*(n-1)+7
    def configure(self, config):
        self.set_common_parameters(config)
        mlp_dims = [int(x) for x in config.get('multi_rl', 'mlp2_dims').split(', ')]
        global_dims =  config.getint('multi_rl', 'global_state_dim')#lstm
        self.with_om = config.getboolean('multi_rl', 'with_om')
        # with_interaction_module = config.getboolean('multi_rl', 'with_interaction_module')
        self.multiagent_training = config.getboolean('lstm_rl', 'multiagent_training')

        self.model = ValueNetwork(self.input_dim(), self.robot_self_dim, self.robots_other_dim, mlp_dims, global_dims)

        self.action_space = build_action_space()
    
    def reach_destination(self, state):
        dis = state[0,0]
        if dis < state[0, 3]:
            return True
        else:
            return False    
    def predict(self, state, no):
        
        if self.phase is None or self.device is None:
            raise AttributeError('Phase, device attributes have to be set!')

        if no == 0:
            state = self.transform(state, 0)
        if no == 1:         
            state = self.transform(state, 1)
        if no == 2:         
            state = self.transform(state, 2)
        other_action1 = ActionXY(state[0,8], state[0,9])
        other_action2 = ActionXY(state[1,8], state[1,9])
        #state:
        # dg, v_pref, theta, radius, vx, vy, px1, py1, vx, vy, radius, da, radius_sum
        #  0    1      2       3      4   5   6    7   8   9     10     11    12  
        # N行17列的转换后的当前坐标下的机器人状态信息
        if self.reach_destination(state):
            return ActionXY(0, 0) if self.kinematics == 'holonomic' else ActionRot(0, 0) 
        if self.phase == 'train' and self.epsilon is None:
            raise AttributeError('Epsilon attribute has to be set in training phase')    
        if self.action_space is None:
            self.build_action_space(state[0,1])     

        probability = np.random.random()
        if self.phase == 'train' and probability < self.epsilon:
            max_action = self.action_space[np.random.choice(len(self.action_space))]
        else:
            self.robot_action_values = list()
            max_value = float('-inf')
            max_action = None
            for action in self.action_space:
                if self.query_env:#self-attention
                    if no == 0: 
                        next_states, reward, done, info = self.env.onestep_lookahead(ActionXY(0, 0) , ActionXY(0, 0) , other_action1)
                        next_self_state = self.propagate(next_states.robot1_state, action)
                        next_states.robot1_state = next_self_state
                    elif no == 1:
                        next_states, reward, done, info = self.env.onestep_lookahead(ActionXY(0, 0) , action, ActionXY(0, 0) )
                        next_self_state = self.propagate(next_states.robot2_state, action)

                        next_states.robot2_state = next_self_state
                    elif no == 2:
                        next_states, reward, done, info = self.env.onestep_lookahead(ActionXY(0, 0) , ActionXY(0, 0) , action)
                        next_self_state = self.propagate(next_states.robot3_state, action)
                        next_states.robot2_state = next_self_state

                next_states= self.transform(next_states, no)

                #logging.info('%s', next_states)
                
                next_states = next_states.unsqueeze(0)
                # logging.info('next_states dtype:%s',next_states.dtype)
                next_state_value = self.model(next_states.to(torch.float32))
                #logging.info('value:%s',next_state_value)
                next_state_value = next_state_value.data.item()
                # logging.info('%s', next_states)
                # logging.info('%s', next_state_value)
                value = reward[no] + pow(self.gamma, self.time_step * state[0, 1]) * next_state_value
                    # self.robot1_action_values.append(value_1)
                if value > max_value:
                    max_value = value
                    max_action = action
                if max_action is None:
                    raise ValueError('Value network is not well trained. ')    
        # logging.info('action%s', max_action)
        return max_action                                       

    def transform(self, state, no):
        
        state = Trans_OB(state.robot1_state, state.robot2_state, state.robot3_state, state.human_states)
        if no == 0:
            robot_self_state = state.robot1_state
            robot_adj1_state = state.robot2_state
            robot_adj2_state = state.robot3_state
        elif no == 1:
            robot_self_state = state.robot2_state
            robot_adj1_state = state.robot1_state
            robot_adj2_state =state.robot3_state
        elif no == 2:
            robot_self_state = state.robot3_state
            robot_adj1_state = state.robot1_state
            robot_adj2_state =state.robot2_state
        self_list = (robot_self_state.px, robot_self_state.py, robot_self_state.vx, robot_self_state.vy, robot_self_state.radius,
                     robot_self_state.gx, robot_self_state.gy, robot_self_state.v_pref, robot_self_state.theta)
        adj1_list = (robot_adj1_state.px, robot_adj1_state.py, robot_adj1_state.vx, robot_adj1_state.vy, robot_adj1_state.radius)
        adj2_list = (robot_adj2_state.px, robot_adj2_state.py, robot_adj2_state.vx, robot_adj2_state.vy, robot_adj2_state.radius)
        adj1_full_state = (robot_adj1_state.px, robot_adj1_state.py, robot_adj1_state.vx, robot_adj1_state.vy, robot_adj1_state.radius, 
                           robot_adj1_state.gx, robot_adj1_state.gy, robot_adj1_state.v_pref, robot_adj1_state.theta)
        adj2_full_state = (robot_adj2_state.px, robot_adj2_state.py, robot_adj2_state.vx, robot_adj2_state.vy, robot_adj2_state.radius, 
                           robot_adj2_state.gx, robot_adj2_state.gy, robot_adj2_state.v_pref, robot_adj2_state.theta)
        

        state_adj1_tensor = torch.cat([torch.tensor([self_list + adj1_list])]) 
        state_adj2_tensor = torch.cat([torch.tensor([self_list + adj2_list])])
        def dist1(obstacle):
            return np.linalg.norm(np.array(obstacle.position) - np.array(robot_self_state.position))
        def dist2(obstacle):
            return np.linalg.norm(np.array(obstacle.position) - np.array(robot_adj1_state.position))
        def dist3(obstacle):
            return np.linalg.norm(np.array(obstacle.position) - np.array(robot_adj2_state.position))
        
        #self_state transform
        state.human_states = sorted(state.human_states, key=dist1, reverse=True)
        state_obstacle_tensor = torch.cat([torch.Tensor([self_list + 
                                        (human_state.px, human_state.py, human_state.vx, human_state.vy, human_state.radius)]).to(self.device)
                                        for human_state in state.human_states], dim=0)
        swarm_adj1_list  = torch.tensor([self_list + adj1_list]).to(self.device)
        swarm_adj2_list  = torch.tensor([self_list + adj2_list]).to(self.device)
        swarm_tensor = torch.cat((swarm_adj1_list,swarm_adj2_list),dim=0)
        # logging.info("swarm_adj1_list%s",swarm_adj1_list)
        # logging.info("swarm_tensor%s",swarm_adj2_list )
        # logging.info("swarm_tensor%s",swarm_tensor)
        state_obstacle_tensor = torch.cat([swarm_tensor,state_obstacle_tensor], dim=0)        

        #adjacent_state transform          
        state.human_states = sorted(state.human_states, key=dist2, reverse=True)      
        state_adj1_obstacle_tensor = torch.cat([torch.Tensor([adj1_full_state + 
                                        (human_state.px, human_state.py, human_state.vx, human_state.vy, human_state.radius)]).to(self.device)
                                         for human_state in state.human_states], dim=0)
        
        state.human_states = sorted(state.human_states, key=dist3, reverse=True)      
        state_adj2_obstacle_tensor = torch.cat([torch.Tensor([adj2_full_state + 
                                        (human_state.px, human_state.py, human_state.vx, human_state.vy, human_state.radius)]).to(self.device)
                                         for human_state in state.human_states], dim=0)
        state_self_tensor = self.rotate(state_obstacle_tensor)
        state_adj1_tensor = self.rotate(state_adj1_obstacle_tensor)
        state_adj2_tensor = self.rotate(state_adj2_obstacle_tensor)
        #引入编队权重概念（暂时没有）通过编队权重，更改其他机器人的特征向量的排列顺序
        state_tensor = torch.cat([state_self_tensor, state_adj2_tensor, state_adj1_tensor],dim = 0)
        #0-1 其他智能体在当前智能体下的特征向量
        #2-4 当前智能体下的障碍物特征信息
        #5-7 邻居1状态下的障碍物特征信息
        #8-10 邻居2状态下的障碍物特征信息
        return state_tensor
    

    def rotate(self, state):


        #robot1 
        # 'px', 'py', 'vx', 'vy', 'radius', 'gx', 'gy', 'v_pref', 'theta'
        # 0     1      2     3      4        5     6      7         8
        #robot_others
        #'px', 'py', 'vx', 'vy', 'radius', 'gx', 'gy', 'v_pref', 'theta'
        # 9     10     11     12     13     14    15      16        17
        # 'px1', 'py1', 'vx1', 'vy1', 'radius1'
        #   18     19     20     21       22

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
        #robot other
        # px2 = (state[:, 9] - state[:, 0]) * torch.cos(rot) + (state[:, 10] - state[:, 1]) * torch.sin(rot)
        # px2 = px2.reshape((batch, -1))
        # py2 = (state[:, 10] - state[:, 1]) * torch.cos(rot) - (state[:, 9] - state[:, 0]) * torch.sin(rot)
        # py2 = py2.reshape((batch, -1))
        # vx2 = (state[:, 11] * torch.cos(rot) + state[:, 12] * torch.sin(rot)).reshape((batch, -1))
        # vy2 = (state[:, 12] * torch.cos(rot) - state[:, 11] * torch.sin(rot)).reshape((batch, -1))       
        # radius2 = state[:, 13].reshape((batch, -1))
        radius1 = state[:, 13].reshape((batch, -1))
        radius_sum = radius + radius1
        da = torch.norm(torch.cat([(state[:, 0] - state[:, 9]).reshape((batch, -1)), (state[:, 1] - state[:, 10]).
                                  reshape((batch, -1))], dim=1), 2, dim=1, keepdim=True)
        new_state = torch.cat([dg, v_pref, theta, radius, vx, vy, px1, py1, vx1, vy1, radius1, da, radius_sum], dim=1)
        
        # dg, v_pref, theta, radius, vx, vy, px2, py2, vx2, vy2, radius2 px1, py1, vx, vy, radius, da, radius_sum
        #  0    1      2       3      4   5   6    7   8   9     10      11    12  13  14   15     16     17   
        # N行17列的转换后的当前坐标下的机器人状态信息

        return new_state
     

    
    def load_model(self, model):
        self.model = model
    def set_device(self, device):
        self.device = device
        self.model.to(device)

    def get_multi_model(self, model):
        self.model = model
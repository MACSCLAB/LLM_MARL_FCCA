#!/usr/bin/env python3
import torch
import rospy
import sys
from geometry_msgs.msg import Twist
from functools import partial

from policy.mappo.MAPPOPolicy import RMAPPOPolicy as Policy
from config.config import get_config    
from gym import spaces
from policy.multi_discrete import MultiDiscrete
from marl.msg import Obstaclelist, robot_Odometry
import math
import numpy as np
from std_msgs.msg import Bool
import time

# 0.initialize the policy
    #从launch文件中启动时，决定该节点机器人的编号以及选取的网络权重
# 1.订阅motive中自己的位置信息，并联合自身的速度消息进行发布,将消息合并到里程计Odom中一起发布
    #需要订阅一个经过数据处理过的节点，该节点接收动捕数据以及自身的速度信息
    #(*最基本的速度从小车的地盘直接获取，可能不准，可以用动捕imu融合一下(卡尔曼滤波)*(可选))
            #该节点名称为Odom_get.py
                #订阅小车的原始里程计，将位置信息替换为动捕(vrpn_ros)传回来的位置信息
            #节点名称/robot{i}/Odometry
    #/
# 2.订阅障碍物的坐标信息，并将障碍物的信息进行转换
    #具体实现在另外一个节点中，pub_obstacle.py，接收动捕传回来的障碍物刚体数据
    #/obstacle 
    #msg.pos[0]- pos[n-1]
    #msg.vel[0] - vel[n-1]
# 3.订阅其他智能体的相关状态信息
    #
# 4.将障碍物信息，其他智能体的相关信息进行转换，规范为策略输入state的格式
# self.stainer.policy.act
# 5.从actor输出中获取动作，并将动作发布给对应的\cmd_vel


def parse_args(args,parser):
    parser.add_argument('--num_robots', type=int,default=3, help="number of players")
    all_args = parser.parse_known_args(args)[0]
    return all_args

paraser = get_config()
all_args = parse_args(sys.argv[1:], paraser)
all_args.model_dir = '/home/shenzhen/marl0.0/results/train/run18/models'
device = torch.device("cuda:0")

agent_ob_space = spaces.Box(low=-np.inf, high=+np.inf,shape=(all_args.robot_obs_dim,),dtype=np.float32,)
obstacle_space = spaces.Box(low=-np.inf, high=+np.inf,shape=(all_args.robot_obs_dim,),dtype=np.float32,)
share_ob = spaces.Box(low=-np.inf, high=+np.inf,shape=((
            max(all_args.num_humans,all_args.num_attention_agents)+all_args.num_robots)*all_args.robot_obs_dim,),
            dtype=np.float32,)
vel_act_space = spaces.Discrete(all_args.vel_action_dim)
forward_act_space = spaces.Discrete(all_args.dir_action_dim)
total_action_space = []
total_action_space.append(forward_act_space)
total_action_space.append(vel_act_space)
act_space = MultiDiscrete([[0, act_space.n-1] for act_space in total_action_space])


###########################formation reference############################
def get_weight(x1, y1, x2, y2):
    return (x1 - x2) ** 2 + (y1 - y2) ** 2

con_vec = [[0,0], [0.73,0.4], [0,0.8]]


class RLrobot:
    def __init__(self):
        torch.manual_seed(all_args.seed)
        torch.cuda.manual_seed_all(all_args.seed)
        np.random.seed(all_args.seed)

        self.dir_action_dim = all_args.dir_action_dim
        self.vel_action_dim = all_args.vel_action_dim

        self.robot_state = np.zeros((all_args.robot_obs_dim))
        #initialize the obstacle
        self.obstacle_state = np.zeros((all_args.robot_obs_dim))
        self.obstacle_state = np.stack(self.obstacle_state) # ?
        self.robotnum = 3 #正常情况下需要从ros的参数服务器中获取
        self.id = int(rospy.get_param('~robot_id', '0'))

        W = np.array([[0,1,1],
                      [1,0,1],
                      [1,1,0]])
        D = np.diag(sum(W))
        L = D - W
        D_sys = np.diag(pow(sum(W),-1/2))  #sysmmetric normolize
        self.L_des = D_sys @ L @ D_sys

        #initialize the mappo policy
        self.rnn_states = np.zeros((self.robotnum, 1, 64,),dtype=np.float32,)
        self.masks = np.ones((self.robotnum, 1), dtype=np.float32)

        self.policy = Policy(all_args, agent_ob_space, obstacle_space, share_ob, act_space, device)
        policy_actor_state_dict = torch.load(str(all_args.model_dir) + "/actor_agent" + str(self.id) + ".pt")
        self.policy.actor.load_state_dict(policy_actor_state_dict)
        
        # the robot swarm
            #each robots get px,py,gx,gy,v
        self.swarm_robots = np.empty((3,6)) # robot_num, (px,py,gx,gy,v,theta)
        self.goals = []
        self.start_flag = False

        self.v = 0
        self.theta = 0

        #subscribe the robot vel from the odom
        for i in range(1, self.robotnum + 1):
            #订阅互相通信的机器人的里程计数据
            #px,py,gx,gy,v，这里最好改为append进来一个类，类名为订阅的机器人编号class.no，因为可能后续并非全部都互相通信
            #这边为了图省事，swarm_robots[0]为1号，swarm_robots[2]为2号...
            # self.swarm_robots.append(np.array([0,0,0,0,0]))
            #从参数服务器中获取当前机器人的目的地
            self.goals.append(np.array([rospy.get_param(f'/robot{i}/gx'),
                                          rospy.get_param(f'/robot{i}/gy')]))
            
            rospy.Subscriber(f"/robot{i}/odom", robot_Odometry, lambda msg, index=i-1: self.robot_callback(msg,index), queue_size=1)

        #subscribe the robot
        rospy.Subscriber("/obstacle",Obstaclelist,self.obstacle_callback, queue_size=1)
        rospy.Subscriber("/start_flag",Bool,self.start_callback)
        print("init finish")

    def act(self):
        #define publisher
        pub = rospy.Publisher("/robot/cmd_vel", Twist, queue_size=1)
        rate = rospy.Rate(10)
        step = 0
        while not rospy.is_shutdown():
            if self.robot_state is not None:
                step += 1
                # print('step',step)
                # print('obstacle_state', self.obstacle_state[1][:2])
                # print('robot_state', self.robot_state[:2])

                vel_msg = Twist()
                dmin = np.sqrt((self.obstacle_state[0][0] - self.robot_state[0])**2 + (self.obstacle_state[0][1] - self.robot_state[1])**2)
                if dmin <= 0.6:
                    print("collision!")

                robot_state = np.stack([self.robot_state])
                obstacle_state = np.stack([self.obstacle_state])
                action, _ = self.policy.act(robot_state, 
                                            obstacle_state,
                                            self.rnn_states[int(self.id)],
                                            self.masks[int(self.id)].astype(np.float32),
                                            deterministic=True,)
                
                action = action.detach().cpu().numpy()
                action = action[0]
                # print('action',action)
                for i in range(self.dir_action_dim):
                    if action[0] == i:
                        self.theta = 2 * np.pi / self.dir_action_dim * i
                        # print(self.theta)
                for i in range(self.vel_action_dim):
                    if action[1] == i:
                        self.v = 0.1 * (i+1)

                dist2goal = np.sqrt(self.robot_state[2]**2 + self.robot_state[3]**2)
                if dist2goal <= 0.45:
                    self.v = 0
                    print("reach goal!")
                vx = self.v*math.cos(self.theta)
                vy = self.v*math.sin(self.theta)
                # print('velosity',vx,vy)
                vel_msg.linear.x = vx
                vel_msg.linear.y = vy
                if self.start_flag == True:
                    pub.publish(vel_msg)
                    # print(vx, vy)
                else:
                    rospy.loginfo("some robot hasn't prepared yet")
                    pub.publish(Twist())
            rate.sleep()

    def obstacle_callback(self, msg):
        # print(msg)
        self.obstacle_state = self.obstacle_state_Trans(msg)
    
    def start_callback(self,msg):
        if msg.data == True:
            self.start_flag = True
        else:
            self.start_flag = False
    
    def robot_callback(self, msg ,index):
        # print(msg)
        px = msg.pos.x
        py = msg.pos.y
        gx = self.goals[index][0] - px
        gy = self.goals[index][1] - py
        # v = ((msg.vel.linear.x)**2 + (msg.vel.linear.y)**2)**0.5

        current_state = np.array([px, py, gx, gy, self.v,self.theta])
        self.swarm_robots[index] = current_state
        feature, formation_vx, formation_vy = self.get_formation_feature(self.swarm_robots)
        # print('id',index)
        #给智能体添加编队特征状态
        if index == self.id:
            # print('position',px,py)
            robot_state = np.append(current_state, feature)
            robot_state = np.append(robot_state, formation_vx)
            robot_state = np.append(robot_state, formation_vy)
            self.robot_state = robot_state

    
    def get_formation_feature(self,swarm_robots):
        """swarm_robots
        0  1  2  3  4  
        px py gx gy v
        """
        swarm_robots = np.array(swarm_robots)
        len = swarm_robots.shape[0]
        W = np.zeros((self.robotnum, self.robotnum))
        for i in range(len):
            for j in range(len):
                W[i][j] = get_weight(swarm_robots[i][0],swarm_robots[i][1],
                                     swarm_robots[j][0],swarm_robots[j][1])
        D = np.diag(sum(W))  
        L = D - W
        D_sys = np.diag(pow(sum(W),-1/2))  #sysmmetric normolize
        # print(D_sys)
        L_hat = D_sys @ L @ D_sys 
        f = np.trace(np.transpose((L_hat - self.L_des)) @ (L_hat - self.L_des))

        vx = 0
        vy = 0
        for i in range(len):
            if i != self.id:
                vx -= (self.robot_state[0] - self.swarm_robots[i][0]  -  (con_vec[self.id][0] - con_vec[i][0]))
                vy -= (self.robot_state[1] - self.swarm_robots[i][1]  -  (con_vec[self.id][1] - con_vec[i][1]))
        return f,vx, vy
    
    def obstacle_state_Trans(self, msg):
        obstacle = np.empty((len(msg.pos),all_args.human_obs_dim))
        for i in range(len(msg.pos)):
            px = msg.pos[i].x
            py = msg.pos[i].y
            vx = msg.vel[i].linear.x
            vy = msg.vel[i].linear.y
            theta = np.arctan2(vx, vy)
            obstacle[i] = (np.array([px,py,vx,vy,theta]))
        rx,ry = self.robot_state[0],self.robot_state[1]
        distances = [math.sqrt((px - rx)**2 + (py - ry)**2) for px, py, *_ in obstacle]
        sorted_indices = np.argsort(distances)
        sorted_obstacle = obstacle[sorted_indices]
        sorted_obstacle = np.stack(sorted_obstacle)
        return sorted_obstacle


if __name__ == "__main__":
    rospy.init_node('agent_policy_node')
    agent = RLrobot()
    time.sleep(1)
    agent.act()
    rospy.spin()

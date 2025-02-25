import numpy as np
import rvo2
from policy.policy_human.policy import Policy
from envs.utils.math_utils import get_weight
import torch

class ORCAPolicy(Policy):
    def __init__(self, args):

        super().__init__()
        self.trainable = False
        self.multiagent_training = None
        self.kinematics = 'holonomic'
        self.neighbor_dist = args.neighbor_dist
        self.max_neighbors = args.max_neighbors
        self.time_horizon = args.time_horizon
        self.time_horizon_obst = args.time_horizon_obst
        self.max_speed = args.max_speed
        self.robot_num = args.num_robots
        self.human_num = args.num_humans
        self.robot_radius = args.robot_radius
        self.human_radius = args.human_radius
        self.v_pref = args.v_pref
        self.safety_space = 0
        self.radius = 0.3
        self.time_step = args.time_step
        self.sim = None

    def formation_control(self, states):
        # param states: [px,py,vx,vy,gx,gy] * (robot_num + human_num)
        # print(states)
        # h_desire = np.array([0,0,np.sqrt(3),1,0,2])
        # L = []
        h_desire = np.array([[0,-2,1.5,1,-1.5,1]])
        L = np.array([[-2, 1, 1], [1, -2 , 1], [1, 1, -2]])
        x = []
        u_bot = []
        I = np.array([[1,0],[0,1]])
        mini_bias = 0.00001
        W = np.zeros((self.robot_num, self.robot_num))

        #TODO check the validity of each matrix
        for i,state in enumerate(states[:self.robot_num]):
            # print(state)
            # h_desire = np.concatenate((h_desire, [state[-2],state[-1]]))
            x = np.concatenate((x, [state[0],state[1]]))
        #     for j, st in enumerate(states[:self.robot_num]):
        #         W[i][j] = get_weight(state[-2], state[-1], st[-2], st[-1])
        # D = np.diag(sum(W))
        # L = D - W
        
        x = np.array([x])
        kron_L = np.kron(L, I)
        u = (np.dot(kron_L,x.T)-h_desire.T)

        for i in range(self.robot_num):
            u_temp = []
            u_norm = 0
            u_temp = u[i*2:i*2+2]
            u_norm = np.linalg.norm(u_temp + mini_bias)
            u_bot.append(-u_temp/u_norm)
        assert len(u_bot) == self.robot_num, 'u_bot compute error!'
        
        return u_bot

    def calculate(self, states ,u):
        action = []  # vx,vy
        params = self.neighbor_dist, self.max_neighbors, self.time_horizon, self.time_horizon_obst
        self.sim = rvo2.PyRVOSimulator(self.time_step, *params, self.radius, self.max_speed)

        for i,state in enumerate(states):
            if i < self.robot_num:
                self.sim.addAgent((state[0],state[1]), *params, self.robot_radius+self.safety_space, self.v_pref, (state[2],state[3]))
            else:
                self.sim.addAgent((state[0],state[1]), *params, self.human_radius+self.safety_space, self.v_pref, (state[2],state[3]))

        # Set the preferred velocity to be a vector of unit magnitude (speed) in the direction of the goal.
        velocity = np.array((states[0][-2] - states[0][0], states[0][-1] - states[0][1]))
        speed = np.linalg.norm(velocity)
        pref_vel = velocity / speed if speed > 1 else velocity
        pref_vel = pref_vel + 0.1 * u.T.reshape(-1)
        self.sim.setAgentPrefVelocity(0, tuple(pref_vel))

        for i in range(self.human_num):
            # unknown goal position of other humans
            self.sim.setAgentPrefVelocity(i + self.robot_num, (0, 0))

        self.sim.doStep()
        action.extend(self.sim.getAgentVelocity(0))
        return action

    def predict(self, states):
        """
        Create a rvo2 simulation at each time step and run one step
        Python-RVO2 API: https://github.com/sybrenstuvel/Python-RVO2/blob/master/src/rvo2.pyx
        How simulation is done in RVO2: https://github.com/sybrenstuvel/Python-RVO2/blob/master/src/Agent.cpp

        Agent doesn't stop moving after it reaches the goal, because once it stops moving, the reciprocal rule is broken

        :param state:
        :return:
        """
        actions = []
        u_bot = self.formation_control(states)
        states = np.array(states)
        
        for no in range(self.robot_num):
            new_states = self.transform(states.copy(), no)
            actions.append(self.calculate(new_states, u_bot[no]))

        self.device = torch.device("cpu")
        if self.device is None:
            raise AttributeError('Phase, device attributes have to be set!')
        return actions


    def transform(self, states, no):
        temp_state = states[0].copy()
        states[0] = states[no]
        states[no] = temp_state
        return states


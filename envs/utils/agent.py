import numpy as np
import math
from numpy.linalg import norm
import abc
import logging
from envs.utils.action import ActionXY, ActionRot
from envs.utils.state import ObservableState, FullState

# Base class for robot and human. Have the physical attributes of an agent.
class Agent(object):
    def __init__(self):
        self.px = None # Current x-coordinate position of the agent.
        self.py = None # Current y-coordinate position of the agent.
        self.gx = None # Goal x-coordinate position of the agent.
        self.gy = None # Goal y-coordinate position of the agent.
        self.theta = None # Orientation angle of the agent.
        self.time_step = None # Discrete time step for simulation or movement updates.
        self.success = None # Flag indicating whether the agent has successfully reached its goal.
        self.radius = None # Radius of the agent, used for collision detection and avoidance.

        self.kinematics = 'holonomic'


    def config(self, config, section):
        self.visible = config.getboolean(section, 'visible')
        self.sensor = config.get(section, 'sensor')
        pass


    def print_info(self):
        logging.info('Agent is {} and has {} kinematic constraint'.format(
            'visible' if self.visible else 'invisible', self.kinematics))

    def set_policy(self, policy):
        self.policy = policy
        self.kinematics = policy.kinematics

    def set(self, px, py, gx, gy, radius=None, theta=np.pi/2, vx=0, vy=0, v_pref=None):
        self.px = px
        self.py = py
        self.gx = gx
        self.gy = gy
        self.vx = vx
        self.vy = vy
        if theta:
            self.theta = theta
        if radius is not None:
            self.radius = radius
        if v_pref is not None:
            self.v_pref = v_pref
        self.dist2goal = math.sqrt((self.gx - self.px) ** 2 + (self.gy - self.py) ** 2)
        

    def get_observable_state(self):
        return ObservableState(self.px, self.py, self.vx, self.vy, self.radius)

    def get_next_observable_state(self, action):
        self.check_validity(action)
        pos = self.compute_position(action, self.time_step)
        next_px, next_py = pos
        if self.kinematics == 'holonomic':
            next_vx = action.vx
            next_vy = action.vy
        else:
            next_theta = self.theta + action.r
            next_vx = action.v * np.cos(next_theta)
            next_vy = action.v * np.sin(next_theta)
        return ObservableState(next_px, next_py, next_vx, next_vy, self.radius)

    def get_full_state(self):
        return FullState(self.px, self.py, self.vx, self.vy, self.radius, self.gx, self.gy, self.v_pref, self.theta)

    def get_position(self):
        return self.px, self.py

    def set_position(self, position):
        self.px = position[0]
        self.py = position[1]

    def get_goal_position(self):
        return self.gx, self.gy

    def get_velocity(self):
        return self.vx, self.vy

    def set_velocity(self, velocity):
        self.vx = velocity[0]
        self.vy = velocity[1]

    @abc.abstractmethod
    def act(self, ob):
        """
        Compute state using received observation and pass it to policy

        """
        return

    def check_validity(self, action):
        if self.kinematics == 'holonomic':
            assert isinstance(action, ActionXY)
        else:
            assert isinstance(action, ActionRot)

    def compute_position(self, action, delta_t):
        self.check_validity(action)
        if self.kinematics == 'holonomic':
            px = self.px + action.vx * delta_t
            py = self.py + action.vy * delta_t
        else:
            theta = self.theta + action.r
            px = self.px + np.cos(theta) * action.v * delta_t
            py = self.py + np.sin(theta) * action.v * delta_t

        return px, py

    def reached_destination(self):
        return norm(np.array(self.get_position()) - np.array(self.get_goal_position())) < self.radius


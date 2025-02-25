from envs.utils.agent import Agent
from envs.utils.state import JointState
import numpy as np

class Robot(Agent):
    def __init__(self, args):
        super().__init__()
        self.v = None # The robot velosity
        self.collision = False  # Flag indicating if a collision has occurred.
        self.dmin = float('inf') # The distance to the nearest agent(include obstacle and other robot) from this robot.
        self.id = 0  # Unique identifier for the robot.
        self.L_des = None # Desired formation Laplacian matrix for maintaining the formation structure.
        self.vx_formation = None # Reference velocity in the x-direction for formation consistency.
        self.vy_formation = None # Reference velocity in the y-direction for formation consistency.
        self.for_std = None # Reference vector for calculating formation consistency.
        self.pre_v = None # Last step velosity
        self.pre_theta = None # Last step angle for the robot.
        self.goal_flag = None # Flag indicating whether the robot has reached its destination.
        self.for_feature = None # The final feature of the formation, i.e. the trace of the DIFFERENCE between the Laplacian matrix of the ideal formation and the Laplacian matrix of the current formation.
        self.dist2goal = None # Distance to goal 
        self.pre_dist2goal = None # Last step distance to goal
        self.set_config(args)

    def set_config(self, args):
        self.radius = args.robot_radius # Radius of the robot.
        self.edge = args.for_edge # Edge length of the formation shape.
        self.discomfort_dist = args.dcf_dist # Distance at which another agent is considered too close, causing discomfort.

    def act(self, ob):
        if self.policy is None:
            raise AttributeError('Policy attribute has to be set!')
        state = JointState(self.get_full_state(), ob)
        action, action_indice = self.policy.predict(state)

        return action, action_indice

def generate_robot(args):
    robots = [Robot(args) for i in range(args.num_robots)]
    set_bias = np.random.random() * 0.5
    px = -5
    py = -5
    # px = 0
    # py = 0
    # coordinates_list = [(4,6),(3,6),(2,6),(1,6),(0,6),(-1,6),(-2,6),(4,5),(3,5),(1,5),(-1,5),(-5,4),(-5,6)]
    coordinates_list = [(5,6),(4,6),(3,6),(2,6),]
    # coordinates_list = [(5,6),(4,6),(3,6),(2,6),(1,6)]
    # coordinates_list = [(2,2), (2,1.5), (2,1), (2,0.5)]
    random_index = np.random.randint(len(coordinates_list))
    random_coordinate = coordinates_list[random_index]
    gx = random_coordinate[0] + set_bias
    gy = random_coordinate[1] + set_bias
    n = 0
    fx = 0
    fy = 0
    for robot in robots:
        robot.vx_formation = 0
        robot.vy_formation = 0
        robot.set(px, py, gx, gy)
        # px += (robot.edge / 2) * np.sqrt(3) * (-1) ** n
        py += (robot.edge / 2)
        gx += (robot.edge / 2) * np.sqrt(3) * (-1) ** n
        robot.id = n
        gy += (robot.edge / 2)
        n += 1
        robot.v = 0
        robot.theta = 0
        robot.dmin = float('inf')
        robot.collision = None
        robot.success = None
        fx += (robot.edge / 2) * np.sqrt(3) * (-1) ** n
        fy += (robot.edge / 2)
        robot.for_std = [fx,fy]
    # 编队一致性
    for robot in robots:
        for r in robots:
            if robot != r:
                robot.vx_formation -= (robot.px - r.px - (robot.for_std[0] - r.for_std[0]))
                robot.vy_formation -= (robot.py - r.py - (robot.for_std[1] - r.for_std[1]))

    return robots
from envs.utils.agent import Agent    
from envs.utils.state import JointState
from policy.policy_human.orca import ORCA
import numpy as np


class Human(Agent):
    def __init__(self, args):
        super().__init__()
        self.vx = None
        self.vy = None
        self.policy = ORCA()
        self.dist2rob = None
        self.set_config(args)
        
    def set_config(self, args):
        self.radius = args.human_radius
        self.v_pref = args.v_pref
        
        
    def act(self, ob):
        """
        The state for human is its full state and all other agents' observable states
        :param ob:
        :return:
        """
        state = JointState(self.get_full_state(), ob)
        self.policy.safety_space = 0.1
        action = self.policy.predict(state)
        return action
    
    def step(self, action):
        """
        Perform an action and update the state
        """
        self.check_validity(action)
        pos = self.compute_position(action, self.time_step)
        self.px, self.py = pos
        # print(self.px, self.py)
        if self.kinematics == 'holonomic':
            self.vx = action.vx
            self.vy = action.vy
        else:
            self.theta = (self.theta + action.r) % (2 * np.pi)
            self.vx = action.v * np.cos(self.theta)
            self.vy = action.v * np.sin(self.theta)

class ObstacleGenerator:
    def __init__(self, robots, num_obstacles, map_boundary=(-10, 10, -10, 10), obstacle_radius_range=(0.35, 0.4)):
        self.map_boundary = map_boundary
        self.robots = robots  # Robots are now in the format [px, py, gx, gy, radius]
        self.num_obstacles = num_obstacles
        self.obstacle_radius_range = obstacle_radius_range
        self.robots_obs = self.format_robots()

    def generate_obstacle(self, existing_obstacles):
        while True:
            is_dynamic = np.random.rand() < 0.7
            
            if is_dynamic:
                start_position = self._random_point_within_map()
                end_position = self._random_point_within_map()
            else:
                start_position = self._random_point_within_map()
                end_position = start_position.copy()
            
            radius = np.random.uniform(*self.obstacle_radius_range)

            # Ensure at least one of the positions (start or end) does not collide with any robot or existing obstacles
            if (not self._collides_with_robot(start_position, radius) and 
                not self._collides_with_obstacles(start_position, radius, existing_obstacles) and
                (not is_dynamic or 
                 (not self._collides_with_robot(end_position, radius) and 
                  not self._collides_with_obstacles(end_position, radius, existing_obstacles)))):
                return [start_position[0], start_position[1], end_position[0], end_position[1], radius]

    def _random_point_within_or_outside_map(self):
        """Generate a point that may be inside or outside the map."""
        x = np.random.uniform(self.map_boundary[0] - 1, self.map_boundary[1] + 1)
        y = np.random.uniform(self.map_boundary[2] - 1, self.map_boundary[3] + 1)
        return np.array([x, y])

    def _random_point_within_map(self):
        """Generate a point within the map boundaries."""
        x = np.random.uniform(self.map_boundary[0], self.map_boundary[1])
        y = np.random.uniform(self.map_boundary[2], self.map_boundary[3])
        return np.array([x, y])

    def _collides_with_robot(self, obstacle_center, obstacle_radius):
        for robot in self.robots_obs:
            for pos in [(robot[0], robot[1]), (robot[2], robot[3])]:
                robot_pos = np.array(pos)
                robot_radius = robot[4]
                distance = np.linalg.norm(robot_pos - obstacle_center)
                if distance < (robot_radius + obstacle_radius) * 2:
                    return True
        return False

    def _collides_with_obstacles(self, obstacle_center, obstacle_radius, existing_obstacles):
        for obs in existing_obstacles:
            for pos in [(obs[0], obs[1]), (obs[2], obs[3])]:
                if obs[0] == obs[2] and obs[1] == obs[3]:  # Static obstacle
                    pos = (obs[0], obs[1])
                else:  # Dynamic obstacle
                    pos = (obs[2], obs[3])
                other_radius = obs[4]
                distance = np.linalg.norm(np.array(pos) - obstacle_center)
                if distance < obstacle_radius + other_radius:
                    return True
        return False
    
    def format_robots(self):
        robots_obs = []
        for robot in self.robots:
            robots_obs.append([robot.px, robot.py, robot.gx, robot.gy, robot.radius])
        return robots_obs

    def generate_all_obstacles(self):
        obstacles = []
        for _ in range(self.num_obstacles):
            new_obstacle = self.generate_obstacle(obstacles)
            obstacles.append(new_obstacle)
        return obstacles

def generate_human(args, robots):
    humans = []
    obstacle_generator = ObstacleGenerator(robots, args.num_humans)
    obstacles = obstacle_generator.generate_all_obstacles()
    for i in range(args.num_humans):
        human = Human(args)
        human.set(*obstacles[i])
        humans.append(human)
    
    return humans




    


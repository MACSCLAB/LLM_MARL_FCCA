import gym
from gym import spaces
import numpy as np
from envs.env_core import EnvCore
import configparser
from .multi_discrete import MultiDiscrete
from envs.utils.math_utils import reach_goal


class DiscreteActionEnv(gym.Env):
    """
    对于离散动作环境的封装
    Wrapper for discrete action environment.
    """

    def __init__(self, all_args):
        self.env = EnvCore(all_args)

        self.robot_num = self.env.robot_num
        self.human_num = self.env.human_num
        self.att_agents = self.env.att_agents
        self.obs_dim = self.env.obs_dim
        self.dir_action_dim = self.env.dir_action_dim
        self.vel_action_dim = self.env.vel_action_dim
        self.time_step = self.env.time_step

        self.total_obs = self.env.total_obs
        self.attention_weights = self.env.attention_weights
        self.method = all_args.method
        self.base_v = all_args.base_v

        # configure spaces
        self.action_space = []
        # self.observation_space = []   #gym里面有这个定义
        self.robot_observation_space = []
        self.human_observation_space = []
        self.share_observation_space = []
        share_obs_dim = 0

        self.use_time = 0
        self.record_time = False
        self.collision_flg = False

        for agent_idx in range(self.robot_num):
            total_action_space = []

            for_action_space = spaces.Discrete(self.dir_action_dim)
            total_action_space.append(for_action_space)

            vel_action_space = spaces.Discrete(self.vel_action_dim)
            total_action_space.append(vel_action_space)
            
            # total action space 
            if len(total_action_space) > 1:
                # all action spaces are discrete, so simplify to MultiDiscrete action space
                if all([isinstance(act_space, spaces.Discrete) for act_space in total_action_space]):
                    act_space = MultiDiscrete(
                        [[0, act_space.n-1] for act_space in total_action_space])
                else:
                    act_space = spaces.Tuple(total_action_space)
                self.action_space.append(act_space)
            else:
                self.action_space.append(total_action_space[0])

            # observation space
            share_obs_dim += self.obs_dim+2 # add px and py
            self.robot_observation_space.append(
                spaces.Box(
                    low=-np.inf,
                    high=+np.inf,
                    shape=(self.obs_dim+2,),
                    dtype=np.float32,
                ))  # [-inf,inf]
        
        for human in range(max(self.human_num, self.att_agents)):
            share_obs_dim += self.obs_dim+2 
            self.human_observation_space.append(
                spaces.Box(
                    low=-np.inf,
                    high=+np.inf,
                    shape=(self.obs_dim+2,),  #有,表示长度为self.obs_dim的元组，没有,则就是一个整数
                    dtype=np.float32,
                ))  # [-inf,inf]
            
        self.share_observation_space = [
            spaces.Box(low=-np.inf, high=+np.inf, shape=(share_obs_dim,), dtype=np.float32)
            for _ in range(self.robot_num + max(self.human_num, self.att_agents))
        ]

    def step(self, actions):
        if self.method == 'ppo':
            robot_success = []
            for i,robot in enumerate(self.env.robots):
                self._set_action(actions[i], robot)
                robot_success.append(robot.success)
            if all(robot_success) and not self.record_time:
                self.use_time += self.env.global_time
                self.record_time = True
            results = self.env.step()
            obs, share_obs, rewards, dones, reward_info = results
            return np.stack(obs), np.stack(share_obs), np.stack(rewards), np.stack(dones), reward_info

        elif self.method == 'orca':
            for i,robot in enumerate(self.env.robots):
                robot.vx = actions[i][0]
                robot.vy = actions[i][1]
                if reach_goal(robot):
                    robot.vx = 0
                    robot.vy = 0
                    if robot.success == None:
                        robot.success = True
                robot.px += robot.vx * self.time_step
                robot.py += robot.vy * self.time_step
            return self.env.step()

        elif self.method == 'apf':
            for i,robot in enumerate(self.env.robots):
                robot.vx = actions[i][0]
                robot.vy = actions[i][1]
                if reach_goal(robot):
                    robot.vx = 0
                    robot.vy = 0
                    if robot.success == None:
                        robot.success = True
                robot.px += robot.vx * self.time_step
                robot.py += robot.vy * self.time_step
            return self.env.step()

    def reset(self):
        obs,share_obs = self.env.reset()
        self.record_time = False
        return obs, share_obs

    def seed(self, seed):
        if seed is None:
            np.random.seed(1)
        else:
            np.random.seed(seed)

    #set each robot actions and update positions
    def _set_action(self, action, robot):
        robot.pre_theta = robot.theta
        robot.pre_v = robot.v
        for i in range(self.dir_action_dim):
            if action[0] == i:
                robot.theta = 2*np.pi / self.dir_action_dim * i
        for i in range(self.vel_action_dim):
            if action[1] == i:
                robot.v = self.base_v * (i + 1)
            if reach_goal(robot):
                robot.v = 0
                if robot.success == None:
                    robot.success = True

        robot.px = robot.px + self.time_step * robot.v * np.cos(robot.theta)
        robot.py = robot.py + self.time_step * robot.v * np.sin(robot.theta)
    
    def render(self, mode='vedio', visualize = False):
        from matplotlib import animation
        import matplotlib.pyplot as plt
        from matplotlib import patches
        import matplotlib.lines as mlines

        plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'  #需要用到系统自带的ffmpeg来生成视频（视频解析工具）

        x_offset = 0.11
        y_offset = 0.11
        # cmap = plt.cm.get_cmap('hsv', 10)  # color library
        goal_robot1_color = 'red'
        arrow_color = 'red'
        arrow_style = patches.ArrowStyle("->", head_length=4, head_width=4)
        self.env.total_obs = np.array(self.env.total_obs)
        robot_states = self.env.total_obs[:, :self.robot_num] # shape[step, human_num+robot_num, obs_dim]
        human_states = self.env.total_obs[:, self.robot_num:]

        episode_success = False

        if mode == 'vedio':
            if visualize == True:
                fig,ax = plt.subplots(figsize=(7,7))     #设置图像弹窗的长和宽
                ax.tick_params(labelsize=16)  # the params of axis
                ax.set_xlim(-10, 10)
                ax.set_ylim(-10, 12)
                ax.set_xlabel('x', fontsize=16)
                ax.set_ylabel('y', fontsize=16)

                #这里的设置是初始化，下面会更新
                ## add robot and its goal
                robots = [plt.Circle(robot_states[0][i][:2], self.env.robots[i].radius, fill=True)
                        for i in range(self.robot_num)]
                robot_numbers = [plt.text(robots[i].center[0] - x_offset, robots[i].center[1] - y_offset, str(i),
                                color='black', fontsize=12) 
                                for i in range(self.robot_num)]
                robot_labels = [f'Robot {i}' for i in range(len(robots))]
                for i,robot in enumerate(robots):
                    ax.add_artist(robot)
                    ax.add_artist(robot_numbers[i])
                plt.legend(robots, robot_labels, fontsize=16)

                robot_goals = [mlines.Line2D([robot_states[0][i][2]], [robot_states[0][i][3]],
                                            color=goal_robot1_color, marker='*', linestyle='None',markersize=15, label='Goal_robot1')
                                            for i in range(self.robot_num)]
                goal_labels = [f'RobotGoal {i}' for i in range(len(robot_goals))]
                for i,robot_goal in enumerate(robot_goals):
                    ax.add_artist(robot_goal)
                plt.legend(robot_goals, goal_labels, fontsize=16)

                # add humans and their numbers
                humans = [plt.Circle(human_states[0][i][:2], self.env.humans[i].radius, fill=False)
                        for i in range(self.human_num)]
                human_numbers = [plt.text(humans[i].center[0] - x_offset, humans[i].center[1] - y_offset, str(i),
                                        color='black', fontsize=12) 
                                        for i in range(self.human_num)]
                for i, human in enumerate(humans):
                    ax.add_artist(human)
                    ax.add_artist(human_numbers[i])

                # add time annotation#注释
                time = plt.text(-1, 5, 'Time: {}'.format(0), fontsize=16)   #在图中显示时间
                ax.add_artist(time)

                # compute attention scores
                if self.attention_weights is not None:
                    attention_scores = [
                        plt.text(-5.5, 5 - 0.5 * i, 'Human {}: {:.2f}'.format(i + 1, self.attention_weights[0][i]),
                                fontsize=16) for i in range(self.human_num)]

                # compute orientation in each step and use arrow to show the direction
                radius_robot = self.env.robots[0].radius

                #  orientation
                if self.env.robots[0].kinematics == 'unicycle':
                    orientation = [((state[0].px, state[0].py), (state[0].px + radius_robot * np.cos(state[0].theta),
                                    state[0].py + radius_robot * np.sin(state[0].theta))) for state in self.states]
                    orientations = [orientation]
                else:
                    orientations = []
                    for i in range(self.human_num + self.robot_num):
                        orientation = []
                        for state in self.env.total_obs:
                            if i < self.robot_num:
                                theta = state[i][5]
                            else:
                                theta = np.arctan2(state[i][2], state[i][3])
                            orientation.append(((state[i][0], state[i][1]), (state[i][0] + radius_robot * np.cos(theta),
                                                                                state[i][1] + radius_robot * np.sin(theta))))
                        orientations.append(orientation)

                arrows= [patches.FancyArrowPatch(*orientation[0],color=arrow_color, arrowstyle=arrow_style)
                                for orientation in orientations]
                for arrow in arrows:
                    ax.add_artist(arrow)
                global_step = 0

                def update(frame_num):
                    nonlocal global_step
                    nonlocal arrows
                    global_step = frame_num
                    for i,robot in enumerate(robots):
                        robot.center = robot_states[frame_num][i]
                        robot_numbers[i].set_position((robot.center[0] - x_offset, robot.center[1] - y_offset))
                    for i, human in enumerate(humans):
                        human.center = human_states[frame_num][i]
                        human_numbers[i].set_position((human.center[0] - x_offset, human.center[1] - y_offset))#set_position是plt的方法
                    for arrow in arrows:
                        arrow.remove()
                    arrows = [patches.FancyArrowPatch(*orientation[frame_num], color=arrow_color,
                                                        arrowstyle=arrow_style) for orientation in orientations]
                    for arrow in arrows:
                        ax.add_artist(arrow)
                    if self.attention_weights is not None:
                        human.set_color(str(self.attention_weights[frame_num][i]))
                        attention_scores[i].set_text('human {}: {:.2f}'.format(i, self.attention_weights[frame_num][i]))

                    time.set_text('Time: {:.2f}'.format(frame_num * self.time_step))

                anim = animation.FuncAnimation(fig, update, frames=len(robot_states), interval=self.time_step * 1000)
                anim.running = True
                plt.show()

            #calculate success rate
            success_robots = 0
            for robot in self.env.robots:
                if robot.success == True:
                    success_robots += 1
            if success_robots == self.robot_num:
                episode_success = True
        return episode_success,self.use_time

import numpy as np
from envs.utils.math_utils import get_weight, cal_distance


class APFPolicy():
    def __init__(self, args):
        self.robot_num = args.num_robots
        self.human_num = args.num_humans
        self.robot_radius = args.robot_radius
        self.human_radius = args.human_radius
        self.edge = args.for_edge
        self.max_speed = 1.5
        

    def act(self,states):
        actions = []
        states = np.array(states)
        # print(states[2][0])
        for no in range(self.robot_num):
            new_states = self.transform(states.copy(), no)
            # print(new_states[0])
            actions.append(self.calculate(new_states, no))

        return actions

    def calculate(self,states,no):
        # repulsive parameters
        b = 0.5
        c = b
        b0 = 2
        c0 = b0
        r_min = 2 * self.robot_radius
        r1 = self.edge - self.robot_radius
        r2 = self.edge + self.robot_radius
        r_max = 3 * self.edge
        ro_min = self.robot_radius + self.human_radius
        ro_max = 20 * ro_min
        # attractive parameters
        k = (1/self.edge) * (np.exp(self.edge/c) / (np.exp(self.edge/c)-np.exp(r_min/c)))
        # print(k)
        vf_gain = 0.5
        gamma = 1 #编队一致性参数

        v1 = np.zeros(2)
        v2 = np.zeros(2)
        vf = np.zeros(2)
        v = np.zeros(2)
        P = np.zeros(2)
        dist_vector = np.zeros(2)
        x, y, vx, vy, gx, gy = states[0]
        
        for state in (states[1:self.robot_num]):
            dist = 0
            dist = cal_distance(x, y, state[0], state[1])
            dist_vector = [(x-state[0])/dist, (y-state[1])/dist]
            # print('dist_vector',dist_vector)
            P = np.add(dist_vector, P)
            if r_min < dist < r1 or r2 < dist < r_max:
                v1 = np.add(v1, np.dot((b/c) * (np.exp(dist/c) / (np.exp(dist/c) - np.exp(r_min/c))**2) - k*dist, dist_vector))  # 原文还要补充通信权重

            vf = np.add(vf, [(gx-state[-2])-(x-state[0]), (gy-state[-1])-(y-state[1])])  # 编队一致性待完善

        for state in states[self.robot_num:]:
            dist = 0
            dist = cal_distance(x, y, state[0], state[1])
            dist_vector = [(x-state[0])/dist, (y-state[1])/dist]
            # print(vx,vy)
            delta_v = np.array((vx-state[2], vy-state[3]))
            delta_p = np.array((x-state[0], y-state[1]))
            v_io = np.linalg.norm(delta_v)  #这里的方向与论文相反
            vp = np.dot(delta_v, delta_p)
            if vp < 0 and ro_min < dist < ro_max:
                # print(np.dot((1 + np.exp(1/v_io)) * (b0/c0) * (np.exp(dist/c) / (np.exp(dist/c) - np.exp(ro_min/c))**2), dist_vector))
                v2 = np.add(v2, np.dot((1 + np.exp(1/v_io)) * (b0/c0) * (np.exp(dist/c) / (np.exp(dist/c) - np.exp(ro_min/c))**2), dist_vector))
        v2[0] = v = max(min(v2[0], self.max_speed), -self.max_speed)
        v2[1] = v = max(min(v2[1], self.max_speed), -self.max_speed)
        
        velocity = np.array((gx-x, gy-y))
        speed = np.linalg.norm(velocity)
        v_pref = velocity / speed if speed > 1 else velocity
        vc = np.add(np.add(v1, v2), v_pref)
        # I = np.eye(2)
        # v_nsb = np.add(vc.T, np.subtract(I, P.T @ P) @ vf.T)  # 不能用
        v = np.add(vc, np.dot(vf_gain, vf))
        return v

    def transform(self,states,no):
        temp_state = states[0].copy()
        states[0] = states[no]
        states[no] = temp_state
        return states



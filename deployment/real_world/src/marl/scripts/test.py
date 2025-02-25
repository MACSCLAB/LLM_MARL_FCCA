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


class RLrobot:
    def __init__(self):
        self.id = 2
        self.current_state = None


        # rospy.Subscriber(f"/robot1/odom", robot_Odometry, self.robot_callback)
        # rospy.Subscriber(f"/robot2/odom", robot_Odometry, self.robot_callback)
        rospy.Subscriber(f"/robot3/odom", robot_Odometry, partial(self.robot_callback, id=1), queue_size=1)

    def robot_callback(self, msg, id):
        px = msg.pos.x
        
        py = msg.pos.y
        print(px,py)

        self.current_state = np.array([px, py])
        

    def act(self):
        #define publisher
        #10Hz, 0.1s
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            rate.sleep()


if __name__ == "__main__":
    rospy.init_node('agent_policy_node')
    agent = RLrobot()
    time.sleep(1)
    agent.act()
    rospy.spin()
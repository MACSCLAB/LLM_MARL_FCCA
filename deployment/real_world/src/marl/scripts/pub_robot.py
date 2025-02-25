#!/usr/bin/env python3
import rospy
import numpy as np
from geometry_msgs.msg import Point, TwistStamped, PoseStamped, Twist
from marl.msg import robot_Odometry


'''具体发布的消息数据查看官方wiki：https://wiki.ros.org/vrpn_client_ros
robot_Odometry    
    geometry_msgs/Point pos
    geometry_msgs/Twist vel'''

def quaternion_multiply(q1, q2):
    # 四元数乘法
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.array([w, x, y, z])

def rotate_point(point, q):
    # 点的四元数表示
    p = np.array([0] + list(point))
    # 四元数共轭
    q_conj = np.array([q[0], -q[1], -q[2], -q[3]])
    # 旋转点
    p_rotated = quaternion_multiply(quaternion_multiply(q, p), q_conj)
    # 返回旋转后的向量部分
    return p_rotated[1:]


class Pub_robot:
    def __init__(self):
        '''收集相关机器人的刚体数据'''
        self.pre_pose = None
        self.robot_odom = robot_Odometry()
        self.robot_odom_pub = rospy.Publisher("/robot/Odometry",robot_Odometry,queue_size=10)
        self.robot_pose_subscribe = rospy.Subscriber("/Pose", PoseStamped, self.robot_pose_callback)
        self.id = int(rospy.get_param('~robot_id', '0'))

    def robot_pose_callback(self, msg):
        # self.robot_odom.pos.x = msg.pose.position.x + 1.8
        # self.robot_odom.pos.y = -msg.pose.position.z + 1.8
        if self.id == 0:
            self.robot_odom.pos.x = msg.pose.position.x
            self.robot_odom.pos.y = msg.pose.position.y
        elif self.id == 1:
            self.robot_odom.pos.x = msg.pose.position.x
            self.robot_odom.pos.y = msg.pose.position.y
        elif self.id == 2:
            self.robot_odom.pos.x = msg.pose.position.x
            self.robot_odom.pos.y = msg.pose.position.y
        else:
            raise NotImplementedError
        #查分计算当前机器人的速度信息
        if self.pre_pose is not None:
            dt = (msg.header.stamp - self.pre_pose.header.stamp).to_sec()
            if dt > 0.05:
                #caculate the velocity
                velocity = TwistStamped()
                velocity.header.stamp = rospy.Time.now()
                velocity.header.frame_id = msg.header.frame_id
                velocity.twist.linear.x = (msg.pose.position.x - self.pre_pose.pose.position.x) / dt
                velocity.twist.linear.y = (msg.pose.position.y - (self.pre_pose.pose.position.y)) / dt
                self.pre_pose = msg

        else:
            self.pre_pose = msg

    def robot_state_pub(self):
        rate = rospy.Rate(50)
        while not rospy.is_shutdown():
            self.robot_odom_pub.publish(self.robot_odom)
        rate.sleep()


if __name__ == "__main__":
    rospy.init_node('robot_state_pub')
    pub = Pub_robot()
    pub.robot_state_pub()
    # rospy.spin()
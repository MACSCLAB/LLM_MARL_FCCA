#!/usr/bin/env python3
import rospy
import numpy as np
from geometry_msgs.msg import Point, Twist, TwistStamped, PoseStamped
from marl.msg import Obstaclelist

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

class ObstacleAggregator:
    def __init__(self):
        # 初始化Obstaclelist类型的消息
        self.obstacle = Obstaclelist()
        self.obstacle.pos = []  # 初始化pos列表
        self.obstacle.vel = []  # 初始化vel列表
        self.num_obstacles = rospy.get_param('~num_obstacles', 3)
        self.pub = rospy.Publisher('/obstacle', Obstaclelist, queue_size=10)
        self.pre_ob_pose = []
        # Subscribers for pose and velocity
        for i in range(1, self.num_obstacles + 1):
            self.pre_ob_pose.append(None)
            rospy.Subscriber(f"/obstacle{i}/pose", PoseStamped, self.pose_callback, i-1)

    def pose_callback(self, msg, index):
        # 确保列表长度足够
        while len(self.obstacle.pos) <= index:
            self.obstacle.pos.append(Point())
        
        while len(self.obstacle.vel) <= index:
            self.obstacle.vel.append(Twist())
        
        # self.obstacle.pos[index].x =  msg.pose.position.x + 1.8
        # self.obstacle.pos[index].y =  -msg.pose.position.z + 1.8
        if index == 0:
            self.obstacle.pos[index].x =  msg.pose.position.x
            self.obstacle.pos[index].y =  msg.pose.position.y
        elif index == 1:
            self.obstacle.pos[index].x =  msg.pose.position.x
            self.obstacle.pos[index].y =  msg.pose.position.y
        else:
            self.obstacle.pos[index].x =  msg.pose.position.x
            self.obstacle.pos[index].y =  msg.pose.position.y
        if self.pre_ob_pose[index] is not None:
            dt = (msg.header.stamp - self.pre_ob_pose[index].header.stamp).to_sec()
            if dt > 0.05:
                #caculate the velocity
                velocity = TwistStamped()
                velocity.header.stamp = rospy.Time.now()
                velocity.header.frame_id = msg.header.frame_id
                velocity.twist.linear.x = (msg.pose.position.x - self.pre_ob_pose[index].pose.position.x) / dt
                velocity.twist.linear.y = (msg.pose.position.y - (self.pre_ob_pose[index].pose.position.y)) / dt
                self.obstacle.vel[index] = velocity.twist
                self.pre_ob_pose[index] = msg
        else:
            self.pre_ob_pose[index] = msg        
        self.publish_if_complete(index)


    def publish_if_complete(self, index):
        # 确保索引在列表的有效范围内
        if index < len(self.obstacle.pos) and index < len(self.obstacle.vel):
            # 检查是否所有数据都已收到
            if self.obstacle.pos[index] is not None and self.obstacle.vel[index] is not None:
                self.pub.publish(self.obstacle)
        else:
            # 如果索引超出范围，记录错误或警告，有助于调试和维护
            rospy.logwarn("Index {} is out of range. pos length: {}, vel length: {}".format(index, len(self.obstacle.pos), len(self.obstacle.vel)))

if __name__ == '__main__':
    rospy.init_node('obstacle_aggregator')
    agg = ObstacleAggregator()
    rospy.spin()
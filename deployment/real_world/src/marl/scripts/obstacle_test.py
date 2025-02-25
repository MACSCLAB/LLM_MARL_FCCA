#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import Point, Twist, PoseStamped
# from policy.mappo import *
from config.config import get_config 
# model = 'model'
def publish_obstacle():
    # 初始化ROS节点
    rospy.init_node('obstacle_publisher')
    
    # 创建发布器，发布到 /obstacle 话题
    pub = rospy.Publisher('/obstacle/pose', PoseStamped, queue_size=10)
    

    # 设置发布频率为100Hz
    rate = rospy.Rate(100)

    # 创建Point和Twist消息
    Pose = PoseStamped()

    while not rospy.is_shutdown():
        # 发布消息
        Pose.header.stamp = rospy.Time.now()
        pub.publish(Pose)
        rate.sleep()

if __name__ == '__main__':
    try:
        publish_obstacle()
    except rospy.ROSInterruptException:
        pass
#!/usr/bin/env python3
import rospy
import numpy as np
from std_msgs.msg import Float64

def PointPub(min_range=0.0,max_range=100.0):

    closepub = rospy.Publisher('/closest_point', Float64 , queue_size=10)
    farpub = rospy.Publisher('/furthest_point', Float64, queue_size=10)
    
    rate = rospy.Rate(10) # 10hz


    while not rospy.is_shutdown():
        closepub.publish(Float64(min_range))
        farpub.publish(Float64(max_range))
        rate.sleep()

#!/usr/bin/env python3
import rospy
import numpy as np
from sensor_msgs.msg import LaserScan as lscan
from PointPub import PointPub

def callback(data):
    #if not (np.isnan(data.angle_min) or np.isnan(data.angle_max)):
    #    print(f'Start angle of scan is {data.angle_min} and end angle is {data.angle_max}')
    ranges = data.ranges
    min_val = np.min(ranges)
    max_val = np.max(ranges)
    print(ranges.type)
    PointPub(min_range=min_val,max_range=max_val) # Publish maximum value of ranges on /furthest_point topic and minimum value on /closest_point topic
    

def ScanSub():
    rospy.init_node('ScanSub', anonymous=True)   # Initialize node
    rospy.Subscriber('/scan',lscan,callback) # Subscribe to topic

    rospy.spin() # Continue receiving messages till stopped

if __name__ == '__main__':
    ScanSub()





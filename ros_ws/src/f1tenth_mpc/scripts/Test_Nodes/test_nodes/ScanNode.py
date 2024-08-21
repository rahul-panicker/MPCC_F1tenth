#!/usr/bin/env python3
import rospy
import numpy as np
from sensor_msgs.msg import LaserScan as lscan
from std_msgs.msg import Header, Float64
from f1tenth_simulator.msg import scan_range

def PtPub(min_range=0.0,max_range=100.0):
    """
    This function creates 2 publishers that send messages on two separate topics, '/closest_point' and '/furthest_point'. As the names suggest, 
    these topics contain the closest and furthest points from the laserscan data 

    """
    closepub = rospy.Publisher('/closest_point', Float64 , queue_size=10)
    farpub = rospy.Publisher('/furthest_point', Float64, queue_size=10)
    
    rate = rospy.Rate(10) # 10hz

    if not (np.isnan(min_range) or np.isnan(max_range)):
        while not rospy.is_shutdown():
            closepub.publish(Float64(min_range))
            farpub.publish(Float64(max_range))
            rate.sleep()


def ScanPub(min_range=0.0,max_range=100.0):
    """
    This function defines a message of type scan_range with the fields, header, min_range and max_range and publishes said message on the '/scan_range'
    topic. 
    """

    pub = rospy.Publisher('/scan_range', scan_range, queue_size=10) # Create publisher object
    rate = rospy.Rate(10)

    if not (np.isnan(max_range) or np.isnan(min_range)):
        while not rospy.is_shutdown():
            scanMsg = scan_range()   # Create a message instance
            
            scanMsg.header.stamp = rospy.Time.now()  # Assign values to the message fields
            scanMsg.max_range = max_range  
            scanMsg.min_range = min_range
            pub.publish(scanMsg)
            rate.sleep()


def callback(data):

    ranges = data.ranges
    min_val = np.min(ranges)
    max_val = np.max(ranges)

    ScanPub(min_range=min_val,max_range=max_val) # Publish maximum value of ranges on /furthest_point topic and minimum value on /closest_point topic
    

def ScanSub():
    rospy.init_node('ScanSub', anonymous=True)   # Initialize node
    rospy.Subscriber('/scan',lscan,callback) # Subscribe to topic

    rospy.spin() # Continue receiving messages till stopped

if __name__ == '__main__':
    ScanSub()





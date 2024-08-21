#!/usr/bin/env python3
import rospy
import numpy as np
from sensor_msgs.msg import LaserScan as lscan
from f1tenth_simulator.msg import vert_dist

scan_data = [0, 0, 0]
Dt = 0
dt = 0
Dt1 = 0

def callback(data):
    global scan_data, Dt, Dt1, dt
    #if not (np.isnan(data.angle_min) or np.isnan(data.angle_max)):
    #    print(f'Start angle of scan is {data.angle_min} and end angle is {data.angle_max}')
    L = 0.5
    ranges = data.ranges
    del_theta = data.angle_increment
    theta0 = data.angle_min
    theta = -np.pi/4

    theta1 = -np.pi/2
    theta2 = theta
    theta3 = np.pi/2

    idx1 = int((theta1 - theta0) / del_theta)
    idx2 = int((theta2 - theta0) / del_theta)
    idx3 = int((theta3 - theta0) / del_theta)

    a = ranges[idx1]
    b = ranges[idx2]
    c = ranges[idx3]

    alpha = np.arctan(a*np.cos(abs(theta)) - b)/(a*np.sin(abs(theta)))
    Dt = a * np.cos(abs(alpha))
    dt = c * np.cos(abs(alpha))
    Dt1 = Dt + L * np.sin(abs(alpha))

    scan_data = [a, b, c]


def Pub():
    global scan_data, Dt, Dt1, dt
    new_pub = rospy.Publisher('/Wall_Dist', vert_dist, queue_size=10)
    DistMsg = vert_dist()

    while not rospy.is_shutdown():
        print(f'Distance to right wall {Dt}, Distance to left wall {dt}')
        DistMsg.Dt = Dt
        DistMsg.dt = dt
        DistMsg.Dt1 = Dt1
        new_pub.publish(DistMsg)
        rospy.sleep(0.1)


def ScanSub():
    rospy.init_node('ScanSub', anonymous=True)   # Initialize node
    rospy.Subscriber('/scan', lscan, callback)  # Subscribe to topic

    Pub()



if __name__ == '__main__':
    ScanSub()





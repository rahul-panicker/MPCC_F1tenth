#!/usr/bin/env python3

import rospy
import numpy as np
from ackermann_msgs.msg import AckermannDriveStamped as acker
from sensor_msgs.msg import LaserScan as lscan
from nav_msgs.msg import Odometry as odom
from std_msgs.msg import Bool as bool

class Braking:
    """
    class to handle Emergency braking
    """
    def __init__(self):
        # PUBLISHERS
        self.drive_pub = rospy.Publisher('/drive', acker, queue_size=10)  # Create a publisher to the /drive topic
        self.brake_pub = rospy.Publisher('/brake_bool', bool, queue_size=10)  # Create a publisher to the /brake_bool topic

        # SUBSCRIBERS
        self.ScanSub = rospy.Subscriber('/scan', lscan, self.ScanCallback)  # Create a subscriber to the /scan topic
        self.OdomSub = rospy.Subscriber('/odom', odom, self.OdomCallback)  # Create a subscriber to the /odom topic

        # OTHERS
        self.speed = 0.0

    def OdomCallback(self, odomdata):
        """
    Callback function for the ScanSub object created in Braking_Node()

    :Inputs:
        odomdata: Message Data of type nav_msgs/Odometry (http://docs.ros.org/en/noetic/api/nav_msgs/html/msg/Odometry.html)
        """
        self.speed = odomdata.twist.twist.linear.x

    def ScanCallback(self, scandata):
        """
        Inputs:
            scandata: Message Data of type scan_msgs/LaserScan (http://docs.ros.org/en/melodic/api/sensor_msgs/html/msg/LaserScan.html)

        """
        r = np.array(scandata.ranges)

        theta_0 = scandata.angle_min
        theta_inc = scandata.angle_increment
        theta = np.array([theta_0 + i * theta_inc for i in range(r.shape[0])])
        r_dot = self.speed * np.cos(theta)
        TTC = r/(np.maximum(-r_dot, 0))

        if np.any(TTC <= 0.5):
            brakeMsg = bool()
            ackerMsg = acker()

            brakeMsg.data = True
            ackerMsg.drive.speed = 0.0
            self.brake_pub.publish(brakeMsg)
            self.drive_pub.publish(ackerMsg)

    def drive(self, speed=0.0):
        """
        Function to drive the car at a specific speed and heading.

        :Inputs:
            v: desired velocity of the car (+ve for forward and -ve for reverse, 0 to stop.)

        """
        ackerMsg = acker()  # Create a message of type ackermann_msgs/AckermannDriveStamped
        ackerMsg.drive.speed = speed  # Assign speed value to

        rate = rospy.Rate(10)  # 10 Hz

        while not rospy.is_shutdown():
            self.drive_pub.publish(ackerMsg)
            rate.sleep()


def main():
    rospy.init_node('AEB_Node')
    breaking = Braking()
    rospy.spin()


if __name__ == '__main__':
    main()
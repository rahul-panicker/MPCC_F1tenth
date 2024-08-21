#!/usr/bin/env python3
from __future__ import print_function
import sys
import math
import numpy as np

# ROS Imports
import rospy
from sensor_msgs.msg import Image
from sensor_msgs.msg import LaserScan as lscan
from ackermann_msgs.msg import AckermannDriveStamped as acker
from ackermann_msgs.msg import AckermannDrive


class WallFollow:

    def __init__(self):
        # PID PARAMS
        self.kp = 1
        self.kd = 0.05
        self.ki = 0.02
        self.errors = [0.0, 0.0, 0.0]
        self.dt = 0.002

        # SUBSCRIBERS AND PUBLISHERS
        self.scan_sub = rospy.Subscriber('/scan', lscan, self.scan_callback)  # Subscribe to LIDAR

        # MISC VARIABLES
        self.d_r = 0.0
        self.d_l = 0.0
        self.ref_d = 0.75
        self.scan_msg = None

    def get_range(self, scan_data, angle):
        """
        Simple helper to return the corresponding range measurement at a given angle. Make sure you take care of NaNs and infs.

        Args:
            scan_data: A single message from the /scan topic
            angle: any angle such that angle_min < angle < angle_max of the LiDAR

        Returns:
            r: range measurement in meters at the given angle
        """
        theta0 = scan_data.angle_min
        del_theta = scan_data.angle_increment
        idx = int((angle - theta0)/del_theta)
        r = scan_data.ranges[idx]

        return r

    def get_error(self, scan_data, dist):
        """
        Calculates the error to the wall. Follow the wall to the left (going counter clockwise in the Levine loop).
        You potentially will need to use get_range()

        Args:
            scan_data: single range array from the LiDAR
            dist: desired distance to the wall

        Returns:
            error: calculated error
        """
        theta1 = -np.pi / 2
        theta2 = np.pi / 2

        theta = - 70 * np.pi/180   # Known angle for measuring alpha value

        a = self.get_range(scan_data=scan_data, angle=theta)
        b = self.get_range(scan_data=scan_data, angle=theta1)
        c = self.get_range(scan_data=scan_data, angle=theta2)

        alpha = np.arctan(a * np.cos(abs(theta)) - b) / (a * np.sin(abs(theta)))
        self.d_r = a * np.cos(abs(alpha))
        self.d_l = c * np.cos(abs(alpha))

        error = self.d_l - dist

        print(f'Left distance {self.d_l} right distance {self.d_r}')

        return error

    def pid_control(self, error):
        """
        Based on the calculated error, publish vehicle control

        Args:
            error: calculated error

        Returns:
            None
        """
        drive_pub = rospy.Publisher('/drive', acker, queue_size=10)  # Publish to drive topic

        d_error = (self.errors[2] - 2*self.errors[1] + self.errors[0])/(2*self.dt)
        I_error = (self.dt/2) * (self.errors[2] + 2*self.errors[1] + self.errors[0])

        angle = self.kp*error + self.kd*d_error + self.ki*I_error

        if 0 <= abs(angle) < 0.175:
            velocity = 2.5
        elif 0.175 <= abs(angle) < 0.350:
            velocity = 1.5
        else:
            velocity = 0.5

        drive_msg = acker()

        drive_msg.header.stamp = rospy.Time.now()
        drive_msg.header.frame_id = "laser"
        drive_msg.drive.steering_angle = angle
        drive_msg.drive.speed = velocity
        drive_pub.publish(drive_msg)

    def scan_callback(self, msg):
        """
        Callback function for LaserScan messages. Calculate the error and publish the drive message in this function.

        Args:
            msg: Incoming LaserScan message

        Returns:
            None
        """
        self.scan_msg = msg  # Store latest message

        # Calculate error
        error = self.get_error(msg, self.ref_d)

        # Update error buffer
        self.errors[0] = self.errors[1]
        self.errors[1] = self.errors[2]
        self.errors[2] = error

        # Drive the car using PID controller
        self.pid_control(error)


def main():
    rospy.init_node('WallFollow',anonymous=True)
    wf = WallFollow()
    rospy.spin()


if __name__ == '__main__':
    main()
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


# PID CONTROL PARAMS
error = 0
Dt = 0
dt = 0
errors = [0.0, 0.0, 0.0]
integral = 0.0


def getRange(data):
    """
    Member function that returns the distance from the car to the nearest wall

    :param:
        data - a single message from the /scan topic
    :return:
        a, b, c - Range readings at pi/4, 0 and pi radians respectively
    """
    global Dt, dt

    ranges = data.ranges
    del_theta = data.angle_increment
    theta0 = data.angle_min
    theta = -np.pi / 4

    theta1 = -np.pi / 2
    theta2 = theta
    theta3 = np.pi / 2

    idx1 = int((theta1 - theta0) / del_theta)
    idx2 = int((theta2 - theta0) / del_theta)
    idx3 = int((theta3 - theta0) / del_theta)

    a = ranges[idx1]
    b = ranges[idx2]
    c = ranges[idx3]

    alpha = np.arctan(a * np.cos(abs(theta)) - b) / (a * np.sin(abs(theta)))
    Dt = a * np.cos(abs(alpha))
    dt = c * np.cos(abs(alpha))


def pid_control(error):
    """
    Member function that implements a PID controller
    :Inputs:
        error - Current error

    """
    kp = 0.1
    kd = 0.0
    ki = 0.0
    # CALCULATE INPUT SIGNAL
    angle = kp * error # + self.kd * de + self.ki * ie


def drive(angle, velocity):

    drive_pub = rospy.Publisher('/drive', acker, queue_size=10)  # Publish to drive topic

    drive_msg = acker()
    drive_msg.header.stamp = rospy.Time.now()
    drive_msg.header.frame_id = "laser"
    drive_msg.drive.steering_angle = angle
    drive_msg.drive.speed = velocity
    drive_pub.publish(drive_msg)


def lidar_callback(data):
    """
    Function to calculate the error term to pass on to the PID Controller
    :Inputs:
        data - Laser Scan data from the '/scan' topic
    """
    global error, Dt, dt

    DESIRED_DISTANCE_RIGHT = 0.5  # meters

    getRange(data)
    error = DESIRED_DISTANCE_RIGHT - Dt


def main():
    # WALL FOLLOW PARAMS
    ANGLE_RANGE = 270  # Hokuyo 10LX has 270 degrees scan
    DESIRED_DISTANCE_LEFT = 0.5
    VELOCITY = 2.00  # meters per second
    CAR_LENGTH = 0.50  # Traxxas Rally is 20 inches or 0.5 meters

    rospy.init_node('WallFollowing',anonymous=True)

    lidar_sub = rospy.Subscriber('/scan', lscan, lidar_callback)  # Subscribe to LIDAR




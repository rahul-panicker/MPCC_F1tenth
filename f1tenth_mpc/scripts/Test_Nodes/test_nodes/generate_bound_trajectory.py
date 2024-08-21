#!/usr/bin/env python3


import numpy as np
import pandas as pd

# ROS Imports
import rospkg
import rospy
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, Point
from std_msgs.msg import Header
from Helper_scripts.trackParser import get_bounds


def generate_left_bound_msg(center_line, dist, t_samp=1.0):
    """
    Function to generate a ROS message of type nav_msgs/Path which represents the left bounds of the track

    :Input:
        center_line : numpy array containing coordinates of the track center line
        dist : Distance of track boundary from center line

    :Returns:
        leftboundMsg : ROS message of type nav_msgs/Path representing the left boundary of the track
    """

    leftboundMsg = Path()
    leftboundMsg.header = Header()
    leftboundMsg.header.frame_id = 'map'
    leftboundMsg.header.stamp = rospy.Time(0)

    left_bounds, _ = get_bounds(center_line,dist)

    x_left = left_bounds[:, 0]
    y_left = left_bounds[:, 1]

    timestamps = np.arange(0, len(x_left), 1)
    time_interp = np.arange(timestamps[0], timestamps[-1], t_samp)

    x_left_b = np.interp(time_interp, timestamps, x_left)
    y_left_b = np.interp(time_interp, timestamps, y_left)

    left_border_pts = np.column_stack([x_left_b, y_left_b])

    left_pose_stamped = PoseStamped()

    for i, point in enumerate(left_border_pts):
        left_pose_stamped.pose.position = Point(point[0], point[1], 0.0)
        left_pose_stamped.header.stamp = leftboundMsg.header.stamp + i * rospy.Duration(nsecs=int(t_samp*1e9))
        left_pose_stamped.header.frame_id = leftboundMsg.header.frame_id

        leftboundMsg.poses.append(left_pose_stamped)

    return leftboundMsg


def generate_right_bound_msg(center_line, dist, t_samp=1.0):
    """
    Function to generate a ROS message of type nav_msgs/Path which represents the right bounds of the track

    :Input:
        center_line : numpy array containing coordinates of the track center line
        dist : Distance of track boundary from center line

    :Returns:
        rightboundMsg : ROS message of type nav_msgs/Path representing the right boundary of the track
    """

    rightboundMsg = Path()
    rightboundMsg.header = Header()
    rightboundMsg.header.frame_id = 'map'
    rightboundMsg.header.stamp = rospy.Time(0)

    _, right_bounds = get_bounds(center_line, dist)

    x_right = right_bounds[:, 0]
    y_right = right_bounds[:, 1]

    timestamps = np.arange(0, len(x_right), 1)
    time_interp = np.arange(timestamps[0], timestamps[-1], t_samp)

    x_right_b = np.interp(time_interp, timestamps, x_right)
    y_right_b = np.interp(time_interp, timestamps, y_right)

    right_border_pts = np.column_stack([x_right_b,y_right_b])

    right_pose_stamped = PoseStamped()

    for i, point in enumerate(right_border_pts):

        right_pose_stamped.header.stamp = rightboundMsg.header.stamp + i * rospy.Duration(nsecs=int(t_samp*1e9))
        right_pose_stamped.header.frame_id = rightboundMsg.header.frame_id
        right_pose_stamped.pose.position = Point(point[0], point[1], 0.0)

        rightboundMsg.poses.append(right_pose_stamped)

    return rightboundMsg


def main():
    rospy.init_node('bounds_generator', anonymous=True)
    leftboundPub = rospy.Publisher('/left_bound', Path, queue_size=10)
    rightboundPub = rospy.Publisher('/right_bound', Path, queue_size=10)

    rospack = rospkg.RosPack()
    track = 'Silverstone'
    pkg_path = rospack.get_path('f1tenth_simulator')
    file_path = pkg_path + f'/scripts/Additional_maps/{track}/{track}_centerline.csv'

    df = pd.read_csv(file_path)
    center_line = df.iloc[0::2, :2].values
    center_line = np.vstack([center_line, center_line[:2, :]])
    d = df.iloc[0, 3]

    leftboundMsg = generate_left_bound_msg(center_line, d, t_samp=0.02)
    rightboundMsg = generate_right_bound_msg(center_line, d, t_samp=0.02)
    rate = rospy.Rate(50)

    while not rospy.is_shutdown():
        leftboundPub.publish(leftboundMsg)
        rightboundPub.publish(rightboundMsg)
        rate.sleep()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException as e:
        print(f"Error: {e}")




        # leftMarkerMsg = generate_point_marker_msg(leftboundMsg,id=2)
        # rightMarkerMsg = generate_point_marker_msg(rightboundMsg)
        #
        # leftMarkerPub = rospy.Publisher('/visualization_marker', Marker, queue_size=10)
        # rightMarkerPub = rospy.Publisher('/visualization_marker', Marker, queue_size=10)

        # rightboundPub.publish(rightboundMsg)
        # leftMarkerPub.publish(leftMarkerMsg)
        #

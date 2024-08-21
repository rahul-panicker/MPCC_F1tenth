#!/usr/bin/env python3

import numpy as np
import pandas as pd

# ROS Imports
import rospy
from nav_msgs.msg import Odometry
import rospkg
from std_msgs.msg import Float64
from geometry_msgs.msg import PoseWithCovariance as Pose
from tf.transformations import quaternion_from_euler
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point


class refPub:
    """
    Class describing the reference error and closest point on the reference trajectory from the current position of the
    car. All references are calculated w.r.t global (map) frame.
    """
    def __init__(self):
        # extracting centerline info from .csv file
        rospack = rospkg.RosPack()
        track = 'Silverstone'
        pkg_path = rospack.get_path('f1tenth_simulator')
        file_path = pkg_path + f'/scripts/Additional_maps/{track}/{track}_centerline.csv'
        df = pd.read_csv(file_path)

        # Initializing values
        self.error = 0.0
        self.x_filt = []
        self.y_filt = []
        self.x_vals = df.iloc[:, 0].values
        self.y_vals = df.iloc[:, 1].values

        # Initializing reference pose
        self.xref = 0.0
        self.yref = 0.0
        self.psiref = np.pi/2

        # Publishers and subscribers
        self.odomSub = rospy.Subscriber('/odom', Odometry, self.OdomCallback)
        self.errorPub = rospy.Publisher('/center_error', Float64, queue_size=10)
        self.PosePub = rospy.Publisher('/ref_pose',Pose, queue_size=10)
        self.markerPub = rospy.Publisher('/visualization_marker',Marker,queue_size=10)

    def OdomCallback(self,odom_data):
        """
        Callback function for subscriber to /odom topic
        """

        pos = odom_data.pose.pose.position

        self.x_filt = [x for x in self.x_vals if (pos.x-5) <= x <= (pos.x+5)]
        self.y_filt = [y for y in self.y_vals if (pos.y-5) <= y <= (pos.y+5)]\

        xvar = np.array(self.x_filt)
        yvar = np.array(self.y_filt)

        dist_vec = self.calculate_distances(pos)

        self.error = min(dist_vec)  # Minimum distance from a list of points
        idx = np.argmin(dist_vec)
        self.xref = xvar[idx]
        self.yref = yvar[idx]

        dx = xvar[idx+1]-xvar[idx]
        dy = yvar[idx+1]-yvar[idx]
        self.psiref = np.arctan2(dy, dx)

    def calculate_distances(self, pos):

        """
        Function to calculate the Euclidean distances to the points on the reference line that are close to the car's
        current position.
        """

        xvar = np.array(self.x_filt)
        yvar = np.array(self.y_filt)
        car_pos = np.array([pos.x, pos.y])
        dist_vec = []
        for x, y in zip(xvar, yvar):
            d = np.sqrt((car_pos[0] - x) ** 2 + (car_pos[1] - y) ** 2)
            dist_vec.append(d)

        return np.array(dist_vec)

    def Publish_and_visualize(self):
        """
        Member function to publish and visualize the calculated reference point on the center line
        """
        refPose = Pose()
        refMarker = Marker()
        point = Point()
        point.x = self.xref
        point.y = self.yref
        point.z = 0.0

        # Pose message
        refPose.pose.position.x = self.xref
        refPose.pose.position.y = self.yref
        refPose.pose.position.z = 0.0
        psi_ref_quat = quaternion_from_euler(0, 0, self.psiref)
        refPose.pose.orientation.x = psi_ref_quat[0]
        refPose.pose.orientation.y = psi_ref_quat[1]
        refPose.pose.orientation.z = psi_ref_quat[2]
        refPose.pose.orientation.w = psi_ref_quat[3]

        # Pose marker message
        refMarker.header.frame_id = 'map'
        refMarker.header.stamp = rospy.Time.now()
        refMarker.id = 0
        refMarker.ns = 'Reference_points'
        refMarker.type = Marker.POINTS
        refMarker.action = Marker.ADD
        refMarker.pose.orientation.w = 1.0
        refMarker.scale.x = 0.05
        refMarker.scale.y = 0.05
        refMarker.color.g = 1.0
        refMarker.color.a = 1.0
        refMarker.points.append(point)

        """Publish error value  and reference pose on respective topics"""
        self.errorPub.publish(self.error)
        self.PosePub.publish(refPose)
        self.markerPub.publish(refMarker)


if __name__ == '__main__':
    try:
        rospy.init_node('getRef', anonymous=True)
        ref = refPub()
        ref.Publish_and_visualize()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass



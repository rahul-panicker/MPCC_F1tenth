#!/usr/bin/env python3


import rospy
import rospkg
import numpy as np
import pandas as pd
from generate_vizMarkerMsg import generate_LineSegment
from visualization_msgs.msg import Marker


def interpolate_path(path, n):
    """
    Function to use linear interpolation to create more-fine grained path data.

    :Parameters:
        path - Nx2 numpy array containing the x and y coordinates of the reference path
        n    - number of points between 2 consecutive coordinates required
    """

    x = path[:, 0]
    y = path[:, 1]

    x_interp = []
    y_interp = []

    for i in range(len(x)-1):
        x_new = np.linspace(x[i], x[i+1], n)
        y_new = np.linspace(y[i], y[i+1], n)

        x_interp = np.append(x_interp, x_new[:-1])
        y_interp = np.append(y_interp, y_new[:-1])

    x_new_e = np.linspace(x[-1], x[0], n)
    y_new_e = np.linspace(y[-1], y[0], n)

    x_interp = np.append(x_interp, x_new_e[:-1])
    y_interp = np.append(y_interp, y_new_e[:-1])

    path_interp = np.column_stack([x_interp, y_interp])
    return path_interp


def main():

    rospack = rospkg.RosPack()
    track = 'Silverstone'
    pkg_path = rospack.get_path('f1tenth_simulator')
    file_path = pkg_path + f'/scripts/Additional_maps/{track}/{track}_centerline.csv'

    rospy.init_node('Reference_trajectory', anonymous=True)
    vizmarker = rospy.Publisher('/visualization_marker', Marker, queue_size=10)

    df = pd.read_csv(file_path)
    center_line = df.iloc[:, :2].values
    d = df.iloc[0, 3] - 0.35
    center_line_interp = interpolate_path(center_line, n=5)

    while not rospy.is_shutdown():

        refMsg = generate_LineSegment(center_line_interp[:2], id=1,colors=[1.0, 1.0, 0.0], namespace='Center_Line')
        vizmarker.publish(refMsg)


if __name__ == '__main__':
    main()

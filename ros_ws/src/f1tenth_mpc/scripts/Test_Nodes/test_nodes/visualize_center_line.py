#!/usr/bin/env python3

import rospy
from visualization_msgs.msg import Marker
from generate_vizMarkerMsg import generate_trajectoryMarkerMsg
from nav_msgs.msg import Path

global trajMsg


def CenterLineMarkerMsg(trajMsg):

    center_line_id = 0
    center_line_ns = 'Center_Line'
    markerMsg = generate_trajectoryMarkerMsg(trajMsg, center_line_id, center_line_ns)

    return markerMsg


def trajCallback(data):

    global trajMsg

    trajMsg = data


def main():

    global trajMsg

    rospy.init_node('Center_Line_Visualizer', anonymous=True)
    markerPub = rospy.Publisher('/visualization_marker', Marker, queue_size=10)
    trajSub = rospy.Subscriber('/ref_trajectory', Path, trajCallback)

    rospy.sleep(5)

    markerMsg = CenterLineMarkerMsg(trajMsg)
    rate = rospy.Rate(50)
    while not rospy.is_shutdown():
        markerPub.publish(markerMsg)
        rate.sleep()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass



# def visualize_center():
#     rospack = rospkg.RosPack()
#     track = 'Silverstone'
#     pkg_path = rospack.get_path('f1tenth_simulator')
#     file_path = pkg_path + f'/scripts/Additional_maps/{track}/{track}_centerline.csv'
#     df = pd.read_csv(file_path)
#
#     x_vals = df.iloc[:, 0].values
#     y_vals = df.iloc[:, 1].values
#
#     x_vals = np.append(x_vals, x_vals[0])
#     y_vals = np.append(y_vals, y_vals[0])
#
#     rospy.init_node('refGenNode', anonymous=True)
#     markerPub = rospy.Publisher('/visualization_marker', Marker, queue_size=10)
#
#     rate = rospy.Rate(1000)
#
#     lineMsg = Marker()
#     i = 0
#     while not rospy.is_shutdown():
#         lineMsg.header.frame_id = 'map'
#         lineMsg.header.stamp = rospy.Time.now()
#         lineMsg.id = 0
#         lineMsg.type = Marker.LINE_STRIP
#         lineMsg.action = Marker.ADD
#         lineMsg.pose.orientation.w = 1.0
#         lineMsg.scale.x = 0.05
#         lineMsg.ns = 'Center_line'
#
#         lineMsg.color.r = 1
#         lineMsg.color.a = 1
#         lineMsg.lifetime = rospy.Duration(0)
#
#         if i < len(x_vals)-1:
#             point1 = Point()
#             point2 = Point()
#             point1.x = x_vals[i]
#             point1.y = y_vals[i]
#             point2.x = x_vals[i+1]
#             point2.y = y_vals[i+1]
#             lineMsg.points.append(point1)
#             lineMsg.points.append(point2)
#             i = i+1
#         markerPub.publish(lineMsg)
#         rate.sleep()
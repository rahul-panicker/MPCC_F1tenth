#!/usr/bin/env python3


import rospy
from visualization_msgs.msg import Marker
from generate_vizMarkerMsg import generate_vizMarkerMsg
from nav_msgs.msg import Path


global leftboundMsg, rightboundMsg


def CenterLineMarkerMsg(leftboundMsg):

    left_bound_id = 1
    left_bound_ns = 'Left_bound'
    markerMsg = generate_vizMarkerMsg(leftboundMsg, left_bound_id, left_bound_ns, colors=[0.0, 0.0, 1.0])

    return markerMsg


def leftboundCallback(ldata):

    global leftboundMsg

    leftboundMsg = ldata


def main():

    global leftboundMsg

    rospy.init_node('Track_Bound_visualizer', anonymous=True)
    markerPub = rospy.Publisher('/visualization_marker', Marker, queue_size=10)
    leftboundSub = rospy.Subscriber('/left_bound', Path, leftboundCallback)

    rospy.sleep(2)

    markerMsg = CenterLineMarkerMsg(leftboundMsg)
    rate = rospy.Rate(50)
    while not rospy.is_shutdown():
        markerPub.publish(markerMsg)
        rate.sleep()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass

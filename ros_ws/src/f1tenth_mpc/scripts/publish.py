#!/usr/bin/env python3

import rospy
from ackermann_msgs.msg import AckermannDriveStamped
from std_msgs.msg import Header


def main():
    rospy.init_node('velocity_publisher', anonymous=True)
    pub = rospy.Publisher('/vesc/low_level/ackermann_cmd_mux/input/navigation', AckermannDriveStamped, queue_size=10)
    rate = rospy.Rate(10)  # 10hz

    while not rospy.is_shutdown():
        velocity_msg = AckermannDriveStamped()
        velocity_msg.header.stamp = rospy.Time.now()
        velocity_msg.drive.speed = 2.0  # Velocity in m/s
        velocity_msg.drive.acceleration = 0
        velocity_msg.drive.steering_angle = 0.0  # Steering angle in radians
        pub.publish(velocity_msg)
        rate.sleep()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass

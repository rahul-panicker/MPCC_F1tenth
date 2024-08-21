#!/usr/bin/env python3

import numpy as np

import rospy
import tf2_ros
from tf.transformations import quaternion_matrix


def get_transform(source_frame='base_link', target_frame='map'):

    tf_buffer = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tf_buffer)

    try:
        transform = tf_buffer.lookup_transform(target_frame, source_frame, rospy.Time(), rospy.Duration(1))
        t = transform.transform.translation
        q = transform.transform.rotation
        T = quaternion_matrix([q.x, q.y, q.z, q.w])
        T[:3, 3] = np.array([[t.x], [t.y], [t.z]]).reshape(3,)
        return T

    except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
        rospy.logwarn(f'Failed to find transformation: {e}')


def main():

    rospy.init_node('Read_Transforms', anonymous=True)
    get_transform()
    rospy.spin()


if __name__ == '__main__':
    main()


import rospy
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion
import numpy as np
from geometry_msgs.msg import PoseWithCovarianceStamped as pose
from geometry_msgs.msg import PoseStamped


class odom:
    def __init__(self):
        self.pose = None
        self.linear_vel = None
        self.angular_vel = None

        self.Odomsub = rospy.Subscriber('/odom', Odometry, self.odomCallback)

        self.or_hist = np.array([])
        self.init_pos = None
        self.init_linear_vel = None
        self.init_angular_vel = None
        self.init_flag = True

        rospy.sleep(0.5)

    def odomCallback(self, odom_data):

        _, _, psi = euler_from_quaternion([odom_data.pose.pose.orientation.x,
                                         odom_data.pose.pose.orientation.y,
                                         odom_data.pose.pose.orientation.z,
                                         odom_data.pose.pose.orientation.w])

        self.or_hist = np.append(self.or_hist, psi)

        if len(self.or_hist) > 1:
            self.or_hist = np.unwrap(self.or_hist)
        psi = self.or_hist[-1]

        self.pose = np.array([odom_data.pose.pose.position.x,
                              odom_data.pose.pose.position.y,
                              psi])

        self.linear_vel = np.array([odom_data.twist.twist.linear.x,
                                    odom_data.twist.twist.linear.y,
                                    odom_data.twist.twist.linear.z])

        self.angular_vel = np.array([odom_data.twist.twist.angular.x,
                                     odom_data.twist.twist.angular.y,
                                     odom_data.twist.twist.angular.z])

        if self.init_flag:
            self.init_pos = self.pose
            self.init_angular_vel = self.angular_vel
            self.init_linear_vel = self.linear_vel
            self.init_flag = False


class AMCL:
    def __init__(self):
        self.pose = None
        self.amclsub = rospy.Subscriber('/amcl_pose', pose, self.amclCallback)

        self.init_pos = None
        self.init_flag = True
        self.or_hist = np.array([])

        rospy.sleep(0.5)

    def amclCallback(self, amcl_data):
        car_pose = amcl_data.pose.pose
        X = car_pose.position.x
        Y = car_pose.position.y
        _, _, psi = euler_from_quaternion([car_pose.orientation.x,
                                           car_pose.orientation.y,
                                           car_pose.orientation.z,
                                           car_pose.orientation.w])
        self.or_hist = np.append(self.or_hist, psi)

        if len(self.or_hist) > 1:
            self.or_hist = np.unwrap(self.or_hist)
        psi = self.or_hist[-1]

        self.pose = np.array([X, Y, psi])
        if self.init_flag:
            self.init_pos = np.array([X, Y, psi])
            self.init_flag = False


class ParticleFilter:
    def __init__(self):
        self.pose = None
        self.filterSub = rospy.Subscriber('/pf/viz/inferred_pose', PoseStamped, self.filterCallback)

        self.init_pos = None
        self.init_flag = True
        self.or_hist = np.array([])

        rospy.sleep(0.5)

    def filterCallback(self, filter_data):
        car_pose = filter_data.pose
        X = car_pose.position.x
        Y = car_pose.position.y
        _, _, psi = euler_from_quaternion([car_pose.orientation.x,
                                           car_pose.orientation.y,
                                           car_pose.orientation.z,
                                           car_pose.orientation.w])
        self.or_hist = np.append(self.or_hist, psi)

        if len(self.or_hist) > 1:
            self.or_hist = np.unwrap(self.or_hist)
        psi = self.or_hist[-1]

        self.pose = np.array([X, Y, psi])
        if self.init_flag:
            self.init_pos = np.array([X, Y, psi])
            self.init_flag = False

# def get_transform(source_frame='base_link', target_frame='map'):
#
#     tf_buffer = tf2_ros.Buffer()
#     tf_listener = tf2_ros.TransformListener(tf_buffer)
#
#     try:
#         transform = tf_buffer.lookup_transform(target_frame, source_frame, rospy.Time(), rospy.Duration(1))
#         t = transform.transform.translation
#         q = transform.transform.rotation
#         T = quaternion_matrix([q.x, q.y, q.z, q.w])
#         T[:3, 3] = np.array([[t.x], [t.y], [t.z]]).reshape(3,)
#         return T
#
#     except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
#         rospy.logwarn(f'Failed to find transformation: {e}')


# if __name__ == '__main__':
#     rospy.init_node('State_Extractor', anonymous=True)
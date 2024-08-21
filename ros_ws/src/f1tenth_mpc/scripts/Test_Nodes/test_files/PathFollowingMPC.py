#!/usr/bin/env python3

import pandas as pd
import time
from casadi import *

# ROS imports
import rospkg
import rospy
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker
from ackermann_msgs.msg import AckermannDriveStamped

# Helper scripts
from Helper_scripts.referenceParser import get_reference_poses, create_ref_orientation
from Helper_scripts.generate_vizMarkerMsg import generate_LineSegment, generate_arrow_msg
from tf.transformations import euler_from_quaternion
from Helper_scripts.interpolate_path import interpolate_path
from Controller import acados_settings, set_init_guess, generate_init_guess, warm_start
from Helper_scripts.get_boundary import get_bounds
from Helper_scripts.RCDriver import generate_drive_msg
from Helper_scripts.Models import CarModel


class odom:
    def __init__(self):
        self.pose = np.zeros([3, 1])
        self.Odomsub = rospy.Subscriber('/odom', Odometry, self.odomCallback)
        self.init_pos = None
        self.init_pos_flag = True

    def odomCallback(self, odom_data):

        _,_,psi = euler_from_quaternion([odom_data.pose.pose.orientation.x,
                                         odom_data.pose.pose.orientation.y,
                                         odom_data.pose.pose.orientation.z,
                                         odom_data.pose.pose.orientation.w])

        self.pose = np.array([odom_data.pose.pose.position.x,
                              odom_data.pose.pose.position.y,
                              psi
                             ])

        if self.init_pos_flag:
            self.init_pos = self.pose
            self.init_pos_flag = False


def main():
    """
    Main Function
    """

    rospy.init_node('Path_Following_MPC', anonymous=True)

    drivePub = rospy.Publisher('/drive', AckermannDriveStamped, queue_size=10)

    refmarkerPub = rospy.Publisher('/visualization_marker', Marker, queue_size=10)
    # leftboundPub = rospy.Publisher('/visualization_marker', Marker, queue_size=10)
    # rightboundPub = rospy.Publisher('/visualization_marker', Marker, queue_size=10)
    pred_visualizer = rospy.Publisher('/visualization_marker', Marker, queue_size=10)

    rospack = rospkg.RosPack()
    track = 'Silverstone'
    pkg_path = rospack.get_path('f1tenth_simulator')
    file_path = pkg_path + f'/scripts/Additional_maps/{track}/{track}_centerline.csv'

    df = pd.read_csv(file_path)
    center_line = df.iloc[:, :2].values
    d = df.iloc[0, 3] - 0.35

    center_line_interp = interpolate_path(center_line, n=5)
    orientations = create_ref_orientation(center_line_interp)

    Tf = 1
    N = 20
    tcomp_max = 0
    tcomp_sum = 0
    constraint, model, acados_solver = acados_settings(Tf=Tf, N=N)

    car_odometry = odom()
    car_model = CarModel()

    speed = 3.0
    nx = 3
    nu = 2
    i = 0

    opt_xi = []
    opt_ui = []

    opt_x = []
    opt_u = []

    rospy.sleep(0.5)

    x0 = car_odometry.init_pos
    init_x, init_u = generate_init_guess(x0, N, speed)

    rate = rospy.Rate(50)
    while not rospy.is_shutdown():
        car_pose = car_odometry.pose  # Get current state
        x_ref = get_reference_poses(center_line_interp, car_pose, orientations, N+2)  # Get references
        # track_left, track_right = get_bounds(x_ref[:, :2], d)  # Get boundaries

        xr = car_pose
        for j in range(N):
            u_ref = np.array([x_ref[j, 2], speed])
            yref = np.concatenate((x_ref[j, :], u_ref))

            # Set reference for each shooting node
            acados_solver.set(j, 'yref', yref)

            if j == 0:
                # Set initial state at first shooting node at every iteration of MPC loop
                acados_solver.set(0, 'lbx', car_pose)
                acados_solver.set(0, 'ubx', car_pose)
            else:
                # Set track bounds as upper and lower limits of state bounds for each shooting node
                pass
                # lbx = track_left[j, :]
                # ubx = track_right[j, :]
                # lbx = np.array([-1000, -1000])
                # ubx = np.array([1500, 1500])
                # acados_solver.constraints_set(j, 'lbx', lbx)
                # acados_solver.constraints_set(j, 'ubx', ubx)

        # Set reference at final shooting node
        yref_N = x_ref[N, :]
        acados_solver.set(N, "yref", yref_N)

        # Set constraints at final shooting node
        # dist_e = np.sqrt((car_pose[0] - yref_N[0])**2 + (car_pose[1] - yref_N[1])**2)
        # model.con_h_expr = dist_e
        # acados_solver.constraints_set(N, 'uh', np.array([d]))
        # acados_solver.constraints_set(N, 'lh', np.array([0]))

        # lbx_N = track_left[N, :]
        # ubx_N = track_right[N, :]
        # lbx_N = np.array([-1500, -1500])
        # ubx_N = np.array([1500, 1500])
        # acados_solver.set(N, "lbx", lbx_N)
        # acados_solver.set(N, "ubx", ubx_N)

        # Set solver initial guess
        if i == 0:
            set_init_guess(acados_solver, init_x, init_u, N)
        else:
            init_x, init_u = warm_start(opt_xi, opt_ui, nx, nu)
            # init_x = np.vstack((opt_xi[1:], opt_xi[-1, :]))
            # init_u = np.vstack((opt_ui[1:], np.zeros(nu)))
            set_init_guess(acados_solver, init_x, init_u, N)

        # solve ocp
        t = time.time()

        status = acados_solver.solve()
        # acados_solver.print_statistics()
        if status != 0:
            print("acados returned status {} in closed loop iteration {}.".format(status, i))
            # acados_solver.print_statistics()
            print('reference:', x_ref)
            print('Terminal reference:', yref_N)
            # print('Reference input:', u_ref)
            # print(opt_ui)
            # print(opt_xi)
            break

        else:
            print('Solution found')
            acados_solver.print_statistics()

        opt_xi = []
        opt_ui = []

        elapsed = time.time() - t

        tcomp_sum += elapsed
        if elapsed > tcomp_max:
            tcomp_max = elapsed

        # get solution
        for k in range(N+1):
            optx_k = acados_solver.get(k, "x")
            opt_xi = np.concatenate((opt_xi, optx_k))
        opt_xi = opt_xi.reshape(N+1, nx)
        opt_x = np.append(opt_x, opt_xi).reshape((i+1)*(N+1), nx)

        for k in range(N):
            optu_k = acados_solver.get(k, "u")
            opt_ui = np.concatenate((opt_ui, optu_k))
        opt_ui = opt_ui.reshape(N, nu)
        opt_u = np.append(opt_u, opt_ui).reshape((i+1)*N, nu)

        heading = opt_ui[0, 0]
        vel = opt_ui[0, 1]
        # vel = 2.0

        driveMsg = generate_drive_msg(vel, heading)

        drivePub.publish(driveMsg)
        # refmarkerMsg = generate_LineSegment(x_ref[:, :2], id=4)

        predVisMsg = generate_LineSegment(opt_xi[:, :2], id=5, colors=[1.0, 0.0, 0.0], scale=[0.05, 0.05, 0.05])
        terminalState = generate_arrow_msg(pose=yref_N[:3], id=6, colors= [0.0, 1.0, 0.0])

        # leftboundMsg = generate_LineSegment(track_left, id=6, colors=[1.0, 0.0, 0.0])
        # rightboundMsg = generate_LineSegment(track_right, id=7, colors=[1.0, 0.0, 0.0])

        refmarkerPub.publish(terminalState)
        pred_visualizer.publish(predVisMsg)
        # leftboundPub.publish(leftboundMsg)
        # rightboundPub.publish(rightboundMsg)

        i += 1
        # rate.sleep()

    # Parameterize the center line w.r.t it's arc length
    # x_s, y_s = arc_length_parameterize(center_line)

    # s = np.arange(0, 200, 0.01)
    # plt.plot(x_s(s), y_s(s),'.')
    # plt.show()
    # Tf = 1
    # N = 50
    # T = 10.00
    # N_sim = int(T*N/Tf)


if __name__ == '__main__':
        main()

        # refmarkerPub = rospy.Publisher('/visualization_marker', Marker, queue_size=10)
        # carMarkerPub = rospy.Publisher('/visualization_marker', Marker, queue_size=10)

        # refmarkerMsg = generate_PointMarkerMsg(x_ref[0], x_ref[1],id=4)
        # carmarkerMsg = generate_PointMarkerMsg(position[0], position[1], id=5)
        # refmarkerPub.publish(refmarkerMsg)
        # carMarkerPub.publish(carmarkerMsg)

# class Main:
#     def __init__(self, centerline, dist):
#
#         # Initialize variables
#         self.x = []
#         self.x_ref = []
#         self.timesteps = []
#         self.centerline = centerline
#         self.d = dist
#         self.track_left = []
#         self.track_right = []
#
#         # Subscribers and Publishers
#         refSub = rospy.Subscriber('/ref_trajectory', Path, self.refCallback)
#
#     def refCallback(self,trajectory):
#         """
#         Function to process the reference trajectory for MPC implementation
#         """
#         X_ref = []
#         Y_ref = []
#         Psi_ref = []
#
#         for pose_stamped in trajectory.poses:
#             X_ref = np.append(X_ref, pose_stamped.pose.position.x)
#             Y_ref = np.append(Y_ref, pose_stamped.pose.position.y)
#
#             _,_,psi = euler_from_quaternion([pose_stamped.pose.orientation.x,
#                                              pose_stamped.pose.orientation.y,
#                                              pose_stamped.pose.orientation.z,
#                                              pose_stamped.pose.orientation.w])
#             Psi_ref = np.append(Psi_ref, psi)
#             self.timesteps = np.append(self.timesteps, pose_stamped.header.stamp.to_sec())
#
#         self.x_ref = np.vstack([X_ref,Y_ref, Psi_ref])
#
#     def calculate_track_bounds(self):
#
#         left_bound, right_bound = get_bounds(self.centerline, self.d)
#
#         t = np.arange(0, len(left_bound[:, 0]), 1.0)
#
#         x_left = np.interp(self.timesteps, t, left_bound[:, 0])
#         y_left = np.interp(self.timesteps, t, left_bound[:, 1])
#
#         x_right = np.interp(self.timesteps, t, right_bound[:, 0])
#         y_right = np.interp(self.timesteps, t, right_bound[:, 1])
#
#         self.track_left = np.column_stack([x_left.T, y_left.T])
#         self.track_right = np.column_stack([x_right.T, y_right.T])
#
#     def odomCallback(self, odometry):
#
#         """
#         Function to process the odometry data from the simulation
#         """
#
#         odom_data = odometry
#
#
# def main():
#
#     rospy.init_node('Main_Node', anonymous=True)
#
#     rospack = rospkg.RosPack()
#     track = 'Silverstone'
#     pkg_path = rospack.get_path('f1tenth_simulator')
#     file_path = pkg_path + f'/scripts/Additional_maps/{track}/{track}_centerline.csv'
#
#     df = pd.read_csv(file_path)
#     center_line = df.iloc[0::2, :2].values
#
#     d = df.iloc[0, 3]
#     center_line = np.vstack([center_line, center_line[:2, :]])
#
#     main_obj = Main(centerline=center_line, dist=d)
#
#     while main_obj.track_left == [] and main_obj.track_right == []:
#         main_obj.calculate_track_bounds()
#         print(main_obj.track_left)
#
#     rospy.spin()
    # rate = rospy.Rate(10)
    # while not rospy.is_shutdown():
    #
    #     rate.sleep()
    #




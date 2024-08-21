#!/usr/bin/env python3


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.ticker import MultipleLocator
from referenceParser import *
import math
from GetStates import *


def plot_optimal_input_trajectories(opt_u, t_samp):

    N = len(opt_u)
    steer_ang = opt_u[:, 0]
    v = opt_u[:, 1]
    v_theta = opt_u[:, 2]
    labels = ['steering angle', 'velocity', 'projected velocity']
    t = np.linspace(0, N*t_samp, N)
    plt.step(t, steer_ang, where='post')
    plt.step(t, v, where='post')
    plt.step(t, v_theta, where='post')
    plt.xlabel('Time')
    plt.ylabel('Inputs')
    plt.legend(labels)
    plt.grid(True, linewidth=1.0, linestyle='dotted')
    plt.xticks(np.arange(0, (N+1)*t_samp, t_samp))
    plt.show()


def compare_sols(sol1, sol2):
    t_ode = sol1.t
    x1 = sol1.y[0, :]
    x2 = sol2.y[0, :]

    y1 = sol1.y[1, :]
    y2 = sol2.y[1, :]

    str1 = sol1.y[2, :]
    str2 = sol2.y[2, :]

    vel1 = sol1.y[3, :]
    vel2 = sol2.y[3, :]

    o1 = sol1.y[4, :]
    o2 = sol2.y[4, :]

    fig, axs = plt.subplots(5, 1, figsize=(10, 8))

    # evolution of X-coordinate
    axs[0].plot(t_ode, x1, label='Model with slip')
    axs[0].plot(t_ode, x2, label='Recalculated slip')
    axs[0].set_title('X-coordinate')
    axs[0].legend()

    # evolution of Y-coordinate
    axs[1].plot(t_ode, y1, label='Model with slip')
    axs[1].plot(t_ode, y2, label='Recalculated slip')
    axs[1].set_title('Y-coordinate')
    axs[1].legend()

    # evolution of orientation
    axs[2].plot(t_ode, o1, label='Model with slip')
    axs[2].plot(t_ode, o2, label='Recalculated slip')
    axs[2].set_title('Orientation')
    axs[2].legend()

    # evolution of steering angle
    axs[3].plot(t_ode, vel1, label='Model with slip')
    axs[3].plot(t_ode, vel2, label='Recalculated slip')
    axs[3].set_title('Velocity')
    axs[3].legend()

    # evolution of velocity
    axs[4].plot(t_ode, str1, label='Model with slip')
    axs[4].plot(t_ode, str2, '-.', label='Recalculated slip')
    axs[4].set_title('Steering anlge')
    axs[4].legend()

    plt.tight_layout()

    fig2, ax = plt.subplots()
    line1, = ax.plot([], [], 'o', label='Model with slip')
    line2, = ax.plot([], [], 'o', label='Recalculated slip')

    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Car Position Over Time')
    ax.legend()

    def update(frame):
        line1.set_data(x1[:frame], y1[:frame])
        line2.set_data(x2[:frame], y2[:frame])
        return line1, line2

    ani = FuncAnimation(fig, update, frames=len(t_ode), blit=True)
    # ani.save('Model_Comparison.gif', writer='pillow')
    plt.show()


def plot_ode_states(sol, nx, labels):

    t = sol.t
    fig, axs = plt.subplots(nx, 1, figsize=(20, 8))

    for i in range(nx):
        axs[i].plot(t, sol.y[i, :], label=labels[i])
        axs[i].legend()

    plt.tight_layout()
    plt.legend()
    plt.show()


def plot_sim_trajectories(state_traj, ref_traj, t_samp):

    t = np.linspace(0, len(ref_traj)*t_samp, len(ref_traj))
    X_traj = state_traj[:, 0]
    Y_traj = state_traj[:, 1]
    delta_traj = state_traj[:, 2]
    v_traj = state_traj[:, 3]
    psi_traj = state_traj[:, 4]
    w_traj = state_traj[:, 5]
    beta_traj = state_traj[:, 6]
    theta_traj = state_traj[:, 7]

    X_ref_traj = ref_traj[:, 0]
    Y_ref_traj = ref_traj[:, 1]
    psi_ref_traj = ref_traj[:, 2]
    theta_ref_traj = ref_traj[:, 3]

    fig, axs = plt.subplots(4,1)

    axs[0].plot(t, X_traj[:-1], label='X-coordinate')
    axs[0].plot(t, X_ref_traj, label='X-coordinate reference')
    axs[0].legend()

    axs[1].plot(t, Y_traj[:-1], label='Y-coordinate')
    axs[1].plot(t, Y_ref_traj, label='Y-coordinate reference')
    axs[1].legend()

    axs[2].plot(t, psi_traj[:-1], label='Orientation')
    axs[2].plot(t, psi_ref_traj, label='Reference orientation')
    axs[2].legend()

    axs[3].plot(t, theta_traj[:-1], label='Track length')
    axs[3].plot(t, theta_ref_traj, label='Reference track length')
    axs[3].legend()



    fig = plt.figure()
    plt.plot(X_traj, Y_traj, label='Car position')
    plt.plot(X_ref_traj, Y_ref_traj, label='Reference position')
    plt.legend()

    plt.show()


def plot_traj(x_traj, u_traj, ref, iteration, t_samp):

    t = np.arange(0, x_traj.shape[0]) * t_samp
    fig, axs = plt.subplots(8,1, figsize=(20, 8))
    axs[0].plot(t, x_traj[:, 0], label='X')
    axs[0].plot(t, ref[:, 0], label='X_ref')
    axs[1].plot(t, x_traj[:, 1], label='Y')
    axs[1].plot(t, ref[:, 0], label='Y_ref')
    axs[2].plot(t, x_traj[:, 2], label='delta')
    axs[3].plot(t, x_traj[:, 3], label='vel')
    axs[4].plot(t, x_traj[:, 4], label='psi')
    axs[4].plot(t, ref[:, 0], label='psi_ref')
    axs[5].plot(t, x_traj[:, 5], label='w')
    axs[6].plot(t, x_traj[:, 6], label='beta')
    axs[7].plot(t, x_traj[:, 7], label='theta')

    u_fig, u_axs = plt.subplots(3, 1)
    u_axs[0].plot(t[:-1], u_traj[:, 0], label='v_delta')
    u_axs[1].plot(t[:-1], u_traj[:, 1], label='a_long')
    u_axs[2].plot(t[:-1], u_traj[:, 2], label='v_theta')

    fig.suptitle(f"Prediction at OCP iteration {iteration}")


def analysis_plot(opt_xi, opt_ui, ref, ec_ar, el_ar, iteration, N=20, Tf=1):

    t_samp = Tf / N
    t_pred = np.arange(0.0, t_samp * (N + 1), t_samp)
    t_pred_u = np.arange(0.0, t_samp * N, t_samp)

    plt.figure()
    plt.plot(opt_xi[:, 0], opt_xi[:, 1], ".", label="Prediction")
    plt.plot(ref[:, 0], ref[:, 1], ".", label="Reference")
    plt.title(f"Prediction vs reference positions: Iteration {iteration}")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.legend()
    plt.grid(True)

    # fig, ax = plt.subplots(2, 1)
    # fig.suptitle(f"Error vs Time at iteration {iteration}")

    # ax[0].plot(t_pred, ec_ar, ".", label="Ec")
    # ax[0].set_xlabel("Time (Prediction Horizon)")
    # ax[0].set_ylabel("Ec")
    # ax[0].grid(True, which='both')
    # ax[0].xaxis.set_minor_locator(MultipleLocator(t_samp))
    # ax[0].legend()
    #
    # ax[1].plot(t_pred, el_ar, ".", label="El")
    # ax[1].set_xlabel("Time (Prediction Horizon)")
    # ax[1].set_ylabel("El")
    # ax[1].grid(True, which='both')
    # ax[1].xaxis.set_minor_locator(MultipleLocator(t_samp))
    # ax[1].legend()
    #
    # fig, axs = plt.subplots(3, 1)
    # fig.suptitle(f"Inputs vs Time at iteration {iteration}")
    #
    # axs[0].plot(t_pred_u, opt_ui[:, 0], ".")
    # axs[0].set_xlabel("Time (Prediction Horizon)")
    # axs[0].set_ylabel("Steering angle (Rad)")
    # axs[0].grid(True, which='both')
    # axs[0].xaxis.set_minor_locator(MultipleLocator(t_samp))
    #
    # axs[1].plot(t_pred_u, opt_ui[:, 1], ".")
    # axs[1].set_xlabel("Time (Prediction Horizon)")
    # axs[1].set_ylabel("Velocity (m/s)")
    # axs[1].grid(True)
    # axs[1].grid(True, which='both')
    # axs[1].xaxis.set_minor_locator(MultipleLocator(t_samp))
    #
    # axs[2].plot(t_pred_u, opt_ui[:, 2], ".")
    # axs[2].set_xlabel("Time (Prediction Horizon)")
    # axs[2].set_ylabel("Virtual velocity (m/s)")
    # axs[2].grid(True)
    # axs[2].grid(True, which='both')
    # axs[2].xaxis.set_minor_locator(MultipleLocator(t_samp))


def plot_track():
    center_line = get_reference_path_from_file(track='Monza')

    # # Get the car's starting point
    # car_odometry = odom()
    # start_pose = car_odometry.init_pos
    # track_start_pos, _ = get_closest_pose(center_line, start_pose[:2])
    #
    # # reformulate the center line coordinates so that the track starts from the closest point to the car
    # center_line = reformulate_path(center_line, track_start_pos)

    # Close loop
    center_line = np.vstack((center_line, center_line[0, :]))

    # Parameterize and interpolate reference path
    param_center_line = parameterize_path(center_line)

    center_line_x = param_center_line.x(param_center_line.arc_lengths)
    center_line_y = param_center_line.y(param_center_line.arc_lengths)
    ref_orientations = param_center_line.psi(param_center_line.arc_lengths)
    # center_line = get_reference_path_from_file()
    # dy_dx = np.gradient(center_line[:, 1], center_line[:, 0])
    # ref_orientations = np.arctan2(dy_dx, 1)
    #
    # center_line_x = center_line[:, 0]
    # center_line_y = center_line[:, 1]

    # Calculate the direction components (u, v) from the angles
    u = np.cos(ref_orientations)
    v = np.sin(ref_orientations)

    # Plot the center line
    plt.plot(center_line_x, center_line_y, color='black')

    # Plot the arrows representing tangent directions
    plt.quiver(center_line_x, center_line_y, u, v, scale=50)  # Adjust the scale as needed for arrow length
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Tangent Directions along Center Line')
    plt.grid(True)
    plt.axis('equal')  # Ensure equal aspect ratio

    # # Calculate the tangent directions between successive points
    # delta_x = np.diff(center_line_x)
    # delta_y = np.diff(center_line_y)
    # delta_x_f = center_line_x[-1] - center_line_x[0]
    # delta_y_f = center_line_y[-1] - center_line_y[0]
    # tangent_angles = np.arctan2(delta_y, delta_x)
    # angle_f = np.arctan2(delta_y_f, delta_x_f)
    # tangent_angles = np.append(tangent_angles, angle_f)
    #
    # # Calculate the change in angle between the tangent direction and the orientation
    # delta_angles = np.abs(ref_orientations - tangent_angles)
    # delta_angles = np.where(delta_angles > np.pi, 2 * np.pi - delta_angles,
    #                         delta_angles)  # Ensure angles are within [0, π]
    #
    # # Adjust the direction to ensure consistent forward-facing arrows
    # u_adjusted = np.where(delta_angles > np.pi / 2, -u, u)
    # v_adjusted = np.where(delta_angles > np.pi / 2, -v, v)
    #
    # # Plot the center line
    # plt.plot(center_line_x, center_line_y, color='black')
    #
    # # Plot the arrows representing tangent directions
    # plt.quiver(center_line_x[:-1], center_line_y[:-1], u_adjusted, v_adjusted,
    #            scale=50)  # Adjust the scale as needed for arrow length
    # plt.xlabel('X')
    # plt.ylabel('Y')
    # plt.title('Tangent Directions along Center Line')
    # plt.grid(True)
    # plt.axis('equal')  # Ensure equal aspect ratio
    # plt.show()

    plt.show()


if __name__ == '__main__':
    plot_track()

    # rospy.init_node('Visualize_pts', anonymous=True)
    # rospack = rospkg.RosPack()
    # track = 'Silverstone'
    # pkg_path = rospack.get_path('f1tenth_simulator')
    # file_path = pkg_path + f'/scripts/Additional_maps/{track}/{track}_centerline.csv'
    #
    # df = pd.read_csv(file_path)
    # center_line = df.iloc[:, :2].values
    # car = odom()
    # rospy.sleep(0.5)
    # start_point = car.init_pos

    # visualize_parameter_path(center_line, start_point)

    # visualize_center_line_pts(center_line)
    # center_line_interp = interpolate_path(center_line, n=5)
    # N = 20
    # traj, vel = create_traj_from_path(center_line, start_point, interpolate=True, ts_interp=0.05)
    # speed = vel * np.ones(N)
    # visualize_referece_traj_N(traj, N, speed)
    # visualize(trajectory=traj, car=car)


# class odom:
#     def __init__(self):
#         self.pose = np.zeros([3, 1])
#         self.Odomsub = rospy.Subscriber('/odom', Odometry, self.odomCallback)
#         self.init_pos = None
#         self.init_pos_flag = True
#
#     def odomCallback(self, odom_data):
#
#         _,_,psi = euler_from_quaternion([odom_data.pose.pose.orientation.x,
#                                          odom_data.pose.pose.orientation.y,
#                                          odom_data.pose.pose.orientation.z,
#                                          odom_data.pose.pose.orientation.w])
#
#         self.pose = np.array([odom_data.pose.pose.position.x,
#                               odom_data.pose.pose.position.y,
#                               psi
#                              ])
#
#         if self.init_pos_flag:
#             self.init_pos = self.pose
#             self.init_pos_flag = False
#
#
# def visualize(car, path=None, trajectory=None):
#
#     closestPtPub = rospy.Publisher('/visualization_marker', Marker, queue_size=10)
#     currentPtPub = rospy.Publisher('/visualization_marker', Marker, queue_size=10)
#
#     i = 0
#     N = 10
#     speed = np.ones(N)
#     rate = rospy.Rate(20)
#     while not rospy.is_shutdown():
#
#         x = car.pose
#         currentPtMsg = generate_PointMarkerMsg(x[0], x[1], id=2, colors=[1.0, 1.0, 0.0])
#
#         if path is not None:
#             x_close = get_closest_pose(ref_path=path, car_position=x[:2])
#             closePtMsg = generate_PointMarkerMsg(x_close[0], x_close[1], id=1, colors=[0.0, 1.0, 1.0])
#             closestPtPub.publish(closePtMsg)
#
#         if trajectory is not None:
#             x_close_line, _ = get_ref(i, N, speed, trajectory, const_vel=True)
#             close_lineMsg = generate_LineSegment(x_close_line[:,:2], id=1, colors=[1.0, 0.0, 1.0])
#             closestPtPub.publish(close_lineMsg)
#         currentPtPub.publish(currentPtMsg)
#         i += 1
#         rate.sleep()
#
#
# def visualize_center_line_pts(center_line):
#
#     visualizerPub = rospy.Publisher('/visualization_marker', Marker, queue_size=10)
#     x = center_line[:, 0]
#     y = center_line[:, 1]
#
#     x_neq = []
#     y_neq = []
#
#     dist = round(np.sqrt((x[1] - x[0]) ** 2 + (y[1] - y[0]) ** 2), 2)
#     for i in range(len(center_line)):
#         if i < len(center_line)-1:
#             if round(np.sqrt((x[i+1] - x[i])**2 + (y[i+1] - y[i])**2), 2) != dist:
#                 x_neq = np.append(x_neq, x[i])
#                 y_neq = np.append(y_neq, y[i])
#         else:
#             if round(np.sqrt((x[0] - x[i])**2 + (y[0] - y[i])**2), 2) != dist:
#                 x_neq = np.append(x_neq, x[i])
#                 y_neq = np.append(y_neq, y[i])
#
#     center_line_neq = np.column_stack([x_neq, y_neq])
#     centerMsg = generate_LineSegment(center_line_neq, id=1, colors=[1.0, 0.0, 0.0])
#
#     while not rospy.is_shutdown():
#         visualizerPub.publish(centerMsg)
#
#
# def visualize_referece_traj_N(traj, N, speed):
#     visualizerPub = rospy.Publisher('/visualization_marker', Marker, queue_size=10)
#
#     i = 0
#     rate = rospy.Rate(50)
#     while not rospy.is_shutdown():
#         if i % len(traj) == 0:
#             i=0
#
#         x_ref, u_ref = get_ref(i, N, speed, traj)
#
#         if i == 0:
#             lineSegMsg1 = generate_LineSegment(x_ref[0:N, :2], id=1, colors=[1.0, 1.0, 1.0])
#             visualizerPub.publish(lineSegMsg1)
#         else:
#             lineSegMsg2 = generate_LineSegment(x_ref[0:N, :2], id=2, colors=[1.0, 0.0, 1.0])
#             visualizerPub.publish(lineSegMsg2)
#         i += 1
#         rate.sleep()
#
#
# def visualize_parameter_path(center_line, start_pt):
#
#     if not np.array_equal(start_pt[:2], center_line[0, :2]):
#         center_line = reformulate_path(center_line, start_pt)
#
#     visPub = rospy.Publisher('/visualization_marker', Marker, queue_size=10)  # Publisher for Marker messages
#
#     param_center_line = parameterize_path(center_line)
#     arc_lengths = np.cumsum(np.sqrt(
#                                     np.diff(np.append(center_line[:, 0], center_line[0, 0])) ** 2 +
#                                     np.diff(np.append(center_line[:, 1], center_line[0, 1])) ** 2
#                                    )
#                             )
#     theta = 0
#     i = 0
#     t_samp = 0.02
#     rate = rospy.Rate(1/t_samp)
#
#     while not rospy.is_shutdown() and theta <= arc_lengths[-1]:
#         if i == 0:
#             ref = get_param_ref(t_samp=t_samp, N=20, initial=True, ref_path_parameterized=param_center_line)
#         # visPub.publish(refPtMsg)
#
#         v_theta = 5  # 5*abs(np.sin(i*np.pi/180))
#         theta = theta + v_theta*t_samp
#         i += 1
#         rate.sleep()
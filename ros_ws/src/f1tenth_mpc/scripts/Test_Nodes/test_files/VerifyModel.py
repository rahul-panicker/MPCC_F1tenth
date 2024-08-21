#!/usr/bin/env python3

import matplotlib.pyplot as plt
import math
import types
from scipy.integrate import solve_ivp
import multiprocessing
import ctypes

from ackermann_msgs.msg import AckermannDriveStamped
from scripts.Helper_scripts.RCDriver import generate_drive_msg
from scripts.Helper_scripts.GetStates import *


def st_model(t, states, inputs, p):

    g = 9.81  # m/s2

    x1 = states[0]  # X
    x2 = states[1]  # Y
    x3 = states[2]  # (steering angle) delta
    x4 = states[3]  # (velocity) v
    x5 = states[4]  # (orientation) psi
    x6 = states[5]  # psi_dot
    x7 = states[6]  # (slip angle)beta

    u1 = inputs[0]  # steering angle rate
    u2 = inputs[1]  # longitudinal acceleration

    # Differential states
    x1_dot = x4 * np.cos(x5 + x7)
    x2_dot = x4 * np.sin(x5 + x7)
    x3_dot = u1
    x4_dot = u2
    x5_dot = x6

    a1 = p.lf*p.Csf*(g*p.lr - u2*p.hcg)
    a2 = p.lr*p.Csr*(g*p.lf + u2*p.hcg) - p.lf*p.Csf*(g*p.lr - u2*p.hcg)
    a3 = (p.lf**2)*p.Csf*(g*p.lr - u2*p.hcg) + (p.lr**2)*p.Csr*(g*p.lf + u2*p.hcg)

    x6_dot = (p.U*p.m)/(p.Iz*(p.lr + p.lf)) * (a1*x3 + a2*x7 - a3*(x6/x4))

    b1 = p.Csf*(g*p.lr - u2*p.hcg)
    b2 = p.Csr*(g*p.lf + u2*p.hcg) + p.Csf*(g*p.lr - u2*p.hcg)
    b3 = p.Csr*(g*p.lf + u2*p.hcg)*p.lr - p.Csf*(g*p.lr - u2*p.hcg)

    x7_dot = p.U/(x4*(p.lr+p.lf)) * (b1*x3 - b2*x7 + b3*(x6/x4)) - x6

    x_dot = [x1_dot, x2_dot, x3_dot, x4_dot, x5_dot, x6_dot, x7_dot]

    return x_dot


def dyn_model(t, states, inputs, p):

    x1 = states[0]  # X
    x2 = states[1]  # Y
    x3 = states[2]  # (orientation) psi
    x4 = states[3]  # vx
    x5 = states[4]  # vy
    x6 = states[5]  # w

    u1 = inputs[0]  # longitudinal velocity
    u2 = inputs[1]  # steering angle

    xdot_1 = x4 * np.cos(x3) - x5 * np.sin(x3)
    xdot_2 = x4 * np.sin(x3) + x5 * np.cos(x3)
    xdot_3 = x6

    alphaf = -math.atan2((x6 * p.lf + x5), x4) + u2
    alphar = math.atan2((x6 * p.lr - x5), x4)

    Fry = p.D * np.sin(p.C * math.atan(p.B * alphar))
    Ffy = p.D * np.sin(p.C * math.atan(p.B * alphaf))
    xdot_4 = (u1 - x4) * p.Kv
    xdot_5 = (Fry + Ffy * np.cos(u2) - p.m * x4 * x6) / p.m
    xdot_6 = (Ffy * p.lf * np.cos(u2) - Fry * p.lr) / p.Iz

    xdot = [xdot_1, xdot_2, xdot_3, xdot_4, xdot_5, xdot_6]

    return xdot


def simulate_car(speed, start_event):
    rospy.init_node('Simulate_drive', anonymous=True)

    t_samp = 0.1
    t_span = (0.0, t_samp)

    p = types.SimpleNamespace()
    p.lr = 0.17145
    p.lf = 0.15875
    p.lwb = (p.lf + p.lr)
    p.Csf = 4.718
    p.Csr = 5.4562
    p.hcg = 0.074
    p.U = 0.523
    p.m = 3.47
    p.Iz = 0.04712
    p.Kv = 10
    p.B = 10
    p.C = 1.9
    p.D = 1
    car_odometry = odom()

    rate = rospy.Rate(1 / t_samp)
    while speed.value < 0.5:
        rate.sleep()

    start_event.set()
    print('Minimal speed threshold crossed. Starting simulation')

    # Single Track Model
    v0 = car_odometry.linear_vel[0]
    pose = car_odometry.pose
    w0 = car_odometry.angular_vel[2]

    x0_st = [pose[0], pose[1], 0.3, v0, pose[2], 0, w0]
    x0_dyn = [pose[0],pose[1], pose[2], v0, 0.0, w0]
    a_long = 0.0  # m/s2
    v_delta = 0.0  # rad/s

    v_dyn = 3.0
    steer_ang = 0.3

    u_st = [v_delta, a_long]
    u_dyn = [v_dyn, steer_ang]
    current_state = x0_st
    # x_traj = np.array([current_state])

    while not rospy.is_shutdown():
        # Next state of car using solve_ivp
        st_sol = solve_ivp(st_model, t_span, current_state, args=(u_st, p))

        # Update state
        current_state = st_sol.y[:, -1]

        # Update queue
        sim_state_queue.put(current_state)

        # x_traj = np.vstack([x_traj, np.array([st_sol.y[:, -1]])])

        rate.sleep()


def drive_car(speed):
    rospy.init_node('Drive_car', anonymous=True)
    drivePub = rospy.Publisher('/drive', AckermannDriveStamped, queue_size=10000)  # Publisher for drive messages

    t_samp = 0.05
    vel = 1.5
    steer_ang = 0.3
    car_odometry = odom()

    rate = rospy.Rate(int(1/t_samp))
    i = 0

    while not rospy.is_shutdown():
        # Move car
        driveMsg = generate_drive_msg(speed=vel, heading=steer_ang)
        drivePub.publish(driveMsg)

        # Get and add pose to queue
        position = car_odometry.pose[:2]
        drive_pos_queue.put(position)

        speed.value = car_odometry.linear_vel[0]

        i += 1
        rate.sleep()


def plot_positions(q1, q2, speed, start_event):

    rospy.init_node('plot_states', anonymous=True)
    t_samp = 0.05
    rate = rospy.Rate(1 / t_samp)
    while speed.value < 0.5:
        rate.sleep()

    start_event.set()
    print('Event triggered. Starting plots')

    plt.figure()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()

    while True:
        # Get the state from the queue
        sim_state = q1.get()
        drive_state = q2.get()

        plt.plot(sim_state[0], sim_state[1], 'ro', label='Calculated states')
        plt.plot(drive_state[0], drive_state[1], 'bo', label='Simulated states')

        plt.pause(t_samp)  # Allow the plot to update


if __name__ == "__main__":

    # Create a multiprocessing Queues for communication between processes
    sim_state_queue = multiprocessing.Queue()
    drive_pos_queue = multiprocessing.Queue()

    # Shared variable for car states
    speed = multiprocessing.Value(ctypes.c_float, 0)

    # Event to start simulation
    start_event = multiprocessing.Event()

    drive_process = multiprocessing.Process(target=drive_car, args=(speed,))
    sim_process = multiprocessing.Process(target=simulate_car, args=(speed, start_event))
    plot_process = multiprocessing.Process(target=plot_positions, args=(sim_state_queue, drive_pos_queue, speed, start_event))

    drive_process.start()
    sim_process.start()
    plot_process.start()









# def kst_model(t, states, vel, steer):
#
#     p = types.SimpleNamespace()
#     p.lr = 0.17145
#     p.lf = 0.15875
#
#     x1 = states[0]  # X
#     x2 = states[1]  # Y
#     x3 = states[2]  # (orientation) psi
#
#     # Differential states
#     x1_dot = vel(t) * np.cos(x3)
#     x2_dot = vel(t) * np.sin(x3)
#     x3_dot = vel(t) * (np.tan(steer(t)) / (p.lr + p.lf))
#
#     x_dot = [x1_dot, x2_dot, x3_dot]
#
#     return x_dot
#
#
# def dyn_model(t, states, velocity, steering):
#
#     lr = 0.17145
#     lf = 0.15875
#     Kv = 10
#     D = 1
#     C = 1.9
#     B = 10
#     m = 2
#     Iz = 0.04712
#     x = states[0]
#     y = states[1]
#     theta = states[2]
#     vx = states[3]
#     vy = states[4]
#     omega = states[5]
#
#     vxref = velocity(t)
#     delta = steering(t)
#     xdot_1 = vx * np.cos(theta) - vy * np.sin(theta)
#     xdot_2 = vx * np.sin(theta) + vy * np.cos(theta)
#     xdot_3 = omega
#
#     alphaf = -math.atan((omega * lf + vy) / vx) + delta
#     alphar = math.atan((omega * lr - vy) / vx)
#
#     Fry = D * np.sin(C * math.atan(B * alphar))
#     Ffy = D * np.sin(C * math.atan(B * alphaf))
#     xdot_4 = (vxref - vx) * Kv
#     xdot_5 = (Fry + Ffy * np.cos(delta) - m * vx * omega) / m
#     xdot_6 = (Ffy * lf * np.cos(delta) - Fry * lr) / Iz
#
#     xdot = [xdot_1, xdot_2, xdot_3, xdot_4, xdot_5, xdot_6]
#     # x1 = states[0]  # X
#     # x2 = states[1]  # Y
#     # x3 = states[2]  # psi (orientation)
#     # x4 = states[3]  # vx
#     # x5 = states[4]  # vy
#     # x6 = states[5]  # w (Omega/ Angular velocity)
#     #
#     # u1 = steering(t)
#     # u2 = velocity(t)
#     #
#     # # Differential states
#     # x1_dot = x4 * np.cos(x3) - x5 * np.sin(x3)
#     # x2_dot = x4 * np.sin(x3) + x5 * np.cos(x3)
#     # x3_dot = x6
#     # x4_dot = (u2-x4)*p.Kv
#     #
#     # alpha_f = -math.atan2(x6*p.lf + x5, x4) + u1
#     # alpha_r = math.atan2(x6*p.lr - x5, x4)
#     #
#     # k1 = math.atan2(p.B*alpha_r, 1)
#     # k2 = math.atan2(p.B*alpha_f, 1)
#     # Fry = p.D*np.sin(p.C*k1)
#     # Ffy = p.D*np.sin(p.C*k2)
#     #
#     # x5_dot = (Fry + Ffy*np.cos(u1) - p.m*x4*x6)*(1/p.m)
#     # x6_dot = (Ffy*p.lf*np.cos(u1) - Fry*p.lr)*(1/p.Iz)
#     #
#     # x_dot = [x1_dot, x2_dot, x3_dot, x4_dot, x5_dot, x6_dot]
#
#     return xdot
#
#
# def st_model_mod(t, states, inputs, p):
#
#     x1 = states[0]  # X (X CoG)
#     x2 = states[1]  # Y (Y CoG)
#     x3 = states[2]  # delta (steering angle)
#     x4 = states[3]  # v (Velocity)
#     x5 = states[4]  # psi (Orientation)
#     x6 = states[5]  # w (Angular velocity)
#     x7 = states[6]  # beta (slip angle)
#
#     u1 = inputs[0]  # steering angle rate
#     u2 = inputs[1]  # longitudinal acceleration
#
#     # Differential states
#     x1_dot = x4*np.cos(x5+x7)
#     x2_dot = x4*np.sin(x5+x7)
#     x3_dot = u1
#     x4_dot = u2
#     x5_dot = x6
#     x6_dot = (u2*np.tan(x3) + (x4*u1/np.cos(x3)**2))/p.lwb
#
#     g = p.lr * np.tan(x3)/p.lwb
#     x7_dot = u1/(1+g**2) * p.lr/(p.lwb * (np.cos(x3)**2))
#
#     x_dot = [x1_dot, x2_dot, x3_dot, x4_dot, x5_dot, x6_dot, x7_dot]
#     return x_dot
#
#
# def vel(t):
#     return 0.5 + 0.5*t
#
#
# def steer(t):
#     return 0.3

# class odom:
#     def __init__(self):
#         self.pose = np.zeros([3, 1])
#         self.Odomsub = rospy.Subscriber('/odom', Odometry, self.odomCallback)
#         self.first_msg_flag = True
#         self.init_pose = None
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
#         if self.first_msg_flag:
#             self.init_pose = self.pose
#             self.first_msg_flag = False


# def main():
#
#     rospy.init_node('Model_verification', anonymous=True)
#     drivePub = rospy.Publisher('/drive', acker, queue_size=10)
#
#     pointMarkerPub = rospy.Publisher('/visualization_marker', Marker, queue_size=10)
#     N_sim = 300
#
#     car_model = CarModel()
#
#     car_odometry = odom()
#     rospy.sleep(1)
#     x0 = car_odometry.init_pose.reshape(car_model.nx, 1)
#
#     x = x0
#     x_sim_traj_model = []
#
#     i = 0
#     rate = rospy.Rate(50)
#     while not rospy.is_shutdown():
#         vel = 2.0
#         delta = np.sin(i*np.pi/180)
#         u = np.array([delta, vel])
#         driveMsg = generate_drive_msg(vel, delta)
#
#         x = car_model.simulate(x,u).reshape(car_model.nx, 1)
#         drivePub.publish(driveMsg)
#
#         x_sim_traj_model = np.append(x_sim_traj_model, x)
#         pt_x = x[0]
#         pt_y = x[1]
#         lineMsg = generate_PointMarkerMsg(pt_x ,pt_y, id=i, colors=[1.0, 0.0, 0.0])
#
#         pointMarkerPub.publish(lineMsg)
#         i += 1
#         rate.sleep()
#



# def test_models():
#     T = 1000.0
#     dt = 0.05
#     t_span = (0.0, T)
#
#     # Model Parameters
#     p = types.SimpleNamespace()
#     p.lr = 0.17145
#     p.lf = 0.15875
#     p.lwb = (p.lf + p.lr)
#     p.Csf = 4.718
#     p.Csr = 5.4562
#     p.hcg = 0.074
#     p.U = 0.523
#     p.m = 3.47
#     p.Iz = 0.04712
#     p.Kv = 10
#     p.B = 10
#     p.C = 1.9
#     p.D = 1
#
#     # Single Track Model
#     v0 = 0.5
#     x0_st = [0, 0, 0.3, v0, 0, 0, 0]
#     a_long = 0.5  # m/s2
#     v_delta = 0.0  # rad/s
#     u_st = [v_delta, a_long]
#     t_eval = np.arange(0.0, T+dt, dt)
#     st_sol = solve_ivp(st_model, t_span, x0_st, t_eval=t_eval, args=(u_st, p))
#
#     # Dynamic Model
#     dyn_x0 = [0, 0, 0, v0, 0, 0]
#     dyn_sol = solve_ivp(dyn_model, t_span, dyn_x0, t_eval=t_eval, args=(vel, steer))
#     labels_dyn = ["X", "Y", "psi", "vx", "vy", "w"]
#
#     t_ode = dyn_sol.t
#
#     x_dyn = dyn_sol.y[0, :]
#     x_st = st_sol.y[0, :]
#
#     y_dyn = dyn_sol.y[1, :]
#     y_st = st_sol.y[1, :]
#
#     psi_dyn = dyn_sol.y[2, :]
#     psi_st = st_sol.y[4, :]
#
#     vx = dyn_sol.y[3, :]
#     vy = dyn_sol.y[4, :]
#
#     v_dyn = np.sqrt(vx**2 + vy**2)
#     v_st = st_sol.y[3, :]
#
#     plt.plot(x_dyn, y_dyn, label='Dynamic model')
#     plt.plot(x_st, y_st, label='Simulator model')
#     plt.xlabel("X")
#     plt.ylabel("Y")
#     plt.title("Car position")
#     plt.legend()
#
#     plt.figure()
#     plt.plot(t_ode, v_st, label='Simulator model')
#     plt.plot(t_ode, v_dyn, label='Dynamic model')
#     plt.legend()
#     plt.title('Velocity')
#
#     plt.figure()
#     plt.plot(t_ode, psi_st, label='Simulator model')
#     plt.plot(t_ode, psi_dyn, label='Dynamic model')
#     plt.legend()
#     plt.show()
    # plot_ode_states(sol=dyn_sol, nx=len(dyn_x0), labels=labels_dyn)

    # labels_dyn = ["X", "Y", "psi", "vx", "vy", "w"]
    # labels_st = ['X', 'Y', 'delta', 'v', 'psi', 'w', 'beta']
    #
    # plot_ode_states(sol=st_sol, nx=len(x0_st), labels=labels_st)
    # plot_ode_states(sol=dyn_sol, nx=len(dyn_x0), labels=labels_dyn)

    # Single Track Model low velocity
    # st_mod_x0 = [0, 0, 0.3, v0, 0, 0, 0]
    # st_mod_sol = solve_ivp(st_model_mod, t_span, st_mod_x0, t_eval=t_eval, args=(u_st, p))

    # Kinematic Single Track Model
    # x0_ks = [0, 0, 0.3]
    # ks_sol = solve_ivp(kst_model, t_span, x0_ks, t_eval=t_eval, args=(vel, steer))

    # compare_sols(sol1=st_sol, sol2=st_mod_sol)

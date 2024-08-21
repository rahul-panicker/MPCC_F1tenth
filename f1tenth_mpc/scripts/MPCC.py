#!/usr/bin/env python3
import time
import matplotlib.pyplot as plt

# Helper Scripts
from Controller import *
from Helper_scripts.referenceParser import *
from Helper_scripts.GetStates import *
from Helper_scripts.generate_vizMarkerMsg import generate_line_msg
from Helper_scripts.RCDriver import generate_drive_msg

# ROS Imports
import rospy
from ackermann_msgs.msg import AckermannDriveStamped
from visualization_msgs.msg import Marker


def main(simulation=True, odometry=False, controller_type='MPCC', model_type='Kinematic Bicycle', amcl=True):

    # Create Publishers
    if simulation:
        drivePub = rospy.Publisher('/drive', AckermannDriveStamped, queue_size=10)  # Publisher for drive messages
    else:
        drivePub = rospy.Publisher('/vesc/low_level/ackermann_cmd_mux/input/navigation', AckermannDriveStamped,
                                   queue_size=10)  # Publisher for drive messages

    visPub = rospy.Publisher('/visualization_marker', Marker, queue_size=10)  # Publisher for Marker messages

    """ EXTRACT REFERENCE """
    ref_line = get_reference_path_from_file(track="Monza")  # track="Monza","Silverstone", "RST_Hallway", "rst_track"

    # get car starting point
    if odometry:
        carState = odom()
    elif amcl:
        carState = AMCL()
    else:
        carState = ParticleFilter()
    start_point = carState.init_pos

    # Perform arc length parameterization
    param_ref_line = parameterize_ref_path(ref_line)

    # project starting point on reference path
    s_opt = project_point(param_ref_line, start_point)
    closest_pt = np.array([param_ref_line.x(s_opt), param_ref_line.y(s_opt)])

    # Reformulate the reference path so that it starts from the closest point to the car
    param_ref_line = reform_param_path(ref_line, closest_pt)

    """INITIALIZE VARIABLES AND TRACK PARAMETERS"""

    # Track parameters
    track = types.SimpleNamespace()
    track.length = param_ref_line.s[-1]
    track.width = 1.75

    # Initialize trajectory storage variables
    ec_ar = np.array([])
    el_ar = np.array([])

    calc_vel_traj = np.array([0])
    meas_vel_traj = np.array([0])

    ref_array = np.array([])
    closed_loop_state_traj = np.array([])

    opt_xi = []
    opt_ui = []
    s = 0.0

    lap = 0
    t_lap = 0.0
    t_lap_array = np.array([])
    t_opt_array = np.array([])

    """ MPC """

    # Assign initial state
    if controller_type.casefold() == 'MPCC'.casefold():
        if model_type.casefold() != 'Kinematic Bicycle'.casefold():
            x0 = np.append(start_point, np.zeros(7))
        else:
            x0 = np.append(start_point, np.zeros(4))
    else:
        x0 = None

    # Create MPC object and build solver(s)
    Tf = 3  # Prediction horizon in seconds
    N = 45  # Number of shooting nodes in Tf seconds
    stop_time = 60  # Maximum simulation time in seconds
    t_samp = round(Tf / N, 3)  # Sampling time in seconds
    N_stop = int(stop_time / t_samp)  # Number of iterations to run the simulation

    if controller_type.casefold() != 'MPCC'.casefold():
        mpc = None
    else:
        weights = [1e-2, 1, 1]  # [1e-6, 5, 5] - Time Optimal weights, [5.0, 1.0, 0.0] - Reference Tracking weights
        R = np.diag([1e-1, 1e-2, 1e-2])  # weights on del_u
        n_params = 3  # Reference pose at every instance as parameter
        mpc = MPCC(model_type=model_type, horizon=Tf, shooting_nodes=N, track=track, x0=x0,
                   weights=weights, n_params=n_params, constrained=True, R=R)

    i = 0
    current_state = x0
    rate = rospy.Rate(1/t_samp)

    """MPC Main Loop"""
    while not rospy.is_shutdown():

        if controller_type.casefold() == 'MPCC'.casefold():
            """ GET REFERENCES AND INITIAL GUESSES"""
            if i == 0:
                ref = get_reference(current_pos=current_state[:3], param_path=param_ref_line, N=N, t_samp=t_samp,
                                    iteration=i, v_s_max=mpc.ocp.system.model.v_s_max)
                init_x, init_u = mpc.generate_init_guess(x0=x0)
                ref_array = ref[0, :3]
            else:
                ref = get_reference(current_pos=current_state[:3], param_path=param_ref_line, N=N, t_samp=t_samp,
                                    iteration=i, opt_x=opt_xi)
                init_x, init_u = mpc.warm_start()
                ref_array = np.vstack((ref_array, ref[0, :3]))

            """ SOLVE OCP """
            t1 = time.perf_counter()
            opt_xi, opt_ui, status = mpc.solve_ocp(solver=mpc.ocp_solver, x=current_state, x_init=init_x,
                                                   u_init=init_u, iteration=i, param=ref)
            t2 = time.perf_counter()
            t_opt_array = np.append(t_opt_array, t2 - t1)

            # else:
            #     t1 = time.perf_counter()
            #     opt_xi, opt_ui, status = mpc.solve_ocp(solver=mpc.ocp_solver, x=current_state, x_init=init_x,
            #                                            u_init=init_u, iteration=i, param=ref)
            #     t2 = time.perf_counter()
            #     t_opt_array = np.append(t_opt_array, t2-t1)

            """ EXTRACT INPUTS"""
            if model_type.casefold() != 'Kinematic bicycle'.casefold():
                delta = opt_xi[1, 7]
                v = opt_xi[1, 8]
                v_s = opt_xi[1, 9]
                calc_vel_traj = np.append(calc_vel_traj, v)

            else:
                delta = opt_xi[1, 4]  # Steering angle
                v = opt_xi[1, 5]  # Velocity
                v_s = opt_xi[1, 6]
                calc_vel_traj = np.append(calc_vel_traj, v)

            # Calculate the error terms
            curr_pose = current_state[:3]
            ref_pose = ref[0, :3]
            e_c = ec(pose=curr_pose, ref_pose=ref_pose)
            e_l = el(pose=curr_pose, ref_pose=ref_pose)
            ec_ar = np.append(ec_ar, e_c)
            el_ar = np.append(el_ar, e_l)

            # Visualize predicted states and reference path
            predMsg = generate_line_msg(opt_xi[1:, :2], id=1, colors=[0.0, 0.0, 1.0])
            refMsg = generate_line_msg(ref[1:, :2], id=2, colors=[1.0, 0.0, 0.0])
            visPub.publish(refMsg)
            visPub.publish(predMsg)

            """ APPLY INPUTS AND MEASURE NEXT STATE """
            # if simulation:
            # Run car
            driveMsg = generate_drive_msg(heading=delta, speed=v)
            drivePub.publish(driveMsg)

            # Start Lap Timing
            if i == 0:
                t_start = time.perf_counter()
            if s > track.length:
                lap += 1
                t_lap = time.perf_counter() - t_start
                t_lap_array = np.append(t_lap_array, t_lap)
                s = s - track.length

            # Measure next state(s)
            if model_type.casefold() != 'Kinematic bicycle'.casefold():
                next_pose = carState.pose
                s += v_s * t_samp
                vx = carState.linear_vel[0]
                vy = carState.linear_vel[1]
                w = carState.angular_vel[2]
                next_state = np.append(next_pose, [vx, vy, w, s, delta, v, v_s])

            else:
                next_pose = carState.pose
                s += v_s * t_samp
                next_state = np.append(next_pose, [s, delta, v, v_s])

            # else:
            #     # Create and send drive message to car
            #     driveMsg = generate_drive_msg(speed=v, heading=delta)
            #     drivePub.publish(driveMsg)
            #
            #     # Start timing lap
            #
            #     # Measure next state
            #     if model_type.casefold() != 'Kinematic bicycle'.casefold():
            #         next_pose = carState.pose
            #         s += v_s * t_samp
            #         vx = carState.linear_vel[0]
            #         vy = carState.linear_vel[1]
            #         w = carState.angular_vel[2]
            #         next_state = np.append(next_pose, [vx, vy, w, s, delta, v, v_s])
            #
            #     else:
            #         next_pose = carState.pose
            #         s += v_s * t_samp
            #         next_state = np.append(next_pose, [s, delta, v, v_s])

            if i == 0:
                closed_loop_state_traj = np.array(current_state)
            else:
                closed_loop_state_traj = np.vstack((closed_loop_state_traj, next_state))

            # Stop car and exit loop if sim time is completed or error occurs
            if i > N_stop or status != 0:
                N_stop = i
                break

            if odometry:
                vel = carState.linear_vel[0]
                meas_vel_traj = np.append(meas_vel_traj, vel)

            current_state = next_state
            i += 1
            rate.sleep()

        else:
            pass
            # Implement other MPC schemes

    # Stop the car
    brakeMsg = generate_drive_msg(speed=0, heading=0.0)
    drivePub.publish(brakeMsg)

    # Plot results
    t_sim = np.arange(0.0, round((N_stop+1)*t_samp, 2), t_samp)

    plt.figure()
    plt.title("Closed Loop position")
    plt.plot(closed_loop_state_traj[:, 0], closed_loop_state_traj[:, 1], "b.", label="Car Position")
    plt.plot(closed_loop_state_traj[0, 0], closed_loop_state_traj[0, 1], "go", label="Start Point")
    plt.plot(ref_array[:, 0], ref_array[:, 1], "r.", label="Reference")
    # plt.plot(left_boundary[:, 0], left_boundary[:, 1])
    # plt.plot(right_boundary[:, 0], right_boundary[:, 1])
    plt.xlabel("X axis")
    plt.ylabel("Y axis")
    plt.legend()

    # fig = plt.figure()
    # fig.suptitle('Optimization time vs Iteration')
    # plt.plot(t_opt_array, 'b', label='Optimization time')
    #
    # fig1, axs = plt.subplots(3, 1)
    # fig1.suptitle('State vs Time plot')
    # axs[0].plot(t_sim, closed_loop_state_traj[:, 0], '.', label='X coordinates')
    # axs[0].plot(t_sim, ref_array[:, 0], '.', label='Reference X coordinates')
    # axs[0].legend()
    # axs[1].plot(t_sim, closed_loop_state_traj[:, 1], '.', label='Y coordinates')
    # axs[1].plot(t_sim, ref_array[:, 1], '.', label='Reference Y coordinates')
    # axs[1].legend()
    # axs[2].plot(t_sim, closed_loop_state_traj[:, 2], '.', label='Orientation')
    # axs[2].plot(t_sim, ref_array[:, 2], '.', label='Reference Orientation')
    # axs[2].legend()
    #
    # fig2, axs = plt.subplots(2,1)
    # fig2.suptitle("Contouring and Lag Errors")
    # axs[0].plot(t_sim, ec_ar, 'b.', label="Contouring error")
    # axs[0].legend()
    # axs[1].plot(t_sim, el_ar, 'r.', label="Lag error")
    # axs[1].legend()
    #
    # if odometry:
    #     plt.figure()
    #     plt.title("Measured and Calculated velocities")
    #     plt.plot(t_sim, calc_vel_traj[:-1], "r.", label="Calculated velocity")
    #     plt.plot(t_sim, meas_vel_traj, 'g.', label="Measured velocity")
    #     plt.xlabel("Time")
    #     plt.ylabel("Velocity m/s")
    #     plt.legend()
    # else:
    #     plt.figure()
    #     plt.title("Velocity Measurement")
    #     plt.plot(t_sim, calc_vel_traj[:-1], "r.", label="Calculated velocity")
    #     plt.xlabel("Time")
    #     plt.ylabel("Velocity m/s")
    #     plt.legend()
    #
    if t_lap != 0.0:
        print(f'Number of laps completed: {lap}')
        print(f'Minimum lap time: {t_lap_array.min()}')
    else:
        print('Lap not completed')
    plt.show()


if __name__ == '__main__':
    rospy.init_node('MPC_Node', anonymous=True)
    main(model_type='Kinematic Bicycle', controller_type='MPCC', odometry=False, simulation=True, amcl=True)
    # model_type = 'Dynamic' or 'Kinematic Bicycle', controller_type = 'MPCC' '


# def test():
#
#     # Create Publishers
#     drivePub = rospy.Publisher('/drive', AckermannDriveStamped, queue_size=10)  # Publisher for drive messages
#     visPub = rospy.Publisher('/visualization_marker', Marker, queue_size=10)  # Publisher for Marker messages
#
#     """ EXTRACT REFERENCE"""
#     center_line = get_reference_path_from_file(track="RST_Hallway")  # track="Monza","Silverstone", "RST_Hallway"
#
#     # Get the car's starting point
#     car_odometry = odom()
#     amcl_pose = AMCL()
#     start_pose = amcl_pose.init_pos
#
#     # track_start_pos, _ = get_closest_pose(center_line, start_pose[:2])
#     #
#     # # reformulate the center line coordinates so that the track starts from the closest point to the car
#     # center_line = reformulate_path(center_line, track_start_pos)
#     #
#     # # Close loop
#     # center_line = np.vstack((center_line, center_line[0, :]))
#     #
#     # # Parameterize and interpolate reference path
#     # param_center_line = parameterize_ref_path(center_line)
#     # center_line = interpolate_param_path(param_center_line, n=5)
#
#     # Parameterize center line
#
#     # Close loop
#     # center_line = np.vstack((center_line, center_line[0, :]))
#     param_center_line = parameterize_ref_path(center_line)
#     # center_line = interpolate_param_path(param_center_line, n=3)
#
#     # Project starting point on center line
#     s_opt = project_point(param_center_line, start_pose)
#     xs0 = np.array([param_center_line.x(s_opt), param_center_line.y(s_opt), param_center_line.psi(s_opt)])
#     start_pt = generate_PointMarkerMsg(xs0[0], xs0[1], colors=[0.0, 0.0, 0.0])
#     visPub.publish(start_pt)
#
#     """ MPC """
#     model_type = 'Kinematic Bicycle'  # 'Bicycle', 'Dynamic' or 'Kinematic Bicycle'
#     controller = 'MPCC'  # 'TrackingMPC', 'MPCC'
#
#     # Define Initial state
#     init_pose = amcl_pose.init_pos
#     init_vel = car_odometry.init_linear_vel
#     init_delta = 0.0
#     init_v_s = 0.0
#     s = s_opt
#
#     if model_type.casefold() == 'Kinematic Bicycle'.casefold():
#         if controller.casefold() == 'MPCC'.casefold():
#             x0 = np.append(init_pose, [s, init_delta, init_vel[0], init_v_s])
#         else:
#             x0 = np.append(init_pose, [init_delta, init_vel[0]])
#     else:
#         x0 = None
#
#     # Track parameters
#     track = types.SimpleNamespace()
#     track.length = param_center_line.s[-1]
#     track.width = 2.5
#
#     # Initialize trajectory storage variables
#     ec_ar = np.array([])
#     el_ar = np.array([])
#     ref_array = np.array([])
#
#     calc_vel_traj = np.array(init_vel[0])
#     meas_vel_traj = np.array(init_vel[0])
#
#     current_state = x0
#     closed_loop_state_traj = np.array(current_state)
#
#     opt_xi = []
#     opt_ui = []
#     lap = 0
#
#     # Create MPC object and build solver(s)
#     Tf = 3  # Prediction horizon in seconds
#     N = 150  # Number of shooting nodes in Tf seconds
#     stop_time = 60  # Maximum simulation time in seconds
#     t_samp = round(Tf/N, 2)  # Sampling time in seconds
#     N_stop = int(stop_time / t_samp)  # Number of iterations to run the simulation
#
#     if controller.casefold() == 'MPCC'.casefold():
#         weights = [1, 1000, 1]  # [1e-6, 7.5, 5]  # [1e-4, 0.50, 0.5]  # [1e-3, 10, 2]  # [1.0, 10.0, 800.0]
#         R = np.diag([1e-2, 1e-3, 1e-3])  # np.diag([0.1, 1e-1, 0.1])
#         n_params = 3  # Reference pose at every instance as parameter
#         mpc = MPCC(model_type=model_type, horizon=Tf, shooting_nodes=N, track=track, x0=x0,
#                    weights=weights, n_params=n_params, constrained=True, R=R)
#     elif controller.casefold() == 'TrackingMPC'.casefold():
#         Q = np.diag([1.0, 1.0, 1.0, 1.0, 1.0])
#         R = np.diag([1.0, 1.0])
#         mpc = TrackingMPC(model_type=model_type, horizon=Tf, shooting_nodes=N, track=track, x0=x0, Q=Q, R=R)
#     else:
#         mpc = None
#
#     i = 0
#     rate = rospy.Rate(1/t_samp)
#
#     # generate velocity profile
#     vel_ref = velocity_profile(param_path=param_center_line)
#     while not rospy.is_shutdown():
#         if controller.casefold() == 'MPCC'.casefold():
#             if s > track.length:
#                 s = s - track.length
#                 lap += 1
#
#             # Calculate reference and initial guesses for solver
#             if i == 0:
#                 ref = get_param_ref(ref_path_parameterized=param_center_line, s=current_state[3], N=N, t_samp=t_samp,
#                                     initial=True)
#                 # ref = get_reference(current_pos=current_state[:3], param_path=param_center_line, N=N, t_samp=t_samp,
#                 #                     iteration=i)
#                 init_x, init_u = mpc.generate_init_guess(x0=x0)
#                 ref_array = ref[0, :3]
#             else:
#                 ref = get_param_ref(ref_path_parameterized=param_center_line, s=current_state[3], N=N, t_samp=t_samp,
#                                     opt_x=opt_xi)
#                 # ref = get_reference(current_pos=current_state[:3], param_path=param_center_line, N=N, t_samp=t_samp,
#                 #                     iteration=i, opt_x=opt_xi)
#                 init_x, init_u = mpc.warm_start()
#                 ref_array = np.vstack((ref_array, ref[0, :3]))
#
#             # Solve OCP
#             if model_type.casefold() == 'Kinematic bicycle'.casefold():
#                 opt_xi, opt_ui, status = mpc.solve_ocp(solver=mpc.ocp_solver, x=current_state, x_init=init_x, u_init=init_u,
#                                                        iteration=i, param=ref)
#             else:
#                 opt_xi = None
#                 opt_ui = None
#                 status = 5
#
#             # Extract inputs
#             if model_type.casefold() == 'Kinematic bicycle'.casefold():
#                 delta = opt_xi[1, 4]  # Steering angle
#                 v = opt_xi[1, 5]  # Velocity
#                 v_s = opt_xi[1, 6]
#                 calc_vel_traj = np.append(calc_vel_traj, v)
#             else:
#                 delta = None
#                 v = None
#                 v_s = None
#
#             # Calculate the error terms
#             curr_pose = current_state[:3]
#             ref_pose = ref[0, :3]
#             e_c = ec(pose=curr_pose, ref_pose=ref_pose)
#             e_l = el(pose=curr_pose, ref_pose=ref_pose)
#             ec_ar = np.append(ec_ar, e_c)
#             el_ar = np.append(el_ar, e_l)
#
#             # Run car on f1tenth simulator
#             driveMsg = generate_drive_msg(heading=delta, speed=v)
#             drivePub.publish(driveMsg)
#
#             predMsg = generate_line_msg(opt_xi[1:, :2], id=1, colors=[0.0, 0.0, 1.0])
#             refMsg = generate_LineSegment(ref[1:, :2], id=2, colors=[1.0, 0.0, 0.0])
#             visPub.publish(refMsg)
#             visPub.publish(predMsg)
#
#             # Start lap timing
#             if i == 0:
#                 t_start = time.perf_counter()
#             next_pose = amcl_pose.pose
#             meas_vel_traj = np.append(meas_vel_traj, car_odometry.linear_vel[0])
#
#             s += v_s * t_samp
#             # if s > tr_len:
#             #     t_end = time.perf_counter()
#             #     t_lap = t_end - t_start
#             #     print(f'Lap time: {t_lap}')
#             #     N_stop = i
#             #     break
#
#             if i == N_stop or status != 0:
#                 N_stop = i
#                 break
#
#             next_state = np.append(next_pose, [s, delta, v, v_s])
#             closed_loop_state_traj = np.vstack((closed_loop_state_traj, next_state))
#             current_state = next_state
#
#             i += 1
#             rate.sleep()
#         elif controller.casefold() == 'TrackingMPC'.casefold():
#             if i == 0:
#                 # Get reference states at first iteration
#                 pass
#             else:
#                 pass
#
#     brakeMsg = generate_drive_msg(speed=0, heading=0.0)
#     drivePub.publish(brakeMsg)
#
#     # Lap analysis
#     # x_lap = closed_loop_state_traj[:, 0]
#     # y_lap = closed_loop_state_traj[:, 1]
#     # dist = np.cumsum(np.sqrt(np.diff(x_lap) ** 2 + np.diff(y_lap) ** 2))[-1]
#     # print(f'Lap distance: {dist}')
#     # print(f'Reference track length: {tr_len}')
#     # avg_vel = dist/t_lap
#     # print(f'Average speed measured:{avg_vel}')
#
#     # Curve Analysis
#
#     t_sim = np.arange(0.0, round((N_stop+1)*t_samp, 2), t_samp)
#
#     plt.figure()
#     plt.title("Closed Loop position")
#     plt.plot(closed_loop_state_traj[:, 0], closed_loop_state_traj[:, 1], "b.", label="Car Position")
#     plt.plot(ref_array[:, 0], ref_array[:, 1], "r.", label="Reference")
#     # plt.plot(left_boundary[:, 0], left_boundary[:, 1])
#     # plt.plot(right_boundary[:, 0], right_boundary[:, 1])
#     plt.xlabel("X axis")
#     plt.ylabel("Y axis")
#     plt.legend()
#
#     # fig1, axs = plt.subplots(3, 1)
#     # fig1.suptitle('State vs Time plot')
#     # axs[0].plot(t_sim, closed_loop_state_traj[:, 0], '.', label='X coordinates')
#     # axs[0].plot(t_sim, ref_array[:, 0], '.', label='Reference X coordinates')
#     # axs[0].legend()
#     # axs[1].plot(t_sim, closed_loop_state_traj[:, 1], '.', label='Y coordinates')
#     # axs[1].plot(t_sim, ref_array[:, 1], '.', label='Reference Y coordinates')
#     # axs[1].legend()
#     # axs[2].plot(t_sim, closed_loop_state_traj[:, 2], '.', label='Orientation')
#     # axs[2].plot(t_sim, ref_array[:, 2], '.', label='Reference Orientation')
#     # axs[2].legend()
#     #
#     fig2, axs = plt.subplots(2,1)
#     fig2.suptitle("Contouring and Lag Errors")
#     axs[0].plot(t_sim, ec_ar, 'b.', label="Contouring error")
#     axs[0].legend()
#     axs[1].plot(t_sim, el_ar, 'r.', label="Lag error")
#     axs[1].legend()
#
#     # plt.figure()
#     # plt.title("Measured and Calculated velocities")
#     # plt.plot(t_sim, calc_vel_traj[:-1], "r.", label="Calculated velocity")
#     # plt.plot(t_sim, meas_vel_traj[:-1], 'g.', label="Measured velocity")
#     # plt.xlabel("Time")
#     # plt.ylabel("Velocity m/s")
#     # plt.legend()
#     #
#     # print(f'Completed {lap} laps')
#     plt.show()
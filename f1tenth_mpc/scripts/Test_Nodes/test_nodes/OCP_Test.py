#!/usr/bin/env python3
import time

import matplotlib.pyplot as plt
import numpy as np
import rospy

from Helper_scripts.referenceParser import *
from Helper_scripts.GetStates import odom
from MPC import *
from Helper_scripts.generate_vizMarkerMsg import *
from Helper_scripts.interpolate_path import *
from ackermann_msgs.msg import AckermannDriveStamped
from Helper_scripts.RCDriver import generate_drive_msg
from Helper_scripts.get_boundary import get_bounds
from Helper_scripts.trackParser import *


def MPCC_OCP_test(simulator=True):

    # """ REFERENCE PATH """
    # # Get center line as reference path from file
    # center_line = get_reference_path_from_file()
    #
    # # Parameterize and interpolate reference path
    # param_center_line = parameterize_path(center_line)
    # center_line = interpolate_param_path(param_center_line, n=5)
    #
    # # Get start pose of car and set the closest point to it on the track as the starting point of the track
    # car_odometry = odom()
    # start_pose = car_odometry.init_pos
    # track_start_pos, close_pts = get_closest_pose(center_line, start_pose[:2])
    #
    # # reformulate the center line coordinates so that the track starts from the closest point to the car
    # center_line = reformulate_path(center_line, track_start_pos)
    #
    # # Re-parameterize center line so that arc length at starting position is = 0
    # param_center_line = parameterize_path(center_line)

    drivePub = rospy.Publisher('/drive', AckermannDriveStamped, queue_size=10)  # Publisher for drive messages
    visPub = rospy.Publisher('/visualization_marker', Marker, queue_size=10)  # Publisher for Marker messages

    """ EXTRACT REFERENCE"""
    center_line = get_reference_path_from_file(track="RST_Hallway")  # track= "RST_Hallway", "Monza","Silverstone" or "Spa"

    # center_line = extract_centerLine('Helper_scripts/RST_Hallway.png')

    # Get the car's starting point
    car_odometry = odom()
    start_pose = car_odometry.init_pos
    track_start_pos, _ = get_closest_pose(center_line, start_pose[:2])

    # reformulate the center line coordinates so that the track starts from the closest point to the car
    center_line = reformulate_path(center_line, track_start_pos)

    # Close loop
    center_line = np.vstack((center_line, center_line[0, :]))

    # Parameterize and interpolate reference path
    param_center_line = parameterize_ref_path(center_line)
    center_line = interpolate_param_path(param_center_line, n=5)

    # Recalculate car starting point
    track_start_pos, _ = get_closest_pose(center_line, start_pose[:2])

    """ MPCC """
    model_type = 'Kinematic Bicycle'  # 'Bicycle', 'Dynamic' or 'Kinematic Bicycle'

    # Define Initial state
    init_pose = car_odometry.init_pos
    init_vel = car_odometry.init_linear_vel
    init_w = car_odometry.init_angular_vel
    init_theta = 0.0
    theta = init_theta

    if model_type.casefold() == 'Kinematic Bicycle'.casefold():
        x0 = np.append(init_pose, [init_theta])

    elif model_type.casefold() == 'Bicycle'.casefold():
        init_delta = 0.0
        init_beta = 0.0
        x0 = np.array([init_pose[0], init_pose[1], init_delta, init_vel[0], init_pose[2], init_w[2], init_beta,
                       init_theta])

    elif model_type.casefold() == 'Dynamic'.casefold():
        x0 = np.append(init_pose, [init_vel[0], init_vel[1], init_w[2], init_theta])

    else:
        print('Please enter valid model type')
        sys.exit()

    # Create MPC object and build solver(s)
    Tf = 3
    N = 150
    tr_len = param_center_line.arc_lengths[-1]
    tr_wd = 2.75
    weights = [1e-4, 0.50, 3.5]  # [5.0, 1.0, 0.10]
    n_params = 3  # Reference pose at every instance as parameter
    mpc = MPC(model_type=model_type, horizon=Tf, shooting_nodes=N, track_length=tr_len, track_width=tr_wd, x0=x0,
              weights=weights, n_params=n_params, constrained=True)

    # Initialize variables
    stop_time = 120
    t_samp = Tf/N
    N_stop = int(stop_time/t_samp)
    opt_xi = []
    opt_ui = []
    vel_thresh = 0.3
    ec_ar = np.array([])
    el_ar = np.array([])

    rate = rospy.Rate(1/t_samp)
    current_state = x0
    closed_loop_state_traj = np.array(current_state)
    calc_vel_traj = np.array(init_vel[0])
    meas_vel_traj = np.array(init_vel[0])
    i = 0

    while not rospy.is_shutdown():

        # Calculate reference and initial guesses for solver
        if i == 0:
            ref = get_param_ref(ref_path_parameterized=param_center_line, theta=current_state[mpc.nx-1], N=N,
                                t_samp=t_samp, initial=True)
            init_x, init_u = mpc.generate_init_guess(x0=x0)
            left_bounds, right_bounds = get_bounds(ref_pts=ref, dist=tr_wd/2)
            ref_array = np.array(ref[0, :])
            left_boundary = np.array(left_bounds[0, :])
            right_boundary = np.array(right_bounds[0, :])

        else:
            ref = get_param_ref(ref_path_parameterized=param_center_line, theta=current_state[mpc.nx-1], N=N,
                                t_samp=t_samp, opt_x=opt_xi, opt_u=opt_ui)
            init_x, init_u = mpc.warm_start(t_samp=t_samp)
            left_bounds, right_bounds = get_bounds(ref_pts=ref, dist=tr_wd/2)
            ref_array = np.vstack((ref_array, ref[0, :]))
            left_boundary = np.vstack((left_boundary, left_bounds[0, :]))
            right_boundary = np.vstack((right_boundary, right_bounds[0, :]))
        # Set boundary points as parameters for all shooting nodes
        # for j in range(N+1):
        #     mpc.ocp_solver.set_params_sparse(j, left_bound_idx, left_bounds[j, :])
        #     mpc.ocp_solver.set_params_sparse(j, right_bound_idx, right_bounds[j, :])

        # Solve OCP
        if model_type.casefold() == 'Kinematic bicycle'.casefold():
            opt_xi, opt_ui, status = mpc.solve_ocp(solver=mpc.ocp_solver, x=current_state, x_init=init_x, u_init=init_u,
                                                   iteration=i, param=ref)
        elif model_type.casefold() == 'bicycle'.casefold():
            vel = current_state[3]
            if vel < vel_thresh:
                opt_xi, opt_ui, status = mpc.solve_ocp(solver=mpc.ocp_solver_init, x=current_state, x_init=init_x,
                                                       u_init=init_u, iteration=i, param=ref)
            else:
                opt_xi, opt_ui, status = mpc.solve_ocp(solver=mpc.ocp_solver, x=current_state, x_init=init_x,
                                                       u_init=init_u, iteration=i, param=ref)
        elif model_type.casefold() == 'Dynamic'.casefold():
            opt_xi, opt_ui, status = mpc.solve_ocp(solver=mpc.ocp_solver, x=current_state, x_init=init_x, u_init=init_u,
                                                   iteration=i, param=ref)
        else:
            opt_xi = np.array([])
            opt_ui = np.array([])
            status = 5
            print('OCP not solved')

        # Extract inputs
        if model_type.casefold() == 'Kinematic bicycle'.casefold():
            u1 = opt_ui[0, 0]  # Steering angle
            u2 = opt_ui[0, 1]  # Velocity
            calc_vel_traj = np.append(calc_vel_traj, u2)

        elif model_type.casefold() == 'bicycle'.casefold():
            u1 = opt_ui[0, 0]  # Steering angle rate
            u2 = opt_ui[0, 1]  # Longitudinal acceleration
        elif model_type.casefold() == 'dynamic'.casefold():
            u1 = opt_ui[0, 0]  # Longitudinal reference Velocity
            u2 = opt_ui[0, 1]  # Steering angle
        else:
            u1 = 0
            u2 = 0

        u3 = opt_ui[0, 2]  # v_theta

        # Calculate the error terms
        curr_pose = current_state[:3]
        ref_pose = ref[0, :3]
        e_c = ec(pose=curr_pose, ref_pose=ref_pose)
        e_l = el(pose=curr_pose, ref_pose=ref_pose)
        ec_ar = np.append(ec_ar, e_c)
        el_ar = np.append(el_ar, e_l)

        # Apply inputs
        if simulator:
            # Run car on f1tenth simulator
            driveMsg = generate_drive_msg(heading=u1, speed=u2)
            predMsg = generate_line_msg(opt_xi[:, :2], id=1)
            refMsg = generate_LineSegment(ref[:, :2], colors=[1.0, 0.0, 0.0], id=2)
            visPub.publish(refMsg)
            visPub.publish(predMsg)
            drivePub.publish(driveMsg)

            next_pose = car_odometry.pose
            meas_vel_traj = np.append(meas_vel_traj, car_odometry.linear_vel[0])

            theta += opt_ui[0, 2]*t_samp
            next_state = np.append(next_pose, theta)
            closed_loop_state_traj = np.vstack((closed_loop_state_traj, next_state))
            current_state = next_state

        else:
            if model_type.casefold() == 'Bicycle'.casefold():
                if current_state[3] < vel_thresh:
                    next_state = mpc.ocp.init_system.simulate(x=current_state, u=np.array([u1, u2, u3]))
                    closed_loop_state_traj = np.vstack((closed_loop_state_traj, next_state.T))
                    current_state = next_state

                else:
                    next_state = mpc.ocp.system.simulate(x=current_state, u=np.array([u1, u2, u3]))
                    closed_loop_state_traj = np.vstack((closed_loop_state_traj, next_state.T))
                    current_state = next_state

            else:
                next_state = mpc.ocp.system.simulate(x=current_state, u=np.array([u1, u2, u3]))
                closed_loop_state_traj = np.vstack((closed_loop_state_traj, next_state.T))
                current_state = next_state

        if i == N_stop or status != 0:
            N_stop = i
            break

        i += 1
        rate.sleep()

    brakeMsg = generate_drive_msg(speed=0, heading=u1)
    drivePub.publish(brakeMsg)

    t_sim = np.arange(0.0, round((N_stop+1)*t_samp, 2), t_samp)

    plt.figure()
    plt.title("Closed Loop position")
    plt.plot(closed_loop_state_traj[:, 0], closed_loop_state_traj[:, 1], ".", label="Car Position")
    plt.plot(ref_array[:, 0], ref_array[:, 1], ".", label="Reference")
    plt.plot(left_boundary[:, 0], left_boundary[:, 1])
    plt.plot(right_boundary[:, 0], right_boundary[:, 1])
    plt.xlabel("X axis")
    plt.ylabel("Y axis")
    plt.legend()

    fig1, axs = plt.subplots(3, 1)
    fig1.suptitle('State vs Time plot')
    axs[0].plot(t_sim, closed_loop_state_traj[:-1, 0], '.', label='X coordinates')
    axs[0].plot(t_sim, ref_array[:, 0], '.', label='Reference X coordinates')
    axs[0].legend()
    axs[1].plot(t_sim, closed_loop_state_traj[:-1, 1], '.', label='Y coordinates')
    axs[1].plot(t_sim, ref_array[:, 1], '.', label='Reference Y coordinates')
    axs[1].legend()
    axs[2].plot(t_sim, closed_loop_state_traj[:-1, 2], '.', label='Orientation')
    axs[2].plot(t_sim, ref_array[:, 2], '.', label='Reference Orientation')
    axs[2].legend()

    fig2, axs = plt.subplots(2,1)
    fig2.suptitle("Contouring and Lag Errors")
    axs[0].plot(t_sim, ec_ar, 'b.', label="Contouring error")
    axs[0].legend()
    axs[1].plot(t_sim, el_ar, 'r.', label="Lag error")
    axs[1].legend()

    plt.figure()
    plt.title("Measured and Calculated velocities")
    plt.plot(t_sim, calc_vel_traj[:-1], "r.", label="Calculated velocity")
    if simulator:
        plt.plot(t_sim, meas_vel_traj[:-1], 'g.', label="Measured velocity")
    plt.xlabel("Time")
    plt.ylabel("Velocity m/s")
    plt.legend()
    #
    # fig1, axs = plt.subplots(3, 1)
    # fig1.suptitle('Last optimal inputs vs Time plot')
    # axs[0].plot(t_pred[:-1], opt_ui[:, 0], '.', label='steering angle')
    # axs[0].legend()
    # axs[1].plot(t_pred[:-1], opt_ui[:, 1], '.', label='velocity')
    # axs[1].legend()
    # axs[2].plot(t_pred[:-1], opt_ui[:, 2], '.', label='Projected velocity')
    # axs[2].legend()
    # plt.legend()
    plt.show()


def mod_OCP_test():

    drivePub = rospy.Publisher('/drive', AckermannDriveStamped, queue_size=10)  # Publisher for drive messages
    visPub = rospy.Publisher('/visualization_marker', Marker, queue_size=10)  # Publisher for Marker messages

    """ EXTRACT REFERENCE"""
    center_line = get_reference_path_from_file(track="Silverstone")  # track="Monza","Silverstone", "RST_Hallway" or "Spa"

    # Get the car's starting point
    car_odometry = odom()
    car_pos = car_odometry.init_pos

    # Parameterize the reference line
    param_center_line = parameterize_ref_path(center_line)
    center_line = interpolate_param_path(param_center_line, n=5)

    # Project car starting point to the center line  top get initial reference
    s_opt = project_point(param_center_line, car_pos)
    psi_s = param_center_line.psi(s_opt)
    X_s = param_center_line.x(s_opt)
    Y_s = param_center_line.y(s_opt)

    # Visualize center line and starting point
    lineMsg = generate_line_msg(center_line, id=1, colors=[0.0, 0.0, 1.0])
    start_pose = generate_arrow_msg(np.array([X_s, Y_s, psi_s]), id=2, colors=[1.0, 0.0, 0.0])
    visPub.publish(lineMsg)
    visPub.publish(start_pose)

    # Calculate initial state
    e_y0 = sin(psi_s) * (car_pos[0] - X_s) - cos(psi_s) * (car_pos[1] - Y_s)
    e_psi0 = car_pos[2] - psi_s

    # Define initial state
    x0 = np.array([e_y0, e_psi0])

    # Create MPCC solver
    Tf = 1
    N = 20
    t_samp = Tf/N
    tr_len = param_center_line.s[-1]
    tr_wd = 1.75
    mpc = CurvatureMPC(horizon=Tf, shooting_nodes=N, track_length=tr_len, track_width=tr_wd, x0=x0, constrained=True,
                       weights=None, n_params=1, model_type='Spatial_Model')

    # Find local radius of curvature for first shooting node
    rho_s = 1/param_center_line.kappa(s_opt)

    # Create rate object
    rate = rospy.Rate(1 / t_samp)
    i = 0

    # Initialize current state with initial state
    current_state = x0
    closed_loop_state_traj = np.array(current_state)
    s = s_opt
    e_psi = e_psi0
    e_y = e_y0

    x_init = np.array(x0)
    ds_array = np.array([])
    while not rospy.is_shutdown():
        if i == 0:
            # Find local radius of curvature for N shooting nodes
            vx = np.linspace(0, 2, N)  # Assuming linearly increasing velocity
            delta = np.zeros(N)  # Assuming no steering angle
            u_init = np.column_stack((vx, delta))
            car = CarModel()
            car_model = car.kinematic_model()
            for j in range(N):
                ds = (vx[j]*rho_s*np.cos(e_psi)*t_samp)/(rho_s - e_y)
                ds_array = np.append(ds_array, ds)
                s += ds

                car_pos = car.simulate(x=car_pos, u=u_init[j, :])
                rho_s = 1/param_center_line.kappa(s)
                psi_s = param_center_line.psi(s)
                X_s = param_center_line.x(s)
                Y_s = param_center_line.y(s)
                e_y = sin(psi_s) * (car_pos[0] - X_s) - cos(psi_s) * (car_pos[1] - Y_s)
                e_psi = car_pos[2] - psi_s
                x_init = np.vstack((x_init.T, [e_y, e_psi]))

    # def Objective(s, pt, param_line):
    #     return np.linalg.norm(pt - np.array([param_line.x(s), param_line.y(s)]))
    #
    # # Initial guess for s
    # s0 = np.array([0])
    #
    # # The point to project onto the curve
    # point = np.array([car_odometry.pose[0], car_odometry.pose[1]])
    #
    # # Minimize the objective function
    # result = minimize(Objective, s0, args=(point, param_center_line))
    #
    # # The optimal arc length is stored in result.x
    # s_optimal = result.x[0]
    #
    # print(s_optimal)

    # track_start_pos, _ = get_closest_pose(center_line, start_pose[:2])
    # # reformulate the center line coordinates so that the track starts from the closest point to the car
    # center_line = reformulate_path(center_line, track_start_pos)
    # param_center_line = parameterize_ref_path(center_line)
    # n = 1
    # crv = calculate_curvature(param_center_line, n=n)
    #
    # rad_crv = 1/crv
    # tr_len = param_center_line.arc_lengths[-1]
    # s = np.linspace(0.0, n * tr_len, int(n * len(param_center_line.arc_lengths)))
    #
    # plt.scatter(s, rad_crv, c=rad_crv, cmap='viridis')
    # plt.colorbar(label='Radius of Curvature')
    # plt.show()


def MPCC_test2():

    drivePub = rospy.Publisher('/drive', AckermannDriveStamped, queue_size=10)  # Publisher for drive messages
    visPub = rospy.Publisher('/visualization_marker', Marker, queue_size=10)  # Publisher for Marker messages

    """ EXTRACT REFERENCE"""
    center_line = get_reference_path_from_file(track="RST_Hallway")  # track="Monza","Silverstone", "RST_Hallway" or "Spa"

    # Get the car's starting point
    car_odometry = odom()
    start_pose = car_odometry.init_pos

    # Parameterize the reference line
    param_center_line = parameterize_ref_path(center_line)
    center_line = interpolate_param_path(param_center_line, n=5)

    # Project car starting point to the center line  top get initial reference
    s_opt = project_point(param_center_line, start_pose)
    psi_s = param_center_line.psi(s_opt)
    X_s = param_center_line.x(s_opt)
    Y_s = param_center_line.y(s_opt)

    # Visualize center line and starting point
    lineMsg = generate_line_msg(center_line, id=1, colors=[0.0, 0.0, 1.0])
    start_pose = generate_arrow_msg(np.array([X_s, Y_s, psi_s]), id=2, colors=[1.0, 0.0, 0.0])
    visPub.publish(lineMsg)
    visPub.publish(start_pose)

    """ MPCC """
    model_type = 'Kinematic Bicycle'  # 'Bicycle', 'Dynamic' or 'Kinematic Bicycle'

    # Define Initial state
    init_pose = car_odometry.init_pos
    init_vel = car_odometry.init_linear_vel
    init_delta = 0.0
    init_v_theta = 0.0
    theta = s_opt

    if model_type.casefold() == 'Kinematic Bicycle'.casefold():
        x0 = np.append(init_pose, [theta, init_delta, init_vel[0], init_v_theta])
    else:
        x0 = None

    # Create MPC object and build solver(s)
    Tf = 3
    N = 150
    tr_len = param_center_line.s[-1]
    tr_wd = 3
    weights = [1e-6, 2.5, 5]  # [1e-4, 0.50, 0.5]  # [1e-3, 10, 2]  # [1.0, 10.0, 800.0]
    R = np.diag([1e-2, 1e-3, 1e-3])
    # R = np.diag([0.1, 1e-1, 0.1])
    n_params = 3  # Reference pose at every instance as parameter
    mpc = MPC(model_type=model_type, horizon=Tf, shooting_nodes=N, track_length=tr_len, track_width=tr_wd, x0=x0,
              weights=weights, n_params=n_params, constrained=True, R=R)

    # Initialize variables
    stop_time = 120
    t_samp = mpc.t_samp
    N_stop = int(stop_time / t_samp)

    ec_ar = np.array([])
    el_ar = np.array([])
    ref_array = np.array([])

    calc_vel_traj = np.array(init_vel[0])
    meas_vel_traj = np.array(init_vel[0])

    current_state = x0
    closed_loop_state_traj = np.array(current_state)

    opt_xi = []
    opt_ui = []

    i = 0
    rate = rospy.Rate(1 / t_samp)

    while not rospy.is_shutdown():
        # Calculate reference and initial guesses for solver
        if i == 0:
            ref = get_param_ref(ref_path_parameterized=param_center_line, theta=current_state[mpc.nx - 4], N=N,
                                t_samp=t_samp, initial=True)
            init_x, init_u = mpc.generate_init_guess(x0=x0)
            ref_array = ref[0, :3]
        else:
            ref = get_param_ref(ref_path_parameterized=param_center_line, theta=current_state[mpc.nx - 4], N=N,
                                t_samp=t_samp, opt_x=opt_xi, opt_u=opt_ui)
            init_x, init_u = mpc.warm_start()
            # init_x, init_u = mpc.generate_init_guess(x0=current_state)
            ref_array = np.vstack((ref_array, ref[0, :3]))

        # Solve OCP
        if model_type.casefold() == 'Kinematic bicycle'.casefold():
            opt_xi, opt_ui, status = mpc.solve_ocp(solver=mpc.ocp_solver, x=current_state, x_init=init_x, u_init=init_u,
                                                   iteration=i, param=ref)
        else:
            opt_xi = None
            opt_ui = None
            status = 5

        # Extract inputs
        if model_type.casefold() == 'Kinematic bicycle'.casefold():
            delta = opt_xi[1, 4]  # Steering angle
            v = opt_xi[1, 5]  # Velocity
            v_theta = opt_xi[1, 6]
            calc_vel_traj = np.append(calc_vel_traj, v)
        else:
            delta = None
            v = None
            v_theta = None

        # Calculate the error terms
        curr_pose = current_state[:3]
        ref_pose = ref[0, :3]
        e_c = ec(pose=curr_pose, ref_pose=ref_pose)
        e_l = el(pose=curr_pose, ref_pose=ref_pose)
        ec_ar = np.append(ec_ar, e_c)
        el_ar = np.append(el_ar, e_l)

        # Run car on f1tenth simulator
        driveMsg = generate_drive_msg(heading=delta, speed=v)
        predMsg = generate_line_msg(opt_xi[:, :2], id=1, colors=[0.0, 0.0, 1.0])
        refMsg = generate_LineSegment(ref[:, :2], id=2, colors=[1.0, 0.0, 0.0])
        visPub.publish(refMsg)
        visPub.publish(predMsg)
        drivePub.publish(driveMsg)

        # Start lap timing
        if i == 0:
            t_start = time.perf_counter()
        next_pose = car_odometry.pose
        meas_vel_traj = np.append(meas_vel_traj, car_odometry.linear_vel[0])

        theta += v_theta * t_samp
        if theta > tr_len:
           theta = 0.0

        if i == N_stop or status != 0:
            N_stop = i
            break

        next_state = np.append(next_pose, [theta, delta, v, v_theta])
        closed_loop_state_traj = np.vstack((closed_loop_state_traj, next_state))
        current_state = next_state

        i += 1
        rate.sleep()

    brakeMsg = generate_drive_msg(speed=0, heading=0.0)
    drivePub.publish(brakeMsg)

    # Lap analysis
    # x_lap = closed_loop_state_traj[:, 0]
    # y_lap = closed_loop_state_traj[:, 1]
    # dist = np.cumsum(np.sqrt(np.diff(x_lap) ** 2 + np.diff(y_lap) ** 2))[-1]
    # print(f'Lap distance: {dist}')
    # print(f'Reference track length: {tr_len}')
    # avg_vel = dist/t_lap
    # print(f'Average speed measured:{avg_vel}')

    # Curve Analysis

    t_sim = np.arange(0.0, round((N_stop + 1) * t_samp, 2), t_samp)

    plt.figure()
    plt.title("Closed Loop position")
    plt.plot(closed_loop_state_traj[:, 0], closed_loop_state_traj[:, 1], "b.", label="Car Position")
    plt.plot(ref_array[:, 0], ref_array[:, 1], "r.", label="Reference")
    # plt.plot(left_boundary[:, 0], left_boundary[:, 1])
    # plt.plot(right_boundary[:, 0], right_boundary[:, 1])
    plt.xlabel("X axis")
    plt.ylabel("Y axis")
    plt.legend()

    fig1, axs = plt.subplots(3, 1)
    fig1.suptitle('State vs Time plot')
    axs[0].plot(t_sim, closed_loop_state_traj[:, 0], '.', label='X coordinates')
    axs[0].plot(t_sim, ref_array[:, 0], '.', label='Reference X coordinates')
    axs[0].legend()
    axs[1].plot(t_sim, closed_loop_state_traj[:, 1], '.', label='Y coordinates')
    axs[1].plot(t_sim, ref_array[:, 1], '.', label='Reference Y coordinates')
    axs[1].legend()
    axs[2].plot(t_sim, closed_loop_state_traj[:, 2], '.', label='Orientation')
    axs[2].plot(t_sim, ref_array[:, 2], '.', label='Reference Orientation')
    axs[2].legend()

    fig2, axs = plt.subplots(2, 1)
    fig2.suptitle("Contouring and Lag Errors")
    axs[0].plot(t_sim, ec_ar, 'b.', label="Contouring error")
    axs[0].legend()
    axs[1].plot(t_sim, el_ar, 'r.', label="Lag error")
    axs[1].legend()

    plt.figure()
    plt.title("Measured and Calculated velocities")
    plt.plot(t_sim, calc_vel_traj[:-1], "r.", label="Calculated velocity")
    plt.plot(t_sim, meas_vel_traj[:-1], 'g.', label="Measured velocity")
    plt.xlabel("Time")
    plt.ylabel("Velocity m/s")
    plt.legend()

    plt.show()


def refTest():

    visPub = rospy.Publisher('/visualization_marker', Marker, queue_size=10)  # Publisher for Marker messages

    """ EXTRACT REFERENCE"""
    center_line = get_reference_path_from_file(track="RST_Track")  # track="Monza","Silverstone", "RST_Hallway" or "Spa"
    lineMsg = generate_line_msg(center_line, id=1, colors=[0.0, 0.0, 1.0])

    s = 0.0
    v_s = 1.0

    while not rospy.is_shutdown():
        visPub.publish(lineMsg)


if __name__ == '__main__':
    rospy.init_node('Test_Node', anonymous=True)
    refTest()


# def visualize_closest_point():
#
#     visPub = rospy.Publisher('/visualization_marker', Marker, queue_size=10)  # Publisher for Marker messages
#
#     center_line = get_reference_path_from_file()
#
#     car_odometry = odom()
#
#     while not rospy.is_shutdown():
#         car_pose = car_odometry.pose
#         car_position = car_pose[:2]
#
#         closest_pt, close_pts = get_closest_pose(center_line, car_position)
#
#         x = closest_pt[0]
#         y = closest_pt[1]
#         ptMsg = generate_PointMarkerMsg(x, y, id=1, colors=[1.0, 0.0, 1.0])
#         lineMsg = generate_LineSegment(close_pts, id=2, colors=[0.0, 1.0, 0.0])
#
#         visPub.publish(lineMsg)
#         visPub.publish(ptMsg)

# def solve_Nsim_OCP():
#
#     # Declare publisher(s)
#     drivePub = rospy.Publisher('/drive', AckermannDriveStamped, queue_size=10)  # Publisher for drive messages
#     visPub = rospy.Publisher('/visualization_marker', Marker, queue_size=10)  # Publisher for Marker messages
#     car_sim = CarModel(augmented=True)
#
#     # Extract reference line data from csv file
#     center_line = get_reference_path_from_file()
#
#     # Get start pose of car
#     car_odometry = odom()
#     start_pose = car_odometry.init_pos
#     track_start_pos, close_pts = get_closest_pose(center_line, start_pose[:2])
#
#     # Reformulate center line trajectory to start near starting point of car
#     center_line = reformulate_path(center_line, track_start_pos)
#
#     # Parameterize reference path
#     param_center_line = parameterize_path(center_line)
#
#     # Get track bounds from reference path
#     left_bounds, right_bounds = get_bounds(center_line, dist=0.9)
#
#     # Parameterize track bounds
#     left_bound_param = parameterize_path(left_bounds)
#     right_bound_param = parameterize_path(right_bounds)
#
#     # Interpolate track data
#     center_line_interp = interpolate_param_path(param_center_line)
#     left_bound_interp = interpolate_param_path(left_bound_param)
#     right_bound_interp = interpolate_param_path(right_bound_param)
#
#     # Visualize track bounds and reference line
#     # centerLineMsg = generate_LineSegment(center_line_interp, id=1, colors=[0.0, 1.0, 0.0])
#     # leftboundMsg = generate_LineSegment(left_bound_interp, id=2, colors=[1.0, 0.0, 0.0])
#     # rightboundMsg = generate_LineSegment(right_bound_interp, id=3, colors=[1.0, 0.0, 0.0])
#     #
#     # visPub.publish(centerLineMsg)
#     # visPub.publish(leftboundMsg)
#     # visPub.publish(rightboundMsg)
#
#     # Create MPCC solver
#     Tf = 2
#     N = 20
#     model, constraints, solver = ContouringControlOCP(horizon=Tf, shooting_nodes=N, start_pose=start_pose,
#                                                       track_length=param_center_line.arc_lengths[-1], constrained=True)
#
#     # State and input dimensions
#     nx = model.x.size()[0]
#     nu = model.u.size()[0]
#
#     # Initial progress on curve
#     theta = 0
#
#     opt_xi = []
#     opt_ui = []
#
#     ref_idx = np.array([0, 1, 2, 3])
#     left_bound_idx = np.array([4, 5])
#     right_bound_idx = np.array([6, 7])
#
#     t_samp = 100e-3
#     Nsim = 1
#     rate = rospy.Rate(int(1/t_samp))
#     for i in range(Nsim):
#         car_pose = car_odometry.pose
#         x = np.append(car_pose, theta)
#
#         if i == 0:
#             # x = np.append(start_pose, theta)
#             # Get initial reference
#             ref = get_param_ref(x=x, t_samp=t_samp, N=N, initial=True, ref_path_parameterized=param_center_line)
#             refMsg = generate_LineSegment(ref[:, :2], id=1, colors=[0.0, 1.0, 0.0])
#             visPub.publish(refMsg)
#
#             # Get boundary values for initial reference
#             # xleft = left_bound_param.x(ref[:, 3])
#             # yleft = left_bound_param.y(ref[:, 3])
#             # track_left = np.column_stack([xleft, yleft])
#             track_left, track_right = get_bounds(ref[:, :2], dist=1.0)
#
#             # xright = right_bound_param.x(ref[:, 3])
#             # yright = right_bound_param.y(ref[:, 3])
#             # track_right = np.column_stack([xright, yright])
#             track_leftMsg = generate_LineSegment(track_left, id=2, colors=[1.0, 0.0, 0.0])
#             visPub.publish(track_leftMsg)
#
#             track_rightMsg = generate_LineSegment(track_right, id=3, colors=[1.0, 0.0, 0.0])
#             visPub.publish(track_rightMsg)
#
#             # Create first Initial guess for solver
#             x_init, u_init = generate_init_guess(x, N, max_speed=4.0, augmented=True)
#             # x_init = pd.read_csv('init_x_0.csv').values[:, 1:]
#             # u_init = pd.read_csv('init_u_0.csv').values[:, 1:]
#
#         else:
#             # Get reference from previous optimal control values
#             ref = get_param_ref(x=x, t_samp=t_samp, N=N, initial=False, ref_path_parameterized=param_center_line,
#                                 opt_x=opt_xi, opt_u=opt_ui)
#             refMsg = generate_LineSegment(ref[:, :2], id=1, colors=[0.0, 1.0, 0.0])
#             visPub.publish(refMsg)
#
#             # Get boundary values for initial reference
#             # xleft = left_bound_param.x(ref[:, 3])
#             # yleft = left_bound_param.y(ref[:, 3])
#             # track_left = np.column_stack([xleft, yleft])
#             track_left, track_right = get_bounds(ref[:, :2], dist=1.0)
#
#             track_leftMsg = generate_LineSegment(track_left, id=2, colors=[1.0, 0.0, 0.0])
#             visPub.publish(track_leftMsg)
#
#             # xright = right_bound_param.x(ref[:, 3])
#             # yright = right_bound_param.y(ref[:, 3])
#             # track_right = np.column_stack([xright, yright])
#             track_rightMsg = generate_LineSegment(track_right, id=3, colors=[1.0, 0.0, 0.0])
#             visPub.publish(track_rightMsg)
#
#             # Warm start solver
#             x_init, u_init = warm_start(opt_xi, opt_ui, nx, augmented=True, t_samp=t_samp)
#
#         for j in range(N):
#             # Set reference as a parameter at every shooting node
#             solver.set_params_sparse(j, ref_idx, ref[j, :])
#
#             # Set left and right bounds as parameter for constraint evaluation at every node
#             solver.set_params_sparse(j, left_bound_idx, track_left[j, :])
#             solver.set_params_sparse(j, right_bound_idx, track_right[j, :])
#
#             if j == 0:
#                 # Set current state as initial state at first shooting node
#                 solver.set(0, 'lbx', x)
#                 solver.set(0, 'ubx', x)
#
#         solver.set_params_sparse(N, ref_idx, ref[N, :])
#         solver.set_params_sparse(N, left_bound_idx, track_left[N, :])
#         solver.set_params_sparse(N, right_bound_idx, track_right[N, :])
#
#         set_init_guess(solver, init_x=x_init, init_u=u_init, N=N)
#
#         status = solver.solve()  # Solve the OCP
#
#         if status != 0:
#             # print("acados returned status {} in closed loop iteration {}.".format(status, i))
#             solver.print_statistics()
#         else:
#             print('Optimal Solution found')
#             solver.print_statistics()
#
#         opt_xi, opt_ui = get_solution(solver, N, nx, nu)
#         predMsg = generate_LineSegment(opt_xi[:, :2], id=4, colors=[0.0, 0.0, 1.0])
#         visPub.publish(predMsg)
#
#         # opt_x_df = pd.DataFrame(opt_xi)
#         # opt_u_df = pd.DataFrame(opt_ui)
#         #
#         # opt_x_df.to_csv('init_x_0.csv')
#         # opt_u_df.to_csv('init_u_0.csv')
#
#         delta = opt_ui[0, 0]  # Extract heading angle from the first optimal input
#         vel = opt_ui[0, 1]  # Extract velocity from first optimal input
#         v_theta = opt_ui[0, 2]
#         theta += v_theta*t_samp
#
#         driveMsg = generate_drive_msg(vel, delta)  # Create drive message from calculated velocity and heading
#         drivePub.publish(driveMsg)  # Send message to f1tenth car
#
#         i += 1
#         rate.sleep()
#
#     # for i in range(len(opt_xi)):
#     #     predMsg = generate_arrow_msg(opt_xi[i, :3], id=i, colors=[0.0, 1.0, 0.0])
#     #     visPub.publish(predMsg)
#
#     # for i in range(len(opt_ui)):
#     #     delta = opt_ui[i, 0]
#     #     vel = opt_ui[i, 1]
#     #     v_theta = opt_ui[i, 2]
#     #
#     #     driveMsg = generate_drive_msg(vel, delta)  # Create drive message from calculated velocity and heading
#     #     drivePub.publish(driveMsg)  # Send message to f1tenth car
# def solve_n_OCPs(n=1):
    #
    #     drivePub = rospy.Publisher('/drive', AckermannDriveStamped, queue_size=10)  # Publisher for drive messages
    #     visPub = rospy.Publisher('/visualization_marker', Marker, queue_size=10)  # Publisher for Marker messages
    #
    #     # Get center line as reference path from file
    #     center_line = get_reference_path_from_file()
    #
    #     # Get start pose of car
    #     car_odometry = odom()
    #     start_pose = car_odometry.init_pos
    #     track_start_pos, close_pts = get_closest_pose(center_line, start_pose[:2])
    #
    #     # Reformulate center line trajectory to start near starting point of car
    #     center_line = reformulate_path(center_line, track_start_pos)
    #
    #     # Parameterize reference path
    #     param_center_line = parameterize_path(center_line)
    #
    #     # Create MPCC solver
    #     Tf = 1
    #     N = 20
    #     track_width = 1.6
    #     model, constraints, solver = ContouringControlOCP(horizon=Tf, shooting_nodes=N, start_pose=start_pose,
    #                                                  track_length=param_center_line.arc_lengths[-1], constrained=True,
    #                                                  constr_type='contouring error', dmax=track_width/2)
    #
    #     # State and input dimensions
    #     nx = model.x.size()[0]
    #     nu = model.u.size()[0]
    #
    #     # Initial progress on curve
    #     theta = 0
    #     v_theta = 0
    #
    #     opt_xi = []
    #     opt_ui = []
    #
    #     ref_idx = np.array([0, 1, 2, 3])
    #     left_bound_idx = np.array([4, 5])
    #     right_bound_idx = np.array([6, 7])
    #
    #     t_samp = 50e-3
    #     Nsim = n
    #     rate = rospy.Rate(int(1 / t_samp))
    #
    #     for i in range(Nsim):
    #         # Get current state
    #         car_pose = car_odometry.pose
    #         theta += v_theta * t_samp
    #         x = np.append(car_pose, theta)
    #
    #         if i == 0:
    #             # Calculate reference for current state
    #             ref = get_param_ref(x=x, t_samp=t_samp, N=N, initial=True, ref_path_parameterized=param_center_line)
    #
    #             # Visualize current reference
    #             refMsg = generate_LineSegment(ref[:, :2], id=1, colors=[0.0, 1.0, 0.0])
    #             visPub.publish(refMsg)
    #
    #             # Calculate boundary points for each reference point
    #             left_bound, right_bound = get_bounds(ref_pts=ref[:, :2], dist=track_width/2)
    #
    #             # Visualize current boundary points
    #             leftboundMsg = generate_LineSegment(left_bound, id=2, colors=[1.0, 0.0, 0.0])
    #             visPub.publish(leftboundMsg)
    #             rightboundMsg = generate_LineSegment(right_bound, id=3, colors=[1.0, 0.0, 0.0])
    #             visPub.publish(rightboundMsg)
    #
    #         else:
    #             ref = get_param_ref(x=x, t_samp=t_samp, N=N, initial=False, ref_path_parameterized=param_center_line,
    #                                 opt_x=opt_xi, opt_u=opt_ui)
    #
    #             # Visualize current reference
    #             refMsg = generate_LineSegment(ref[:, :2], id=1, colors=[0.0, 1.0, 0.0])
    #             visPub.publish(refMsg)
    #
    #             # Calculate boundary points for each reference point
    #             left_bound, right_bound = get_bounds(ref_pts=ref[:, :2], dist=track_width/2)
    #
    #             # Visualize current boundary points
    #             leftboundMsg = generate_LineSegment(left_bound, id=2, colors=[1.0, 0.0, 0.0])
    #             visPub.publish(leftboundMsg)
    #             rightboundMsg = generate_LineSegment(right_bound, id=3, colors=[1.0, 0.0, 0.0])
    #             visPub.publish(rightboundMsg)
    #
    #         for j in range(N):
    #             # Set reference and track bounds as parameters at first and intermediate shooting nodes
    #             solver.set_params_sparse(j, ref_idx, ref[j, :])
    #             solver.set_params_sparse(j, left_bound_idx, left_bound[j, :])
    #             solver.set_params_sparse(j, right_bound_idx, right_bound[j, :])
    #
    #             if j == 0:
    #                 # Set current state as initial state at first shooting node
    #                 solver.set(0, 'lbx', x)
    #                 solver.set(0, 'ubx', x)
    #
    #         # Set reference and bounds as parameters at final shooting node
    #         solver.set_params_sparse(N, ref_idx, ref[N, :])
    #         solver.set_params_sparse(N, left_bound_idx, left_bound[N, :])
    #         solver.set_params_sparse(N, right_bound_idx, right_bound[N, :])
    #
    #         # Set solver initial guess
    #         if i == 0:
    #             # x_init = pd.read_csv('init_x_0.csv').values[:, 1:]
    #             # u_init = pd.read_csv('init_u_0.csv').values[:, 1:]
    #             x_init, u_init = generate_init_guess(x, N, max_speed=4.0, augmented=True)
    #             set_init_guess(solver=solver, init_x=x_init, init_u=u_init, N=N)
    #         else:
    #             x_init, u_init = warm_start(opt_xi, opt_ui, nx, augmented=True, t_samp=t_samp)
    #             set_init_guess(solver=solver, init_x=x_init, init_u=u_init, N=N)
    #
    #         # Solve OCP
    #         status = solver.solve()  # Solve the OCP
    #
    #         if status != 0:
    #             print("acados returned status {} in closed loop iteration {}.".format(status, i))
    #             solver.print_statistics()
    #
    #             constraint_value = []
    #             for k in range(N + 1):
    #                 constr_eval = constraints.function(opt_xi[k, :], ref[k, :]).full()
    #
    #                 if constraints.constraint_type.casefold() == 'Polyhedral'.casefold():
    #                     bmin = constraints.bmin(ref[k, :], left_bound[k, 0], left_bound[k, 1], right_bound[k, 0],
    #                                             right_bound[k, 1]).full()
    #                     bmax = constraints.bmax(ref[k, :], left_bound[k, 0], left_bound[k, 1], right_bound[k, 0],
    #                                             right_bound[k, 1]).full()
    #                     if bmin > constr_eval or bmax < constr_eval:
    #                         print(f'Constraint violated at shooting node {k} at SQP iteration {i}')
    #                         print(f'evaluated constraint value {constr_eval}')
    #                         print(f'Track bound maximum{bmax}')
    #                         print(f'Track bound minimum{bmin}')
    #
    #                 constraint_value.append(constr_eval)
    #             constraint_value = np.array(constraint_value).reshape(1, N+1)
    #             print(constraint_value)
    #             print(track_width/2)
    #
    #             brakeMsg = generate_drive_msg(speed=0.0, heading=0.0)
    #             drivePub.publish(brakeMsg)
    #             break
    #         else:
    #             print('Optimal Solution found')
    #             solver.print_statistics()
    #
    #         opt_xi, opt_ui = get_solution(solver, N, nx, nu)
    #         predMsg = generate_LineSegment(opt_xi[:, :2], id=4, colors=[0.0, 0.0, 1.0])
    #         visPub.publish(predMsg)
    #
    #         delta = opt_ui[0, 0]  # Extract heading angle from the first optimal input
    #         vel = opt_ui[0, 1]  # Extract velocity from first optimal input
    #         v_theta = opt_ui[0, 2]
    #
    #         rate.sleep()
    #
    #         # plot_optimal_input_trajectories(opt_ui, t_samp)
    #
    #         driveMsg = generate_drive_msg(vel, delta)  # Create drive message from calculated velocity and heading
    #         drivePub.publish(driveMsg)  # Send message to f1tenth car
    #
    #         print(car_odometry.linear_vel)
    #         print(car_odometry.angular_vel)
    #
    #         # brakeMsg = generate_drive_msg(speed=0.0, heading=delta)
    #         # drivePub.publish(brakeMsg)
    #
    #         i+=1
    #
    #
    # def MPCC_test(n=1):
    #
    #     drivePub = rospy.Publisher('/drive', AckermannDriveStamped, queue_size=10)  # Publisher for drive messages
    #     visPub = rospy.Publisher('/visualization_marker', Marker, queue_size=10)  # Publisher for Marker messages
    #
    #     """ REFERENCE PATH """
    #     # Get center line as reference path from file
    #     center_line = get_reference_path_from_file()
    #
    #     # Parameterize and interpolate reference path
    #     param_center_line = parameterize_path(center_line)
    #     center_line = interpolate_param_path(param_center_line, n=5)
    #
    #     # Get start pose of car and set the closest point to it on the track as the starting point of the track
    #     car_odometry = odom()
    #     start_pose = car_odometry.init_pos
    #     track_start_pos, close_pts = get_closest_pose(center_line, start_pose[:2])
    #
    #     # reformulate the center line coordinates so that the track starts from the closest point to the car
    #     center_line = reformulate_path(center_line, track_start_pos)
    #
    #     # Re-parameterize center line so that arc length at starting position is = 0
    #     param_center_line = parameterize_path(center_line)
    #
    #     """ MPCC """
    #     # Create MPCC solver
    #     Tf = 1
    #     N = 20
    #     track_width = 1.6
    #     model_type = 'bicycle'
    #     ocp = OCP(shooting_nodes=N, horizon=Tf)
    #
    #     if model_type.casefold() == 'Dynamic'.casefold():
    #         nx = 7
    #     elif model_type.casefold() == 'bicycle'.casefold():
    #         nx = 8
    #     elif model_type.casefold() == 'kinematic'.casefold():
    #         nx = 3
    #     else:
    #         nx = 3
    #
    #     nu = 3
    #
    #     x0 = np.zeros(nx)
    #     x0[:2] = start_pose[:2]  # Initial pose
    #     x0[4] = start_pose[2]
    #     tr_len = param_center_line.arc_lengths[-1]
    #     model, constraints, solver = ocp.MPCC_OCP(model_type=model_type)
    #
    #     # Initialize variables
    #     opt_xi = []
    #     opt_ui = []
    #
    #     opt_x = []
    #     opt_u = []
    #
    #     t_samp = 50e-3  # 50ms
    #     Nsim = n
    #     rate = rospy.Rate(int(1 / t_samp))
    #
    #     # Start moving car
    #     while car_odometry.linear_vel[0] < 0.3:
    #         moveMsg = generate_drive_msg(speed=0.3, heading=0.0)
    #         drivePub.publish(moveMsg)
    #         rate.sleep()
    #
    #     current_pose = car_odometry.pose
    #     track_start_pos, close_pts = get_closest_pose(center_line, current_pose[:2])
    #
    #     # reformulate the center line coordinates so that the track starts from the closest point to the car
    #     center_line = reformulate_path(center_line, track_start_pos)
    #
    #     # Re-parameterize center line so that arc length at starting position is = 0
    #     param_center_line = parameterize_path(center_line)
    #
    #     theta = 0.0
    #     v_theta = 0.0
    #     for i in range(Nsim):
    #
    #         # MEASURE STATES
    #         x = measure_states(car_odometry, theta, v_theta, t_samp, p=model.p)
    #         theta = x[nx-1]
    #         # PREPARE SOLVER
    #         if i == 0:
    #             # EXTRACT INITIAL REFERENCE
    #             pose_ref = get_param_ref(ref_path_parameterized=param_center_line, theta=theta, initial=True, N=N,
    #                                      t_samp=t_samp)
    #         else:
    #             # EXTRACT REFERENCE DEPENDING ON 'theta' value
    #             pose_ref = get_param_ref(ref_path_parameterized=param_center_line, theta=theta, initial=False, N=N,
    #                                      t_samp=t_samp, opt_x=opt_xi, opt_u=opt_ui)
    #
    #         refMsg = generate_LineSegment(pose_ref[:, :2], id=1, colors=[0.0, 1.0, 0.0])
    #         visPub.publish(refMsg)
    #
    #         # SET REFERENCE POSES AS PARAMETERS IN OCP FOR COST EVALUATION
    #         for j in range(N):
    #             if j == 0:
    #                 # Set current state as initial state at first shooting node
    #                 solver.set(0, 'lbx', x)
    #                 solver.set(0, 'ubx', x)
    #
    #             solver.set(j, 'p', pose_ref[j, :])
    #         solver.set(N, 'p', pose_ref[N, :])
    #
    #         # SET INITIAL GUESS
    #         if i == 0:
    #             # x_init = pd.read_csv('init_x_0.csv').values[:, 1:]
    #             # u_init = pd.read_csv('init_u_0.csv').values[:, 1:]
    #             x_init, u_init = generate_init_guess(x, N, ref=pose_ref, model=model_type, max_speed=2.0, augmented=True,
    #                                                  nx=nx, p=model.p)
    #             ocp.set_init_guess(solver=solver, init_x=x_init, init_u=u_init, N=N)
    #         else:
    #             x_init, u_init = warm_start(opt_xi, opt_ui, model=model_type, nx=nx, augmented=True, t_samp=t_samp, p=model.p)
    #             ocp.set_init_guess(solver=solver, init_x=x_init, init_u=u_init, N=N)
    #
    #         # SOLVE OCP/ OPTIMIZE
    #         status = solver.solve()  # Solve the OCP
    #
    #         if status != 0:
    #             print("acados returned status {} in closed loop iteration {}.".format(status, i))
    #             solver.print_statistics()
    #             brakeMsg = generate_drive_msg(speed=0.0, heading=0.0)
    #             drivePub.publish(brakeMsg)
    #
    #         else:
    #             pass
    #             print("Optimal solution found")
    #             # solver.print_statistics()
    #
    #         # EXTRACT INPUT TO CAR
    #         opt_xi, opt_ui = ocp.get_solution(solver, N, nx, nu)
    #         predMsg = generate_LineSegment(opt_xi[:, :2], id=4, colors=[0.0, 0.0, 1.0])
    #         visPub.publish(predMsg)
    #
    #         opt_x.append(opt_xi)
    #         opt_u.append(opt_ui)
    #
    #         vel = opt_xi[1, 3]  # Extract velocity
    #         delta = opt_xi[1, 2]  # Extract heading angle
    #         v_theta = opt_ui[0, 2]
    #
    #         # APPLY INPUT
    #         driveMsg = generate_drive_msg(speed=vel, heading=delta)
    #         drivePub.publish(driveMsg)
    #
    #         i += 1
    #         rate.sleep()

    #     if i == 0:
    #         # calculate initial reference and initial guess
    #         ref_path = get_param_ref(ref_path_parameterized=param_center_line, theta=state[nx-1], N=N, t_samp=t_samp,
    #                                  initial=True)
    #         x_init, u_init = generate_init_guess(x=state, N=N, max_speed=2.0, vel_meas=state[3], nx=nx,
    #                                              vel_thresh=vel_thresh, params=model.p)
    #
    #         ref_path_array = ref_path[0, :]
    #     else:
    #         # Calculate reference and warm-start solver
    #         ref_path = get_param_ref(ref_path_parameterized=param_center_line, theta=state[nx-1], N=N, t_samp=t_samp,
    #                                  initial=False, opt_u=opt_ui)
    #         x_init, u_init = warm_start(opt_x=opt_xi, opt_u=opt_ui, params=model.p, vel_meas=state[3],
    #                                     vel_thresh=vel_thresh)
    #         ref_path_array = np.vstack([ref_path_array, ref_path[0, :]])
    #
    #     # Solve OCP
    #     if state[3] < vel_thresh:
    #         opt_xi, opt_ui, status = ocp.solve_ocp(x=state, solver=solver_init, N=N, nx=nx, nu=nu, x_init=x_init,
    #                                                u_init=u_init, param=ref_path, iteration=i)
    #     else:
    #         opt_xi, opt_ui, status = ocp.solve_ocp(x=state, solver=solver, N=N, nx=nx, nu=nu, x_init=x_init,
    #                                                u_init=u_init, param=ref_path, iteration=i)
    #
    #     # if status != 0:
    #     #     break
    #
    #     # Extract inputs
    #     v_delta = opt_ui[0, 0]
    #     a_long = opt_ui[0, 1]
    #     v_theta = opt_ui[0, 2]
    #
    #     vel_long = opt_xi[1, 3]
    #     steer_ang = opt_xi[1, 2]
    #
    #     # Simulate next state and set it as current state for next iteration
    #     inputs = np.array([v_delta, a_long, v_theta])
    #     car_model = CarModel(model=model_type, p=model.p, vel=state[3], vel_thresh=vel_thresh)
    #     next_state = car_model.simulate(state, inputs)
    #     state = next_state.reshape(nx,)
    #
    #     state_traj_closed_loop = np.vstack([state_traj_closed_loop, state.T])
    #
    #     i += 1
    #     rate.sleep()
    #
    # # if status!=0:
    # #     t = np.linspace(0, i*t_samp, i)
    # #     X_subopt = opt_xi[:, 0]
    # #     Y_subopt = opt_xi[:, 1]
    # #     psi_subopt = opt_xi[:, 4]
    # #
    # #     print(X_subopt)
    # #     print(t)
    # plot_sim_trajectories(state_traj_closed_loop, ref_path_array, t_samp)

# Set reference as a parameter at every shooting node
        # if vel < vel_thresh:
        #     for j in range(N):
        #         if j == 0:
        #             # Set current state as initial state at first shooting node
        #             solver_init.set(0, 'lbx', state)
        #             solver_init.set(0, 'ubx', state)
        #
        #         # Intermediate shooting nodes
        #         solver_init.set(j, 'p', ref_path[j, :])
        #     # Terminal shooting node
        #     solver_init.set(N, 'p', ref_path[N, :])
        #
        # else:
        #     for j in range(N):
        #         if j == 0:
        #             # Set current state as initial state at first shooting node
        #             solver.set(0, 'lbx', state)
        #             solver.set(0, 'ubx', state)
        #
        #         # Intermediate shooting nodes
        #         solver.set(j, 'p', ref_path[j, :])
        #     # Terminal shooting node
        #     solver.set(N, 'p', ref_path[N, :])
        #
        # # Generate initial guess for solver
        # if i == 0:
        #     x_init, u_init = generate_init_guess(x=state, N=N, max_speed=2.0,vel_meas=vel, nx=nx,
        #                                          vel_thresh=vel_thresh, params=model.p)
        # else:
        #     x_init, u_init = warm_start(opt_x=opt_xi, opt_u=opt_ui, params=model.p, vel_meas=vel, vel_thresh=vel_thresh)
        #
        # # Set initial guess and solve OCP
        # if vel < vel_thresh:
        #     ocp.set_init_guess(solver_init, init_x=x_init, init_u=u_init, N=N)
        #     status = solver_init.solve()
        # else:
        #     ocp.set_init_guess(solver, init_x=x_init, init_u=u_init, N=N)
        #     status = solver.solve()  # Solve the OCP
        #
        # # Check solver status
        # if status != 0:
        #     print("acados returned status {} in closed loop iteration {}.".format(status, i))
        #     solver.print_statistics()
        # else:
        #     pass
        #     print("Optimal solution found")
        #     # solver.print_statistics()
        #
        # Extract solution
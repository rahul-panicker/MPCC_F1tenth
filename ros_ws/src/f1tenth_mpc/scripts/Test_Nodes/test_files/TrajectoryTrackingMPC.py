#!/usr/bin/env python3

import pandas as pd
import time
from casadi import *

# ROS imports
import rospkg
import rospy
from visualization_msgs.msg import Marker
from ackermann_msgs.msg import AckermannDriveStamped

# Helper scripts
from ..Helper_scripts.referenceParser import create_traj_from_path, get_ref
from ..Helper_scripts.generate_vizMarkerMsg import generate_LineSegment, generate_PointMarkerMsg
from ..Controller import *
from ..Helper_scripts.RCDriver import generate_drive_msg
from ..Helper_scripts.GetStates import odom


def main():

    rospy.init_node('Trajectory_tracking_MPC', anonymous=True)

    # Publishers
    drivePub = rospy.Publisher('/drive', AckermannDriveStamped, queue_size=10)

    refmarkerPub = rospy.Publisher('/visualization_marker', Marker, queue_size=10)
    predStatePub = rospy.Publisher('/visualization_marker', Marker, queue_size=10)
    TerminalStatePub = rospy.Publisher('/visualization_marker', Marker, queue_size=10)

    # Extract reference line data
    rospack = rospkg.RosPack()
    track = 'Silverstone'
    pkg_path = rospack.get_path('f1tenth_simulator')
    file_path = pkg_path + f'/scripts/Additional_maps/{track}/{track}_centerline.csv'

    df = pd.read_csv(file_path)
    center_line = df.iloc[:, :2].values

    # Create reference trajectory from reference path with linear interpolation
    car_odometry = odom()
    rospy.sleep(0.5)
    t_samp = 0.05
    start_point = car_odometry.init_pos
    ref_traj, vel = create_traj_from_path(center_line, start_point, ts=0.1, interpolate=True, ts_interp=t_samp)

    # import matplotlib.pyplot as plt
    # plt.plot(ref_traj[:, 0], ref_traj[:, 1],".")
    # plt.show()

    """=============================================== MPC Loop ====================================================="""
    # ACADOS solver creation
    Tf = 1
    N = 10
    constraint, model, acados_solver = TrajectoryTrackingOCP(Tf=Tf, N=N)

    # Initialize variables
    tcomp_max = 0
    tcomp_sum = 0

    nx = 3
    nu = 2

    opt_x = []
    opt_u = []

    opt_xi = []
    opt_ui = []

    speed = vel * np.ones(N)
    # speed = generate_vel_profile(ref_traj, k=1, tsamp=t_samp)
    i = 0

    rate = rospy.Rate(int(1/t_samp))
    while not rospy.is_shutdown():

        # Check for lap completion
        if i % len(ref_traj) == 0:
            i = 0

        x = car_odometry.pose  # Get current car pose

        x_ref, u_ref = get_ref(i, N, speed, ref_traj, const_vel=True)  # Get reference for current horizon

        # print("Current car pose:", x)
        # print("Reference pose:", x_ref[0, :])
        # print("Distance:", np.sqrt((x_ref[0, 0] - x[0])**2 + (x_ref[0, 1] - x[1])**2))

        for j in range(N):
            # Assign reference for each shooting node (from 0 to N-1)
            yref = np.concatenate([x_ref[j, :], u_ref[j, :]])
            acados_solver.set(j, 'yref', yref)

            if j == 0:
                # Set current state as initial state at first shooting node at every iteration of MPC loop
                acados_solver.set(0, 'lbx', x)
                acados_solver.set(0, 'ubx', x)
            else:
                pass
                # Set track bounds as constraints for every other shooting node
                # X = x[0]
                # Y = x[1]
                #
                # X_ref = x_ref[j, 0]
                # Y_ref = x_ref[j, 1]
                # theta_ref = x_ref[j, 2]
                #
                # constraint.expr = (X - X_ref)*sin(theta_ref) - (Y-Y_ref)*cos(theta_ref)
                # acados_solver.acados_ocp.model.con_h_expr = constraint.expr
                # print(constraint.expr)
                # acados_solver.constraints_set(j, 'lh', np.array([-1.1]))
                # acados_solver.constraints_set(j, 'uh', np.array([1.1]))

        # Assign reference to terminal shooting node
        yref_N = x_ref[N, :]
        acados_solver.set(N, "yref", yref_N)

        # Set constraints at final shooting node

        # Set initial guess (warm start)
        if i == 0:
            init_x, init_u = generate_init_guess(x, N, speed[0])
            set_init_guess(acados_solver, init_x, init_u, N)
        else:
            init_x, init_u = warm_start(opt_xi, opt_ui, nx, nu)
            set_init_guess(acados_solver, init_x, init_u, N)

        t = time.time()  # Start timing

        status = acados_solver.solve()  # Solve the OCP
        # acados_solver.print_statistics()

        elapsed = time.time() - t  # Time taken to solve the OCP
        tcomp_sum += elapsed  # Total computation time
        if elapsed > tcomp_max:  # Assign maximum computation time
            tcomp_max = elapsed

        if status != 0:
            print("acados returned status {} in closed loop iteration {}.".format(status, i))
            acados_solver.print_statistics()
        else:
            print('Optimal Solution found')
            acados_solver.print_statistics()

        # Extract solution
        opt_xi = []
        opt_ui = []

        for k in range(N + 1):
            optx_k = acados_solver.get(k, "x")
            opt_xi = np.concatenate((opt_xi, optx_k))

        opt_xi = opt_xi.reshape(N + 1, nx)  # Optimal state trajectory at iteration i
        opt_x = np.append(opt_x, opt_xi).reshape((i + 1) * (N + 1), nx)  # Optimal state trajectories for all iterations

        for k in range(N):
            optu_k = acados_solver.get(k, "u")
            opt_ui = np.concatenate((opt_ui, optu_k))

        opt_ui = opt_ui.reshape(N, nu)  # Optimal input trajectory at iteration i
        opt_u = np.append(opt_u, opt_ui).reshape((i + 1) * N, nu)  # Optimal input trajectories  for all iterations

        delta = opt_ui[0, 0]  # Extract heading angle from the first optimal input
        vel = opt_ui[0, 1]  # Extract velocity from first optimal input

        driveMsg = generate_drive_msg(vel, delta)  # Create drive message from calculated velocity and heading
        drivePub.publish(driveMsg)  # Send message to f1tenth car

        # Visualize predictions
        # predictedStates = generate_LineSegment(opt_xi[:, :2], id=5, colors=[1.0, 0.0, 0.0], scale=[0.05, 0.05, 0.05])
        # predictedTerminalState = generate_arrow_msg(opt_xi[N, :3], id=6, colors=[1.0, 0.0, 0.0])
        currentState = generate_PointMarkerMsg(x[0], x[1], id=7, colors=[1.0, 1.0, 0.0])
        currentReference = generate_LineSegment(x_ref[0:N, :2], id=8, colors=[0.0, 0.0, 1.0])

        predStatePub.publish(currentState)
        TerminalStatePub.publish(currentReference)

        # arrowMsg = generate_arrow_msg(ref_traj[i, :3], id=i, namespace="Arrows", colors=[1.0, 0.0, 0.0])
        # refmarkerPub.publish(arrowMsg)

        i += 1
        rate.sleep()


if __name__ == '__main__':
    main()
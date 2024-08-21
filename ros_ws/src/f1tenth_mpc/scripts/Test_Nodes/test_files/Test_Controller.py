#!/usr/bin/env python3
from casadi import *

# ROS Imports
import rospy

from ackermann_msgs.msg import AckermannDriveStamped
from visualization_msgs.msg import Marker


# Helper scripts and nodes
from Helper_scripts.GetStates import odom
from Helper_scripts.referenceParser import (get_reference_path_from_file, parameterize_path, get_closest_pose,
                                            get_param_ref, reformulate_path)
from Helper_scripts.RCDriver import generate_drive_msg
from Helper_scripts.generate_vizMarkerMsg import generate_LineSegment

from Controller import ContouringControlOCP, generate_init_guess, warm_start, set_init_guess, get_solution


def main():
    # Initialize ROS node
    rospy.init_node('Model_Predictive_Contouring_Control', anonymous=True)

    # Declare publisher(s)
    drivePub = rospy.Publisher('/drive', AckermannDriveStamped, queue_size=10)  # Publisher for drive messages
    visPub = rospy.Publisher('/visualization_marker', Marker, queue_size=10)  # Publisher for Marker messages

    # Extract reference line data from csv file
    center_line = get_reference_path_from_file()

    # Get start pose of car
    car_odometry = odom()
    start_pose = car_odometry.init_pos
    track_start_pos, _ = get_closest_pose(center_line, start_pose[:2])

    if not np.array_equal(track_start_pos, center_line[0, :]):
        center_line = reformulate_path(center_line, track_start_pos)

    # Parametrize reference path
    param_center_line = parameterize_path(center_line)

    # Create MPCC solver
    Tf = 1
    N = 20
    model, constraints, solver = ContouringControlOCP(horizon=Tf, shooting_nodes=N, start_pose=start_pose)

    # State and input dimensions
    nx = model.x.size()[0]
    nu = model.u.size()[0]

    # Timing variables
    tcomp_max = 0  # Used to store maximum computation time
    tcomp_sum = 0  # Used to store total computation time
    t_samp = 0.05  # Closed loop sampling time for MPCC algorithm

    # Variables to store optimal state and input values
    opt_xi = []
    opt_ui = []

    # Initial progress on curve
    theta = 0
    """=============================================== MPC Loop ====================================================="""
    i = 0
    rate = rospy.Rate(int(1 / t_samp))

    while not rospy.is_shutdown():

        # Get current car pose from odometry
        car_pose = car_odometry.pose
        # Calculate the evolution of theta
        # theta += v_theta*t_samp

        x = np.append(car_pose, theta)

        if i == 0:
            # Get initial reference
            ref = get_param_ref(x=x, t_samp=t_samp, N=N, initial=True, ref_path_parameterized=param_center_line)

            # Create first Initial guess for solver
            x_init, u_init = generate_init_guess(x, N, max_speed=4.0, augmented=True)

        else:
            # Get reference from previous optimal control values
            ref = get_param_ref(x=x, t_samp=t_samp, N=N, initial=True, ref_path_parameterized=param_center_line,
                                opt_x=opt_xi, opt_u=opt_ui)
            refLineMsg = generate_LineSegment(ref[:, :2], id=1, colors=[1.0, 0.0, 0.0])
            visPub.publish(refLineMsg)

            # Warm start solver
            x_init, u_init = warm_start(opt_xi, opt_ui, nx, augmented=True, t_samp=t_samp)

        for j in range(N):
            # Set reference as a parameter at every shooting node
            solver.set(j, 'p', ref[j, :])

            if j == 0:
                # Set current state as initial state at first shooting node
                solver.set(0, 'lbx', x)
                solver.set(0, 'ubx', x)
            else:
                counter = 0
                # Set other track constraints

        # Set reference as parameter at terminal node
        solver.set(N, 'p', ref[N, :])

        # Set initial guess for solver
        set_init_guess(solver, init_x=x_init, init_u=u_init, N=N)

        status = solver.solve()  # Solve the OCP

        if status != 0:
            print("acados returned status {} in closed loop iteration {}.".format(status, i))
            solver.print_statistics()
        else:
            print('Optimal Solution found')
            solver.print_statistics()

        # Extract solution
        opt_xi, opt_ui = get_solution(solver, N, nx, nu)

        delta = opt_ui[0, 0]  # Extract heading angle from the first optimal input
        vel = opt_ui[0, 1]  # Extract velocity from first optimal input
        theta = opt_xi[0, 3]

        driveMsg = generate_drive_msg(vel, delta)  # Create drive message from calculated velocity and heading
        drivePub.publish(driveMsg)  # Send message to f1tenth car

        i += 1
        rate.sleep()


if __name__ == '__main__':
    main()



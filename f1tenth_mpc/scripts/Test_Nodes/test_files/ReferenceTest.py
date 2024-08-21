#!/usr/bin/env python3

from ..Helper_scripts.referenceParser import *
from ..Helper_scripts.interpolate_path import *
import matplotlib.pyplot as plt
from ..Helper_scripts.GetStates import *
from ..Helper_scripts.Models import CarModel


def main():

    """ EXTRACT REFERENCE"""
    center_line = get_reference_path_from_file()

    # Get the car's starting point
    car_odometry = odom()
    start_pose = car_odometry.init_pos
    track_start_pos, _ = get_closest_pose(center_line, start_pose[:2])

    # reformulate the center line coordinates so that the track starts from the closest point to the car
    center_line = reformulate_path(center_line, track_start_pos)

    # Close loop
    center_line = np.vstack((center_line, center_line[0, :]))

    # Parameterize and interpolate reference path
    param_center_line = parameterize_path(center_line)
    center_line = interpolate_param_path(param_center_line, n=5)

    # Recalculate car starting point
    track_start_pos, _ = get_closest_pose(center_line, start_pose[:2])

    system = CarModel()
    sys_model = system.kinematic_bicycle_model(track_length=param_center_line.arc_lengths[-1])
    theta = param_center_line.arc_lengths[0]
    v_theta = 2.0
    vel = 1.0
    delta = 0.0
    t_samp = 0.05
    rate = rospy.Rate(1/t_samp)
    ec_array = np.array([])
    el_array = np.array([])
    x = np.append(start_pose, theta)
    N_sim = 200
    for i in range(N_sim):
        x_ref = np.array([param_center_line.x(x[3]), param_center_line.y(x[3]), param_center_line.psi(x[3])])
        ec_array = np.append(ec_array, ec(x, x_ref))
        el_array = np.append(el_array, el(x, x_ref))

        x = system.simulate(x, np.array([delta, vel, v_theta]))

        rate.sleep()

    t = np.linspace(0.0, t_samp*N_sim, N_sim)
    plt.plot(t, ec_array)
    plt.show()


def ec(pose, ref_pose):
    X_k = pose[0]
    Y_k = pose[1]

    X_ref = ref_pose[0]
    Y_ref = ref_pose[1]
    psi_ref = ref_pose[2]
    return np.sin(psi_ref) * (X_k - X_ref) - np.cos(psi_ref) * (Y_k - Y_ref)


def el(pose, ref_pose):
    X_k = pose[0]
    Y_k = pose[1]

    X_ref = ref_pose[0]
    Y_ref = ref_pose[1]
    psi_ref = ref_pose[2]
    return -np.cos(psi_ref) * (X_k - X_ref) - np.sin(psi_ref) * (Y_k - Y_ref)


if __name__ == '__main__':
    rospy.init_node('Test_Node', anonymous=True)
    main()

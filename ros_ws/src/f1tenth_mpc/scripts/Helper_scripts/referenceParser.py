import types

import numpy as np
from casadi import *
from tf.transformations import quaternion_from_euler
from scipy.interpolate import CubicSpline

import rospkg
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.spatial import distance

from tf.transformations import euler_from_quaternion
from nav_msgs.msg import Odometry
import rospy


""" PATH FOLLOWING """


def get_reference_poses(ref_path, car_pose, ref_orientation, N,  radius=5.0):
    """
    Function to extract retrieve closest point to the car's current position that lies on the reference path.

    :Param:
        ref_path : X and Y coordinates of reference path (numpy array of dimension N x 2)
        car_position : Current X and Y coordinates of the car (numpy array of dimension 2 X 1)
    """
    car_position = car_pose[:2]
    close_pts = ref_path[np.linalg.norm(ref_path - car_position.T, axis=1) < radius]
    close_ors = ref_orientation[np.linalg.norm(ref_path - car_position.T, axis=1) < radius]

    dist = np.linalg.norm(close_pts - car_position.T, axis=1)

    idx = np.argmin(dist)

    closest_pt = np.array(close_pts[idx])
    closest_or = np.array([close_ors[idx]])

    ref_pose = np.concatenate((closest_pt, closest_or))

    # return ref_pose

    idx_c = np.where(np.all(ref_path == closest_pt, axis=1))[0][0]

    if idx_c+N > len(ref_path)-1:

        m = len(ref_path) - idx_c
        ref_pts = np.concatenate((ref_path[idx_c:], ref_path[:(N-m)])).reshape(N, 2)
        ref_ors = np.concatenate((ref_orientation[idx_c:], ref_orientation[:(N-m)])).reshape(N, 1)
        ref_poses = np.concatenate((ref_pts, ref_ors), axis=1)

    else:
        ref_pts = ref_path[idx_c:idx_c + N].reshape(N, 2)
        ref_ors = ref_orientation[idx_c:idx_c + N].reshape(N, 1)
        ref_poses = np.concatenate((ref_pts, ref_ors), axis=1)

    return ref_poses


def create_ref_orientation(ref_path, end_point=True):

    x_ref = ref_path[:, 0]
    y_ref = ref_path[:, 1]
    psi_array = []

    for i in range(len(x_ref)-1):
        dx = x_ref[i + 1] - x_ref[i]
        dy = y_ref[i + 1] - y_ref[i]
        psiref = np.arctan2(dy, dx)
        psi_array = np.append(psi_array, psiref)
        q_ref = quaternion_from_euler(ai=0, aj=0, ak=psiref)

    if end_point:
        dx_e = x_ref[0] - x_ref[-1]
        dy_e = y_ref[0] - y_ref[-1]
        psiref_e = np.arctan2(dy_e, dx_e)
        psi_array = np.append(psi_array, psiref_e)

    return np.array(psi_array)


def get_closest_pose(ref_path, car_position, radius=5):

    ref_orientation = create_ref_orientation(ref_path)
    close_pts = ref_path[np.linalg.norm(ref_path - car_position.T, axis=1) < radius]
    close_ors = ref_orientation[np.linalg.norm(ref_path - car_position.T, axis=1) < radius]

    dist = np.linalg.norm(close_pts - car_position.T, axis=1)
    # dist_sq = np.sum((close_pts-car_position)**2, axis=1)

    idx = np.argmin(dist)

    closest_pt = np.array(close_pts[idx])
    closest_or = np.array([close_ors[idx]])

    ref_pose = np.concatenate((closest_pt, closest_or))

    return ref_pose, close_pts


""" TRAJECTORY TRACKING """


def create_traj_from_path(path, start_point, ts=0.1, interpolate=False, ts_interp=0.02):

    x_path = path[:, 0]
    y_path = path[:, 1]

    traj_start,_ = get_closest_pose(path, start_point[:2])

    start_idx = np.where(np.all(path == traj_start[:2],axis=1))[0][0]

    x = path[start_idx:, 0]
    y = path[start_idx:, 1]

    if start_idx > 0:
        x = np.append(x, x_path[:start_idx])
        y = np.append(y, y_path[:start_idx])

    t_stamp = np.arange(0.0, x.shape[0]*ts, ts)
    ref_orientations = create_ref_orientation(np.column_stack([x, y]))
    traj = np.column_stack([x, y, ref_orientations, t_stamp])

    if interpolate:
        t_stamp_new = np.arange(0.0, x.shape[0]*ts, ts_interp)
        x_interp = np.interp(t_stamp_new, t_stamp, x)
        y_interp = np.interp(t_stamp_new, t_stamp, y)
        ref_orientations_interp = create_ref_orientation(np.column_stack([x_interp, y_interp]))
        traj = np.column_stack([x_interp, y_interp, ref_orientations_interp, t_stamp_new])

        arc_lengths_interp = np.cumsum(np.sqrt(np.diff(x_interp) ** 2 + np.diff(y_interp) ** 2))
        vel = (arc_lengths_interp[-1]/t_stamp_new[-1])
    else:
        arc_lengths = np.cumsum(np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2))
        vel = (arc_lengths[-1]/t_stamp[-1])

    return traj, vel


def get_ref(i, N, speed, ref_traj, const_vel=True):

    if const_vel:
        if i+N+1 < len(ref_traj):
            x_ref = ref_traj[i:i+(N+1), :3]
            u_ref = np.column_stack([x_ref[:-1, 2], speed])
        else:
            x_ref = np.vstack([ref_traj[i:, :3], ref_traj[:(i+N+1) - len(ref_traj), :3]])
            u_ref = np.column_stack([x_ref[:-1, 2], speed])
    else:
        if i+N+1 < len(ref_traj):
            x_ref = ref_traj[i:i+(N+1), :3]
            u_ref = np.column_stack([x_ref[:-1, 2], speed[i:i+N]])
        else:
            x_ref = np.vstack([ref_traj[i:, :3], ref_traj[:(i+N+1) - len(ref_traj), :3]])
            vel_vector = np.concatenate((speed[i:], speed[:(i + N) - len(ref_traj)]))
            u_ref = np.column_stack([x_ref[:-1, 2], vel_vector])

    return x_ref, u_ref


def velocity_profile(param_path):

    # velocity profile
    vel_max = 5.0
    vel_min = 0.0
    kappa = calculate_curvature(param_path)
    ref_vel = np.abs(1/kappa(param_path.s))
    v_ref = []

    for vel in ref_vel:
        if vel > vel_max:
            vel = vel_max
            v_ref = np.append(v_ref, vel)
        elif vel < vel_min:
            vel = vel_min
            v_ref = np.append(v_ref, vel)
        else:
            v_ref = np.append(v_ref, vel)

    return v_ref


""" CONTOURING CONTROL """


def get_reference_path_from_file(track='Silverstone'):

    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('f1tenth_mpc')
    file_path = pkg_path + f'/scripts/Helper_scripts/Additional_maps/{track}/{track}_centerline.csv'

    df = pd.read_csv(file_path)

    if track.casefold() == 'RST_Hallway'.casefold() or track.casefold() == 'rst_track'.casefold():
        ref_path = df.iloc[:, 1:].values
    else:
        ref_path = df.iloc[:, :2].values

    return ref_path


def parameterize_ref_path(path):

    x_data = path[:, 0]
    y_data = path[:, 1]

    arc_lengths = np.cumsum(np.sqrt(np.diff(x_data) ** 2 + np.diff(y_data) ** 2))
    arc_lengths = np.append(0.0, arc_lengths)

    x_interp = CubicSpline(arc_lengths, x_data, extrapolate='periodic')
    y_interp = CubicSpline(arc_lengths, y_data, extrapolate='periodic')

    ref_path_param = types.SimpleNamespace()
    ref_path_param.x = x_interp
    ref_path_param.y = y_interp
    ref_path_param.s = arc_lengths
    ref_path_param.psi = interpolate_orientation(ref_path_param)
    ref_path_param.kappa = calculate_curvature(ref_path_param)

    return ref_path_param


def interpolate_orientation(param_path):

    arc_lengths = np.linspace(0.0, 100*param_path.s[-1], int(1000*len(param_path.s)))
    x_data = param_path.x(arc_lengths)
    y_data = param_path.y(arc_lengths)

    dx = np.gradient(x_data)
    dy = np.gradient(y_data)
    psi = np.unwrap(np.arctan2(dy, dx))

    psi_interp = CubicSpline(arc_lengths, psi, extrapolate='periodic')

    return psi_interp


def interpolate_param_path(param_path, n=2):

    arc_len_interp = np.linspace(param_path.s[0], param_path.s[-1], int(n*len(param_path.s)))

    x_interp = param_path.x(arc_len_interp)
    y_interp = param_path.y(arc_len_interp)

    ref_path_interp = np.column_stack([x_interp, y_interp])

    return ref_path_interp


def calculate_curvature(param_ref_path):

    # x = ref_path[:, 0]
    # y = ref_path[:, 1]
    #
    # dx = np.gradient(x)
    # dy = np.gradient(y)
    #
    # theta = np.unwrap(np.arctan2(dy, dx))
    # s = np.cumsum(np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2))
    # s = np.append(0.0, s)

    theta = param_ref_path.psi(param_ref_path.s)
    s = param_ref_path.s
    curvature = abs(np.gradient(theta, s))
    curv_interp = CubicSpline(s, curvature, extrapolate='periodic')

    return curv_interp


def project_point(param_ref_path, car_pos):

    curve_points = np.column_stack([param_ref_path.x(param_ref_path.s),
                                    param_ref_path.y(param_ref_path.s)])

    distances = cdist([car_pos[:2]], curve_points)
    closest_index = np.argmin(distances)
    s_init = param_ref_path.s[closest_index]

    s_vals = param_ref_path.s

    x_vals = param_ref_path.x(s_vals)
    y_vals = param_ref_path.y(s_vals)

    xr = interpolant('xr', 'bspline', [s_vals], x_vals)
    yr = interpolant('yr', 'bspline', [s_vals], y_vals)

    s = MX.sym('s')

    point = car_pos[:2]

    # Objective function
    cost = (xr(s) - point[0]) ** 2 + (yr(s) - point[1]) ** 2

    nlp = {'x': s, 'f': cost}
    opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.tol': 1e-2}
    solver = nlpsol('solver', 'ipopt', nlp, opts)

    res = solver(x0=s_init, lbx=0, ubx=1000)
    s_opt = float(res['x'])

    return s_opt


def reformulate_path(ref_path, track_start):

    x_path = ref_path[:, 0]
    y_path = ref_path[:, 1]

    start_idx = np.where(np.all(ref_path == track_start[:2], axis=1))[0][0]

    x = ref_path[start_idx:, 0]
    y = ref_path[start_idx:, 1]

    x = np.append(x, x_path[:start_idx])
    y = np.append(y, y_path[:start_idx])

    reform_path = np.column_stack([x, y])

    return reform_path


def reform_param_path(ref_line, proj_pt):

    distances = cdist([proj_pt], ref_line)
    # new_s = np.roll(param_path.s, -opt_idx, axis=0)

    # x_val = param_path.x(new_s[0])
    # y_val = param_path.y(new_s[0])
    # import matplotlib.pyplot as plt
    # plt.plot(param_path.x(param_path.s), param_path.y(param_path.s), 'r.')
    # plt.plot(param_path.x(param_path.s[0]), param_path.y(param_path.s[0]), 'g*')
    # plt.plot(x_val, y_val, 'bo')
    # plt.show()

    # Reformulate the parameterized path based on the optimal s value

    # track_length = param_path.s[-1]
    # new_s = param_path.s - s_opt
    # # Find the indices where the negative numbers are
    # neg_idx = np.where(new_s[new_s < 0])[0]
    # part1 = new_s[(neg_idx[-1]+1):]
    # part2 = np.linspace(part1[-1], track_length, len(new_s[:(neg_idx[-1]+1)]))
    # new_s = np.concatenate(([0.0], np.unique(np.concatenate((part1, part2)))))
    idx1, idx2 = np.sort(np.argpartition(distances, 2)[0][:2])
    part1 = ref_line[:(idx1+1)]
    part2 = ref_line[idx2:-1]
    new_ref_line = np.vstack((proj_pt, part2, part1))

    new_param_path = parameterize_ref_path(new_ref_line)

    return new_param_path


def get_param_ref(ref_path_parameterized, s, N, t_samp, v_s_max=4, initial=False, opt_x=None):

    if initial:
        s_array = np.array([s])
        v_s = np.linspace(0, v_s_max, N+1)
        for i in range(1, N+1):
            s += v_s[i]*t_samp
            s_array = np.append(s_array, s)
    else:
        s_array = opt_x[1:, 3]
        s_f = s_array[-1] + opt_x[-1, 6]*t_samp
        s_array = np.append(s_array, s_f)

    x_ref = ref_path_parameterized.x(s_array)
    y_ref = ref_path_parameterized.y(s_array)
    psi_ref = ref_path_parameterized.psi(s_array)

    return np.column_stack([x_ref, y_ref, psi_ref, s_array])


def get_reference(current_pos, param_path, iteration, N, t_samp, v_s_max=1.0, opt_x=None):

    if iteration == 0 or opt_x is None:
        s = project_point(param_path, current_pos)
        s_array = np.array([s])
        v_s = np.linspace(0, v_s_max, N+1)
        for i in range(1, N + 1):
            s += v_s[i] * t_samp
            s_array = np.append(s_array, s)
    else:
        s_array = opt_x[1:, 3]
        s_f = s_array[-1] + opt_x[-1, 6] * t_samp
        s_array = np.append(s_array, s_f)

    x_ref = param_path.x(s_array)
    y_ref = param_path.y(s_array)
    psi_ref = param_path.psi(s_array)

    return np.column_stack([x_ref, y_ref, psi_ref, s_array])


def parameterize_path(path):

    x_data = path[:, 0]
    y_data = path[:, 1]

    dx = np.gradient(x_data)
    dy = np.gradient(y_data)
    psi_data = np.arctan2(dy, dx)
    psi_data = np.unwrap(psi_data)
    # psi_data = (psi_data % (2*np.pi))
    # psi_data = np.arctan(m)

    arc_lengths = np.cumsum(np.sqrt(np.diff(x_data)**2 + np.diff(y_data)**2))
    arc_lengths = np.append(0.0, arc_lengths)

    # Ensure that arc lengths are strictly increasing/unique
    # arc_lengths, unique_indices, counts = np.unique(arc_lengths, return_index=True, return_counts=True)

    # if len(arc_lengths[counts > 1]) > 0:
    #     # Sort the x and y coordinates based on the unique indices
    #     x_data = x_data[unique_indices]
    #     y_data = y_data[unique_indices]
    #     psi_data = psi_data[unique_indices]

    x_interp = CubicSpline(arc_lengths, x_data, extrapolate='periodic')
    y_interp = CubicSpline(arc_lengths, y_data, extrapolate='periodic')
    psi_interp = CubicSpline(arc_lengths, psi_data, extrapolate='periodic')

    ref_path_param = types.SimpleNamespace()
    ref_path_param.x = x_interp
    ref_path_param.y = y_interp
    ref_path_param.psi = psi_interp
    ref_path_param.arc_lengths = arc_lengths

    # import matplotlib.pyplot as plt
    # # plt.plot(x_interp, y_interp, '.')
    # # plt.plot(x_data, y_data, '.')
    # sample = np.linspace(arc_lengths[0], 2*arc_lengths[-1], 4000)
    # x_pos = ref_path_param.x(sample)
    # y_pos = ref_path_param.y(sample)
    # orien = psi_interp(sample)
    # pos = np.column_stack([x_pos.T, y_pos.T])
    # #
    # plt.plot(x_pos, y_pos, '.')
    # plt.scatter(pos[:, 0], pos[:, 1], color='blue')

    # for i in range(len(pos)):
    #     ang = orien[i]
    #     d_x = np.cos(ang)
    #     d_y = np.sin(ang)
    #     plt.quiver(pos[i, 0], pos[i, 1], d_x, d_y, scale=1, scale_units='xy', color='red', label='Orientations')

    # plt.show()

    return ref_path_param


# def get_closest_point(pose, ref_path):
#
#     pt = MX.sym('point', 2)
#     crv = MX.sym('curve', len(ref_path), 2)
#
#     distances = sqrt((pt[0] - crv[:, 0]) ** 2 + (pt[1] - crv[:, 1]) ** 2)
#
#     cost_fcn = Function('Cost_Function', [pt, crv], [distances])
#     target_point = pose[:2].tolist()
#     points = ref_path.tolist()
#
#     target_x, target_y = target_point
#     # target_pt_var = MX.sym([target_x, target_y])
#     points_var = MX.sym('points', len(points),2)
#
#     cost = sqrt((points_var[:,0]-target_x)**2 + (points_var[:,1]-target_y)**2)
#     cost_fcn = sum1(cost)
#
#     nlp = {'x': points_var, 'f': cost_fcn}
#     solver_opts = {'ipopt.print_level': 0,
#                    'print_time': 0
#                    }
#
#     solver = nlpsol('solver', 'ipopt', nlp, solver_opts)
#
#     result = solver(x0=points)
#
#     idx = result['x'].argmin()
#     closest_pt = points[idx]
# #     return closest_pt


# if __name__ == '__main__':
#
#     rospack = rospkg.RosPack()
#     track = 'Silverstone'
#     pkg_path = rospack.get_path('f1tenth_simulator')
#     file_path = pkg_path + f'/scripts/Additional_maps/{track}/{track}_centerline.csv'
#
#     df = pd.read_csv(file_path)
#     center_line = df.iloc[:, :2].values
#     center_line_interp = interpolate_path(center_line)
#     psi = create_ref_orientation(ref_path=center_line_interp)
#     print(psi.T.shape)

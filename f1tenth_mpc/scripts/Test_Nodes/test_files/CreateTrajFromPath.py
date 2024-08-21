import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import types

# ROS imports
import rospkg
from referenceParser import create_ref_orientation


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


def main():
    t_samp = 0.05
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

    # Single Track Model
    v0 = 0.5
    x0_st = [0, 0, 0.3, v0, 0, 0, 0]
    a_long = 0.0  # m/s2
    v_delta = 0.0  # rad/s
    u_st = [v_delta, a_long]

    x = x0_st
    x_traj = np.array([x])

    for i in range(1):
        st_sol = solve_ivp(st_model, t_span, x, args=(u_st, p))
        x = st_sol.y[:, -1]
        x_traj = np.vstack([x_traj, np.array([st_sol.y[:, -1]])])

    import matplotlib.pyplot as plt

    # plt.plot(x_traj[:, 0])
    # plt.plot(x_traj[:, 1])
    # plt.show()


if __name__ == '__main__':
    main()
















#
# def create_traj_from_path(path, ts, interpolate=False, ts_interp=0.02):
#
#     x = path[:, 0]
#     y = path[:, 1]
#     t_stamp = np.arange(0.0, x.shape[0]*ts, ts)
#     ref_orientations = create_ref_orientation(path)
#     traj = np.column_stack([x, y, ref_orientations, t_stamp])
#
#     if interpolate:
#         t_stamp_new = np.arange(0.0, x.shape[0]*ts, ts_interp)
#         x_interp = np.interp(t_stamp_new, t_stamp, x)
#         y_interp = np.interp(t_stamp_new, t_stamp, y)
#         ref_orientations_interp = create_ref_orientation(np.column_stack([x_interp, y_interp]))
#         traj = np.column_stack([x_interp, y_interp,ref_orientations_interp, t_stamp_new])
#
#     #     arc_lengths_interp = np.cumsum(np.sqrt(np.diff(x_interp) ** 2 + np.diff(y_interp) ** 2))
#     #     print(arc_lengths_interp[-1]/t_stamp_new[-1])
#     # else:
#     #     arc_lengths = np.cumsum(np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2))
#     #     print(arc_lengths[-1]/t_stamp[-1])
#
#     return traj
#
#
# if __name__ == '__main__':
#
#     rospack = rospkg.RosPack()
#     track = 'Silverstone'
#     pkg_path = rospack.get_path('f1tenth_simulator')
#     file_path = pkg_path + f'/scripts/Additional_maps/{track}/{track}_centerline.csv'
#
#     df = pd.read_csv(file_path)
#     center_line = df.iloc[:, :2].values
#     traj = create_traj_from_path(center_line, ts=0.1)
#     traj_interp = create_traj_from_path(center_line, ts=0.1, ts_interp=0.02, interpolate=True)


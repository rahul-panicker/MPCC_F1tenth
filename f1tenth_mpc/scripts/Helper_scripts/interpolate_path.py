import numpy as np




    # if not np.array_equal(ref_path_interp[0, :], ref_path_interp[-1, :]):
    #     start_pt = ref_path_interp[0, :]
    #     end_pt = ref_path_interp[-1, :]
    #
    #     arc_len_final = np.sqrt((start_pt[0]-end_pt[0])**2 + (start_pt[1]-end_pt[1])**2)
    #     arc_len_total = np.append(arc_len_interp, arc_len_interp[-1]+arc_len_final)
    #     x_interp = param_path.x(arc_len_total)
    #     y_interp = param_path.y(arc_len_total)
    #
    #     ref_path_interp = np.column_stack([x_interp, y_interp])


# if __name__ == '__main__':
#     rospack = rospkg.RosPack()
#     track = 'Silverstone'
#     pkg_path = rospack.get_path('f1tenth_simulator')
#     file_path = pkg_path + f'/scripts/Additional_maps/{track}/{track}_centerline_with_poses.csv'
#
#     df = pd.read_csv(file_path)
#     center_line = df.iloc[:, 1:3].values
#     center_line_interp = interpolate_path(center_line)
#     print(center_line[:,0].shape)
#     print(center_line_interp[:,0].shape)

# def interpolate_path(path, n):
#     """
#     Function to use linear interpolation to create more-fine grained path data.
#
#     :Parameters:
#         path - Nx2 numpy array containing the x and y coordinates of the reference path
#         n    - number of points between 2 consecutive coordinates required
#     """
#
#     x = path[:, 0]
#     y = path[:, 1]
#
#     x_interp = []
#     y_interp = []
#
#     for i in range(len(x)-1):
#         x_new = np.linspace(x[i], x[i+1], n)
#         y_new = np.linspace(y[i], y[i+1], n)
#
#         x_interp = np.append(x_interp, x_new[:-1])
#         y_interp = np.append(y_interp, y_new[:-1])
#
#     x_new_e = np.linspace(x[-1], x[0], n)
#     y_new_e = np.linspace(y[-1], y[0], n)
#
#     x_interp = np.append(x_interp, x_new_e[:-1])
#     y_interp = np.append(y_interp, y_new_e[:-1])
#
#     path_interp = np.column_stack([x_interp, y_interp])
#     return path_interp

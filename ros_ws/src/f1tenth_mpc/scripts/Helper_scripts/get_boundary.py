import numpy as np


def get_bounds(ref_pts, dist):

    dx_c = np.gradient(ref_pts[:, 0])
    dy_c = np.gradient(ref_pts[:, 1])

    d = dist
    abs_val = np.sqrt(dx_c**2 + dy_c**2)
    # Calculate unit normal vector
    unit_normal = np.array([-dy_c/abs_val, dx_c/abs_val]).T

    x_left = ref_pts[:, 0] + d*unit_normal[:, 0]
    y_left = ref_pts[:, 1] + d*unit_normal[:, 1]

    x_right = ref_pts[:, 0] - d*unit_normal[:, 0]
    y_right = ref_pts[:, 1] - d*unit_normal[:, 1]

    left_bounds = np.column_stack([x_left, y_left])
    right_bounds = np.column_stack([x_right, y_right])

    return left_bounds, right_bounds

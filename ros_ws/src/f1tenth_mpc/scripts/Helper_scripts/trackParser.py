#!/usr/bin/env python3

import pandas as pd
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
from referenceParser import *

# ROS Imports
import rospy
import rospkg
import tf2_ros
from generate_vizMarkerMsg import generate_line_msg, generate_pts, generate_PointMarkerMsg
from visualization_msgs.msg import Marker


def extract_centerLine(img_file_path):

    track_map = cv2.imread(img_file_path, cv2.IMREAD_GRAYSCALE)

    _, center_line_bin_map = cv2.threshold(track_map, 252, 255, cv2.THRESH_BINARY)  # 100
    # view_image(center_line_bin_map)

    center_dist_transform = cv2.distanceTransform(center_line_bin_map, cv2.DIST_L2, 3)
    # view_image(center_dist_transform)

    norm_center_dist_tf = cv2.normalize(center_dist_transform, None, 0, 255, cv2.NORM_MINMAX)
    # view_image(norm_center_dist_tf)

    _, center_line_thresholded = cv2.threshold(norm_center_dist_tf, 155, 255, cv2.THRESH_BINARY)
    # view_image(center_line_thresholded)

    center_line_thinned_img = cv2.ximgproc.thinning(np.uint8(center_line_thresholded))
    # cv2.imwrite('Center_Line_thinned_img.png', center_line_thinned_img)
    # view_image(center_line_thinned_img)

    contours, _ = cv2.findContours(center_line_thinned_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # view_image(cv2.drawContours(track_map, contours, -1, (0, 255, 0), thickness=2))
    contour_points = contours[0].reshape(-1, 2)

    origin = [-12.812499, -10.512499, 0.000000]  # [-12.812499, -13.112499, 0.00]
    res = 0.025  # 0.026
    x_center = res * contour_points[:, 0] + origin[0]*np.ones(len(contour_points))
    y_center = res * contour_points[:, 1] + origin[1]*np.ones(len(contour_points))

    return np.column_stack((x_center, y_center))

    # Create a copy of the original image
    # contour_image = track_map.copy()
    #
    # # Draw contours on the contour image
    # cv2.drawContours(contour_image, contours, -1, (0, 255, 0), thickness=2)
    #
    # # Create a named window
    #
    # # Resize the window to your desired dimensions
    # flattened_points = [point for sublist in contour_points for point in sublist]
    #
    # point_array = np.array(flattened_points)
    # x_coordinates = point_array[::2]
    # y_coordinates = point_array[1::2]
    # print(contour_points)
    # center_line = np.column_stack(np.where(center_line_thinned_img > 0))
    #
    # print("Center line extracted.")
    # return center_line


def view_image(img):
    cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Image', 800, 600)
    cv2.imshow('Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def smoothen(line):

    x, y = line.T
    t = np.linspace(0, 1, len(x))
    t2 = np.linspace(0, 1, 100)

    x2 = np.interp(t2, t, x)
    y2 = np.interp(t2, t, y)
    sigma = 3

    x_smooth = gaussian_filter1d(x2, sigma)
    y_smooth = gaussian_filter1d(y2, sigma)

    return np.column_stack((x_smooth, y_smooth))


if __name__ == '__main__':

    rospy.init_node('track_parser', anonymous=True)
    visPub = rospy.Publisher('/visualization_marker', Marker, queue_size=10)  # Publisher for Marker messages

    center_line = extract_centerLine('Additional_maps/rst_track/rst_track.pgm')
    center_line = np.vstack((center_line, center_line[0, :]))
    n = len(center_line)

    curve1 = center_line[15:int(n/4)-5, :2]
    # plt.plot(curve1[:, 0], curve1[:, 1], 'r.')

    straight1 = center_line[int(n/4)-5:int(n/2)+10, :2]
    # plt.plot(straight1[:, 0], straight1[:, 1], 'k.')

    curve2 = center_line[int(n/2)+10:int(3*n/4), :2]
    # plt.plot(curve2[:, 0], curve2[:, 1], 'r.')

    straight2 = np.vstack((center_line[int(3*n/4):-1, :2], center_line[:15, :2]))
    # plt.plot(straight2[:, 0], straight2[:, 1], 'k.')

    curve1_smooth = smoothen(curve1)
    curve2_smooth = smoothen(curve2)
    straight1_smooth = smoothen(straight1)
    straight2_smooth = smoothen(straight2)

    # plt.figure()
    # plt.plot(curve1_smooth[:, 0], curve1_smooth[:, 1], 'r.')
    # plt.plot(curve2_smooth[:, 0], curve2_smooth[:, 1], 'r.')
    # plt.plot(straight1_smooth[:, 0], straight1_smooth[:, 1], 'k.')
    # plt.plot(straight2_smooth[:, 0], straight2_smooth[:, 1], 'k.')

    center_line_smooth = np.vstack((curve1_smooth, straight1, curve2_smooth, straight2))

    param_center = parameterize_ref_path(center_line_smooth)
    x_center, y_center = interpolate_param_path(param_center, n=0.1).T

    center_line = np.column_stack((x_center, y_center))

    theta = 0.0
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])

    center_line_rotated = np.dot(center_line, R.T)

    # center_line_rotated[:, 1] = -center_line_rotated[:, 1]
    # lineMsg = generate_line_msg(center_line_rotated, id=1, colors=[1.0, 0.0, 0.0])
    # visPub.publish(lineMsg)

    df = pd.DataFrame(center_line_rotated)
    df.to_csv('rst_track_centerline.csv')

# def parseTrack(img_file_path, origin, resolution):
#     """
#     Function to extract center line and track bounds using the Euclidean distance transform.
#     :Inputs:
#         img_file_path : Path to image of track
#     :results:
#         track_bounds : X and Y coordinates of track bounds as a numpy array
#         center_line: X and Y coordinates of center line as a numpy array
#     """
#     track_map = cv2.imread(img_file_path, cv2.IMREAD_GRAYSCALE)
#
#     _, center_line_bin_map = cv2.threshold(track_map, 250,255, cv2.THRESH_BINARY)
#     # cv2.imwrite('Center_Line_binary_map.png', center_line_bin_map)
#
#     center_dist_transform = cv2.distanceTransform(center_line_bin_map, cv2.DIST_L2, 3)
#     # cv2.imwrite('Center_Line_dist_transform.png', center_dist_transform)
#
#     norm_center_dist_tf = cv2.normalize(center_dist_transform, None, 0, 255, cv2.NORM_MINMAX)
#     # cv2.imwrite('Center_Line_norm_dist_tf.png', norm_center_dist_tf)
#
#     _, center_line_thresholded = cv2.threshold(norm_center_dist_tf, 75, 255, cv2.THRESH_BINARY)
#     # cv2.imwrite('Center_Line_thresholded.png', center_line_thresholded)
#
#     center_line_thinned_img = cv2.ximgproc.thinning(np.uint8(center_line_thresholded))
#     # cv2.imwrite('Center_Line_thinned_img.png', center_line_thinned_img)
#
#     center_line = np.column_stack(np.where(center_line_thinned_img > 0))
#
#     kernel1 = np.ones((21, 21), np.uint8)
#     kernel2 = np.ones((13, 13), np.uint8)
#
#     complete_track = cv2.dilate(center_line_thinned_img, kernel1, iterations=1)
#     # cv2.imwrite('complete_track_binary_img.png', complete_track)
#
#     dilated_center_line = cv2.dilate(center_line_thinned_img, kernel2, iterations=1)
#     # cv2.imwrite('dilated_center_line_img.png', dilated_center_line)
#
#     track_bounds_img = cv2.subtract(complete_track, dilated_center_line )
#     # cv2.imwrite('track_bounds_img.png', track_bounds_img)
#
#     track_bounds_thinned_img = cv2.ximgproc.thinning(np.uint8(track_bounds_img))
#     # cv2.imwrite('track_bounds_thinned_img.png', track_bounds_thinned_img)
#
#     track_bounds = np.column_stack(np.where(track_bounds_thinned_img > 0))
#
#     x_pixel = center_line[:, 0]
#     y_pixel = center_line[:, 1]
#
#     origin_x = origin[0]
#     origin_y = origin[1]
#
#     x_map = origin_x + resolution * x_pixel
#     y_map = origin_y + resolution * y_pixel
#
#     center_line_map = np.column_stack([x_map, y_map])
#     # center_line_simplified = douglas_peucker(center_line, epsilon=1e2)
#     # center_line_filtered = filter_line(center_line)
#
#     plt.plot(center_line_map[:, 0], center_line_map[:, 1])
#     # plt.plot(center_line_filtered[:, 0], center_line_filtered[:, 1], '.')
#     # plt.plot(track_bounds[:, 0], track_bounds[:, 1], '.')
#     plt.show()
#
#     # return track_bounds, center_line
#
#
# def douglas_peucker(points, epsilon):
#     # Find the point with the maximum distance
#     dmax = 0
#     index = 0
#     end = len(points) - 1
#     for i in range(1, end):
#         d = perpendicular_distance(points[i], points[0], points[end])
#         if d > dmax:
#             index = i
#             dmax = d
#
#     # If the maximum distance is greater than epsilon, recursively simplify
#     if dmax > epsilon:
#         # Recursive call on the left and right segments
#         left_simplified = douglas_peucker(points[:index + 1], epsilon)
#         right_simplified = douglas_peucker(points[index:], epsilon)
#
#         # Concatenate the simplified segments
#         return np.vstack((left_simplified[:-1], right_simplified))
#     else:
#         # Return the original endpoints
#         return np.vstack((points[0], points[end]))
#
#
# def perpendicular_distance(point, start, end):
#     # Calculate the perpendicular distance from a point to a line segment
#     numerator = np.abs(
#         (end[1] - start[1]) * point[0] - (end[0] - start[0]) * point[1] + end[0] * start[1] - end[1] * start[0])
#     denominator = np.sqrt((end[1] - start[1]) ** 2 + (end[0] - start[0]) ** 2)
#     return numerator / denominator
#
#
# def filter_line(line):
#
#     sorted_indices = np.argsort(line[:, 0])
#     sorted_center_line = line[sorted_indices]
#
#     cs = CubicSpline(sorted_center_line[:, 0], sorted_center_line[:, 1])
#     x_interp = np.linspace(sorted_center_line[0, 0], sorted_center_line[-1, 0], num=1000)
#     y_interp = cs(x_interp)
#
#     return np.column_stack([x_interp, y_interp])
#
#
# def track_creator():
#     img = cv2.imread('Track.png', cv2.IMREAD_GRAYSCALE)
#     _, bin_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
#     cv2.imwrite('Track_bin.png', bin_img)
#
#     resolution = 0.004  # meters per pixel
#     origin = [-3.52, -3.577, 0.0]  # origin of the grid
#     occupancy_grid = np.where(bin_img == 0, 0, 100)
#     width, height = bin_img.shape[::-1]  # width and height of the image
#
#     with open('Track.yaml', 'w') as f:
#         f.write('image: Track_occupancy.png\n')
#         f.write(f'resolution: {resolution}\n')
#         f.write(f'origin: {origin}\n')
#         f.write('occupied_thresh: 0.65\n')
#         f.write('free_thresh: 0.196\n')
#         f.write('negate: 0\n')
#         f.write(f'width: {width}\n')
#         f.write(f'height: {height}\n')

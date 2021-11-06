import cv2
import numpy as np

from . import gui


def estimate_homography(original_image, original_points, save_warped_frame=True):
    src = np.reshape(original_points[:4], (4, 1, 2))
    dst = np.reshape(original_points[4:8], (4, 1, 2)) 
    
    homography = cv2.findHomography(src, dst, cv2.RANSAC, 10.0)[0]

    if save_warped_frame:
        warped_image = original_image.copy()
        warped_image = cv2.warpPerspective(warped_image, homography, (warped_image.shape[1], warped_image.shape[0]))
        cv2.imwrite('Homography Estimate.jpg', cv2.hconcat([original_image, warped_image]))

    return homography


def get_homography(input_video_path, scaling_factor):

    # Capture the first frame of the video
    video_capture = cv2.VideoCapture(input_video_path)
    _, frame = video_capture.read()
    frame = cv2.resize(frame, dsize=(0,0), fx=scaling_factor, fy=scaling_factor)
    video_capture.release()

    # Launch the GUI and project the frame onto bird's eye view plane
    points = gui.interactive_gui(frame)
    points = list(map(list, points))
    #points = [[103,11], [262,11], [636,254], [10,261],[0,0], [638,0], [638,287], [0,287],[134,220], [422,418],[129,240], [177,240]]        
    print(points)
    homography = estimate_homography(frame, points)

    # Get distance threshold from GUI
    upper_distance_threshold, tdp1, tdp2 = get_distance_parameters(points[8:10], homography)
    lower_distance_threshold, tdp1, tdp2 = get_distance_parameters(points[10:12], homography)
    return homography, upper_distance_threshold, lower_distance_threshold


def transform_coords(orig_coords, homography):
    '''
    Project the original coordinates from the image into the birds eye view plane
    '''
    orig_coords = orig_coords.reshape(orig_coords.shape[0], 1, orig_coords.shape[1])
    new_coords = cv2.perspectiveTransform(orig_coords, homography)[0][0]
    return new_coords


def get_distance_parameters(points, H):

    distance_points = np.array(points, dtype=np.float32)

    point1 = distance_points[0][np.newaxis, :]
    point2 = distance_points[1][np.newaxis, :]

    transformed_point1 = transform_coords(point1, H)
    transformed_point2 = transform_coords(point2, H)

    distance = np.linalg.norm(transformed_point1 - transformed_point2)

    return distance, transformed_point1, transformed_point2

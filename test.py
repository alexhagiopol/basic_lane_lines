# 3rd party dependencies
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import shutil
from moviepy.editor import VideoFileClip
from IPython.display import HTML

# set tuning parameters for algortithms
# gaussian blurring param:
kernel_size = 5  # Gaussian blur kernel must be odd number
# canny edge detection params:
low_thresh = 50  # low gradient threshold
hi_thresh = 150  # high gradient threshold
# hough transform params:
rho = 1  # distance resolution of the accumulator in pixels
theta = np.pi / 180  # angle resolution of the accumulator in radians
threshold = 2  # votes needed to return line (>threshold).
min_line_len = 10
max_line_gap = 30

def detect(raw_image):
    white_mask = cv2.inRange(raw_image, np.array([200, 200, 200]), np.array([255, 255, 255]))
    yellow_mask = cv2.inRange(raw_image, np.array([120, 120, 0]), np.array([255, 255, 100]))
    combo_mask = cv2.bitwise_or(white_mask, yellow_mask)
    color_image = cv2.bitwise_and(raw_image, raw_image, mask=combo_mask)
    gray_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2GRAY)

    # gaussian blur + canny edge detection
    gaussian_image = cv2.GaussianBlur(gray_image, (kernel_size, kernel_size), 0)
    edges_image = cv2.Canny(gaussian_image, low_thresh, hi_thresh)

    # crop into left and right images for finding left and right lane lines
    left_mask = np.zeros_like(edges_image)
    right_mask = np.zeros_like(edges_image)
    imshape = raw_image.shape
    vertices_left = np.array([[(75, imshape[0] - 25), (imshape[1] / 2 - 50, imshape[0] / 2 + 75),
                          (imshape[1] / 2, imshape[0] / 2 + 75), (imshape[1] / 2, imshape[0] - 25)]], dtype=np.int32)

    vertices_right = np.array([[(imshape[1] / 2, imshape[0] - 25), (imshape[1] / 2, imshape[0] / 2 + 75),
                               (imshape[1] / 2 + 50, imshape[0] / 2 + 75), (imshape[1] - 75, imshape[0] - 25)]],
                             dtype=np.int32)
    cv2.fillPoly(left_mask, vertices_left, 255)
    cv2.fillPoly(right_mask, vertices_right, 255)
    masked_edges_image_left = cv2.bitwise_and(edges_image, left_mask)
    masked_edges_image_right = cv2.bitwise_and(edges_image, right_mask)

    # compute lines via hough transform
    left_lines = cv2.HoughLinesP(masked_edges_image_left, rho, theta, threshold, np.array([]), min_line_len, max_line_gap)
    right_lines = cv2.HoughLinesP(masked_edges_image_right, rho, theta, threshold, np.array([]), min_line_len, max_line_gap)

    processed_image = np.copy(raw_image) # create img copy

    if left_lines is not None:
        # compute "average" line by performing linear regression on hough line endpoints
        leftYs = np.array([y1 for [[x1,y1,x2,y2]] in left_lines] + [y2 for [[x1,y1,x2,y2]] in left_lines])
        leftXs = np.array([x1 for [[x1,y1,x2,y2]] in left_lines] + [x2 for [[x1,y1,x2,y2]] in left_lines])
        leftA = np.vstack([leftXs, np.ones(len(leftXs))]).T
        left_M, left_B = np.linalg.lstsq(leftA, leftYs)[0]
        left_Y1 = imshape[0]
        left_X1 = (left_Y1 - left_B) / left_M
        left_Y2 = imshape[0] / 2 + 100
        left_X2 = (left_Y2 - left_B) / left_M
        # draw lane lines and useful data on images
        cv2.putText(processed_image, '+ slope = {:.2f}'.format(left_M), (imshape[1] - 300, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(processed_image, '+ int = {:.2f}'.format(left_B), (imshape[1] - 300, 130), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 0, 0), 2)
        cv2.line(processed_image, (int(left_X1), int(left_Y1)), (int(left_X2), int(left_Y2)), (255, 0, 0), 5)  # draw lines

    if right_lines is not None:
        # compute "average" line by performing linear regression on hough line endpoints
        rightYs = np.array([y1 for [[x1,y1,x2,y2]] in right_lines] + [y2 for [[x1,y1,x2,y2]] in right_lines])
        rightXs = np.array([x1 for [[x1,y1,x2,y2]] in right_lines] + [x2 for [[x1,y1,x2,y2]] in right_lines])
        rightA = np.vstack([rightXs, np.ones(len(rightXs))]).T
        right_M, right_B = np.linalg.lstsq(rightA, rightYs)[0]
        right_Y1 = imshape[0]
        right_X1 = (right_Y1 - right_B) / right_M
        right_Y2 = imshape[0] / 2 + 100
        right_X2 = (right_Y2 - right_B) / right_M
        # draw lane lines and useful data on images
        cv2.putText(processed_image, '+ slope = {:.2f}'.format(right_M), (imshape[1] - 300, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(processed_image, '+ int = {:.2f}'.format(right_B), (imshape[1] - 300, 130), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 0, 0), 2)
        cv2.line(processed_image, (int(right_X1), int(right_Y1)), (int(right_X2), int(right_Y2)), (255, 0, 0), 5)  # draw lines

    return processed_image

in_video_dir_name = "test_videos"
out_video_dir_name = "result_videos"
if os.path.exists(out_video_dir_name):
    shutil.rmtree(out_video_dir_name)
os.mkdir(out_video_dir_name)
input_video_filenames = os.listdir(in_video_dir_name)

for video_filename in input_video_filenames:
    clip = VideoFileClip(os.path.join(in_video_dir_name, video_filename))
    processed_clip = clip.fl_image(detect)
    processed_clip_name = os.path.join(out_video_dir_name, "processed_" + video_filename)
    processed_clip.write_videofile(processed_clip_name, audio=False)
    print("processed video: ", processed_clip_name)

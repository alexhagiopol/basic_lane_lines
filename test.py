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
    #cv2.imshow('image', color_image)
    #cv2.waitKey(0)
    gray_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2GRAY)

    # gaussian blur + canny edge detection
    gaussian_image = cv2.GaussianBlur(gray_image, (kernel_size, kernel_size), 0)
    edges_image = cv2.Canny(gaussian_image, low_thresh, hi_thresh)

    # crop
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

    #cv2.imshow('image', masked_edges_image_left)
    #cv2.waitKey(0)
    #cv2.imshow('image', masked_edges_image_right)
    #cv2.waitKey(0)

    # lines via hough transform
    left_lines = cv2.HoughLinesP(masked_edges_image_left, rho, theta, threshold, np.array([]), min_line_len, max_line_gap)
    right_lines = cv2.HoughLinesP(masked_edges_image_right, rho, theta, threshold, np.array([]), min_line_len, max_line_gap)
    #print("left=", left_lines)
    #print("right=", right_lines)

    lines = []
    if right_lines is not None:
        for line in left_lines:
            lines += [line]
    if right_lines is not None:
        for line in right_lines:
            lines += [line]
    processed_image = np.copy(raw_image) # create img copy

    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(processed_image, (x1, y1), (x2, y2), (0, 0, 0), 5)


    # extract + lines from hough data structure.
    pos_lines = []
    neg_lines = []
    if right_lines is not None:
        # extract + lines from hough data structure. Associate slope so we can sort by it.
        pos_lines = [[x1, y1, x2, y2, (float(y2) - float(y1)) / (float(x2) - float(x1))] for [[x1, y1, x2, y2]] in right_lines if
                     x2 - x1 != 0 and (float(y2) - float(y1)) / (float(x2) - float(x1)) > 0]  # lines with + slope.
    if left_lines is not None:
        # extract - lines from hough data structure. Associate slope so we can sort by it.
        neg_lines = [[x1, y1, x2, y2, (float(y2) - float(y1)) / (float(x2) - float(x1))] for [[x1, y1, x2, y2]] in left_lines if
                     x2 - x1 != 0 and (float(y2) - float(y1)) / (float(x2) - float(x1)) < 0]  # lines with - slope.

    if len(pos_lines) > 1:
        pos_lines_np = np.array(pos_lines)
        [x1, y1, x2, y2, pos_M] = np.median(pos_lines_np, axis=0)

        cv2.line(processed_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 20)
        pos_B = y1 - pos_M * x1  # compute y intercept
        # compute lane line
        pos_Y1 = imshape[0]
        pos_X1 = (pos_Y1 - pos_B) / pos_M
        pos_Y2 = imshape[0] / 2 + 50
        pos_X2 = (pos_Y2 - pos_B) / pos_M
        cv2.putText(processed_image, '+ slope = {:.2f}'.format(pos_M), (imshape[1] - 300, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(processed_image, '+ int = {:.2f}'.format(pos_B), (imshape[1] - 300, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.line(processed_image, (int(pos_X1), int(pos_Y1)), (int(pos_X2), int(pos_Y2)), (255, 0, 0), 5)  # draw lines

    if len(neg_lines) > 1:
        neg_lines_np = np.array(neg_lines)
        [x1, y1, x2, y2, neg_M] = np.median(neg_lines_np, axis=0)

        cv2.line(processed_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 20)
        neg_B = y1 - neg_M * x1  # compute y intercept
        # compute lane line
        neg_Y1 = imshape[0]
        neg_X1 = (neg_Y1 - neg_B) / neg_M
        neg_Y2 = imshape[0] / 2 + 50
        neg_X2 = (neg_Y2 - neg_B) / neg_M
        cv2.putText(processed_image, '- slope = {:.2f}'.format(neg_M), (imshape[1] - 300, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(processed_image, '- int = {:.2f}'.format(neg_B), (imshape[1] - 300, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.line(processed_image, (int(neg_X1), int(neg_Y1)), (int(neg_X2), int(neg_Y2)), (255, 0, 0), 5)  # draw lines

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

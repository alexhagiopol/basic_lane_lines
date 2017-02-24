# Finding Lane Lines on the Road
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

This project uses a combination of Canny Edge Detection and the Hough Transform to detect lane lines in an input video stream.

## Writeup

### Goals
The goals of this project are the following:

    1. Make a pipeline that finds lane lines on the road.
    2. Reflect on my work in a written report.

### Demo

Execute the [iPython notebook](https://github.com/alexhagiopol/lane_lines/blob/develop/P1.ipynb)
in this repo to view the working implementation and its results. Below is
an example result of my lane line detection pipeline on a video stream:

[![Lane Lines Detection Example](figures/thumbnail.png)](https://www.youtube.com/watch?v=PPtjZ5sC2vk "Lane Lines Detection Example")

### Reflection

#### 1. Pipeline Description

The pipeline begins by preparing the local filesystem to read and write
images and videos. The pipeline then executes the detect function in a loop
on every input image or every input video frame depending on the application.
Finally, the pipeline saves edited images or videos that contain the lane line
estimates drawn on each image or video frame.

The core of the pipeline is a single function, detect(image),
that identifies lane lines in a single image or video frame. detect() has
three stages `1. filtering`, `2. edge and line segment detection`, and
`3. line segment combination`. In the `filtering` stage, assumptions are
made about which image information is useful for lane line detection, and non
useful information is thrown away before further processing. The `filtering`
stage implements 2 techniques: color masking and region of interest masking.
Color masking is used to only process parts of the image that have colors
similar to typical lane lines i.e. colors similar to white and yellow. Region
of interest masking is used to only process parts of the image that are in
the image region where lane lines would typically be found. These masks
are combined using the bitwise AND method to produce an output that typically
only contains the lane lines themselves along with some occasional spurious
information. An example of the filtered input is shown below:

#### 2. Potential Shortcomings

#### 3. Future Work

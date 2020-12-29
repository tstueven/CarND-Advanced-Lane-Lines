# **Finding Lane Lines on the Road**

## The second project of the Self-Driving Car Engineer Nanodegree

### A self driving car needs to find the lane lines in order to stay within them. This is the second shot.
---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set
  of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to
  center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane
  curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/undistort_output.jpg "Undistorted"

[image2]: ./output_images/test1_undist.jpg "Road Transformed"

[image3]: ./output_images/test1_binary_combo_example.jpg "Binary Example"

[image4]: ./output_images/straight_lines1_warped_example.jpg "Warp Example"

[image5]: ./output_images/lane_line_identification.jpg "Fit Visual"

[image6]: ./output_images/example_output.jpg "Output"

[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in  `lane_lines/calibration.py`.

I start by preparing "object points", which will be the (x, y, z) coordinates of
the chessboard corners in the world. Here I am assuming the chessboard is fixed
on the (x, y) plane at z=0, such that the object points are the same for each
calibration image. Thus, `objp` is just a replicated array of coordinates,
and `objpoints` will be appended with a copy of it every time I successfully
detect all chessboard corners in a test image.  `imgpoints` will be appended
with the (x, y) pixel position of each of the corners in the image plane with
each successful chessboard detection.

I then used the output `objpoints` and `imgpoints` to compute the camera
calibration and distortion coefficients using the `cv2.calibrateCamera()`
function.

The result of this I save in `camera_cal/camera_matrix.npy`
and `camera_cal/dist_coeff.npy` for the camera matrix and distortion
coefficients respectively which saves me from redoing the calibration every time
I restart the python session.

I applied this distortion correction to the first calibration image using
the `cv2.undistort()` function and obtained this result:

![alt text][image1]

There was the small obstacle that not all images had the same shape, some were
larger by a pixel in each dimension. I decided that the influence of a possible
offset of one pixel is sufficiently small to ignore this matter.

### Pipeline (single images)

The whole imaging processing pipeline is contained in the `Image_Processor`
class in `lane_lines/Image_Processor.py`. The processor is initialized by
providing the calibration data as well as the matrices for the perspective
transformations (more on this later).

#### 1. Provide an example of a distortion-corrected image.

The first step in processing in to add a new image to the processor via
the `Image_Processor.new_image(image)` method. Afterwards, this image can be
undistorted via the `undistort()` method. Access is possible via
the `img_undist` attribute. An example of this is provided in the third code
cell of the notebook `P2.ipynb` under **Undistort Image**.

Applying this to `test_images/test1.jpg` lead to the following result:

![alt text][image2]

The correction cannot be seen as clearly as in the case of the chess board but
e.g. the white car gives a hint that the image has been transformed.

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image. Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image
which all happens within the `Image_Processor`. With
the `calc_abs_gradient_binary` method the absolute gradient can be calculated
for different Sobel kernel sizes and is then averaged. The same can be done for
the direction by employing the `calc_dir_gradient_binary` method. Internally,
the class takes care that a Sobel operator of a certain size does not have to be
applied repeatedly in theses calculations.

The `calc_hls_thresh_binary` lets you choose a hls color channel and apply a
threshold to get a binary image.

An example of this which also demonstrates the access attributes and method is
provided in the fourth code cell of the notebook `P2.ipynb` under **Thresholding
Image**.

The thresholds for the gradient, especially the direction, are chosen relatively
hard to avoid detecting the roadside instead of the lane line. But in the end
the s channel threshold turned out to be much more important anyways.

Applying this to our undistorted test image leads to:

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The perspective transform is called via the `warp_perspective_binary` method
which takes the combination of the gradient and color thresholded binaries and
transforms the perspective with the transformation matrix that was set in the
initialization of the Image_Processor object.

An example of this which also demonstrates the access attributes and how to get
the colored image warped is provided in the fifth code cell of the
notebook `P2.ipynb` under **Perspective Transform**.

The transformation matrix is calculated before the initialization of the
Image_Processor with the `cv2.getPerspectiveTransform(src, dst)` function. The
source and destination points are hardcoded and chosen as follows

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 238, 688      | 280, 720       | 
| 1071, 688     | 1000, 720      |
| 595, 449      | 280, 0         |
| 687, 449      | 1000, 0        |

These were determined picking pixel coordinates by hand from the two straight
line images that were provided as examples.

I verified that my perspective transform was working as expected by drawing
the `src` and `dst` points onto a test image and its warped counterpart to
verify that the lines appear parallel in the warped image. The result is shown
below.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

There are two methods in the `Image_Processor` to identify the lange lines.
First, there is the `find_lane_pixels_sliding_window` method which identifies
pixels belonging to one or the other lane via histograms, counting, and a
sliding window as it was demonstrated in the lecture. This method is used if no
prior information is avaible or a lane line has not been for too long. The other
method, `find_lane_pixels_poly`, is used if we have prior information about were
the lane line should be from previous images. The polynomial fit (see below) is
used as an estimate and the respective lanes are only searched for in a margin
around that.

The result of applying these two methods to our test image can be seen below. On
the left are the sliding window. On the right, the information obtained from the
sliding window is applied on the same image, showing the polynomial margin in
which to search. One can see that, even though the left lane was only found in
the lower part of the image, the fitted polynamial seems to provide a very good
estimate of were to search in the future.

![alt text][image5]

The polynomial fitting happens in the `Line` class which is defined
in `lane_lines/Line.py`. There is one instance of this class for each lane in
the `Image_Processor` which not only provides all the fitting functionality but
also stores the information of previous images which are helpful in lane line
searching, smoothing the fit, and as the means of information if no lane is
found in a later image.

The polynomial fitting happens in the `fit_polynomial` method which takes in an
array of pixels (those that are found to be from a lane line) and fits a 2nd
degree polynomial to them. After the initial fitting, there are two sanity
checks. The first `sanity_check_other` compares the fit of both lanes in regard
to their curvature and distance. The second, `sanity_check_self`, checks of the
line is consistent with its own path, i.e. if it does not deviate too much at
the bottom.

Are both these checks passed, a smoothed fit is performed where the pixels from
the previous images are also taken into account. The points are weighted, with
more recent ones having a higher weight. If these checks fail, the old smoothed
fit is kept. If they fail too often in a row, the whole procedure starts anew,
with the sliding window and a fresh fit. All this is controlled in the
method `accept_fit` which takes a bool as input.

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The curvature is calculated in the `Line` class' `measure_curvature_real` method
for each line separately and can then be used of the sanity check. There is also
the `measure_curvature_real_smoothed` method which is used to get the curvature
from the smoothed polynomial. This we then use to calculate the output value of
the curvature which gets written onto the output image where we average over the
value from the two lines. The averaging happens
in `lane_lines/Image_Processor.py` line 357.

To get the correct value in metres the polynomial coefficients need to be
rescaled for which the following values were used:

```python
ym_per_pix = 50 / 720  # meters per pixel in y dimension
xm_per_pix = 3.7 / 700  # meters per pixel in x dimension
```

The distance of the individual lines from the centre is calculated in the `Line`
method `calc_line_based_position` and saved in the attribute `line_base_pos`.
The car's position is then determined by adding these two values
in `lane_lines/Image_Processor.py` line 359.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

The `Image_Processor` has a `draw_lane` method which return the output image. An
example is given below:

![alt text][image6]

The curvature seems to be in the correct order of magnitude and the value for
the distance from the centre looks reasonable to me as well.

---

### Pipeline (video)

#### 1. Provide a link to your final video output. Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Also the steps described above are concatenated in the `Image_Processor`'
s `process_image` method. The threshold values are hardcoded here for
convenience. This function can then be provided to the moviepy methods.

Here's a [link to my video result](./output_videos/project_video.mp4).

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project. Where will your pipeline likely fail? What could you do to make it more robust?

The biggest problem was probably to keep track of all the data from previous
images and put everything together as it is supposed to which led to a few
rounds of refactoring.

Of course, I was also faced with the typical problem in image analysis to find
threshold that work under different lighting conditions. The S channel seems
superior but also has problems in a few cases. The result is far from being
perfect byt I'd consider it a good second shot! :)

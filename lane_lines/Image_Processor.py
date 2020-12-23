import numpy as np
import matplotlib.image as mpimg
import cv2


class Image_Processor():
    def __init__(self, image):
        self.img_orig = mpimg.imread(image)
        self.img_undist = None
        self.img_gray = None
        self.img_hls = None
        self.sobel_x_dict = {}
        self.sobel_y_dict = {}
        self.abs_gradient_binary = np.zeros(self.img_orig.shape[:2])
        self.dir_gradient_binary = np.zeros(self.img_orig.shape[:2])
        self.hls_thresh_binary = np.zeros(self.img_orig.shape[:2])
        self.combined_binary = None
        self.combined_binary_warped = None

    def set_perspective_transform(self, perspective_transform_matrix,
                                  perspective_transform_matrix_inv):
        self.perspective_transform_matrix = perspective_transform_matrix
        self.perspective_transform_matrix_inv = perspective_transform_matrix_inv

    def undistort(self, cam_matrix, dist_coeff):
        self.img_undist = cv2.undistort(self.img_orig, cam_matrix, dist_coeff,
                                        None,
                                        cam_matrix)

    def grayscale(self):
        if self.img_gray is None:
            self.img_gray = cv2.cvtColor(self.img_undist,
                                         cv2.COLOR_RGB2GRAY)
        return self.img_gray

    def hls(self):
        if self.img_hls is None:
            self.img_hls = cv2.cvtColor(self.img_undist,
                                        cv2.COLOR_RGB2HLS)
        return self.img_hls

    def sobel_x(self, kernel_size=3):
        if kernel_size not in self.sobel_x_dict.keys():
            self.sobel_x_dict[kernel_size] = cv2.Sobel(self.grayscale(),
                                                       cv2.CV_64F, 1, 0,
                                                       ksize=kernel_size)
        return self.sobel_x_dict[kernel_size]

    def sobel_y(self, kernel_size=3):
        if kernel_size not in self.sobel_y_dict.keys():
            self.sobel_y_dict[kernel_size] = cv2.Sobel(self.grayscale(),
                                                       cv2.CV_64F, 0, 1,
                                                       ksize=kernel_size)
        return self.sobel_y_dict[kernel_size]

    def sobel_mag(self, kernel_sizes=(3, 5, 7)):
        # 3) Calculate the magnitude
        sum_x_sq = np.zeros(self.img_orig.shape[:2], dtype=np.float64)
        sum_y_sq = np.zeros(self.img_orig.shape[:2], dtype=np.float64)
        for kernel_size in kernel_sizes:
            sum_x_sq += self.sobel_x(kernel_size) ** 2
            sum_y_sq += self.sobel_y(kernel_size) ** 2
        abs_sobel = np.sqrt(sum_x_sq + sum_y_sq)
        # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
        return np.uint8(255 * abs_sobel / np.max(abs_sobel))

    def sobel_dir(self, kernel_sizes=(3, 5, 7)):
        # 3) Take the absolute value of the x and y gradients
        sum_x = np.zeros(self.img_orig.shape[:2], dtype=np.float64)
        sum_y = np.zeros(self.img_orig.shape[:2], dtype=np.float64)
        for kernel_size in kernel_sizes:
            sum_x += np.absolute(self.sobel_x(kernel_size))
            sum_y += np.absolute(self.sobel_y(kernel_size))
        # abs_sobel_x = np.absolute(self.sobel_x(kernel_size))
        # abs_sobel_y = np.absolute(self.sobel_y(kernel_size))
        # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient
        return np.arctan2(sum_y, sum_x)

    def calc_abs_gradient_binary(self, kernel_sizes=(3, 5, 7), thresh=(0, 255)):
        abs_gradient = self.sobel_mag(kernel_sizes)
        self.abs_gradient_binary = np.zeros(self.img_orig.shape[:2])
        self.abs_gradient_binary[
            (abs_gradient >= thresh[0]) & (abs_gradient <= thresh[1])] = 1

    def calc_dir_gradient_binary(self, kernel_size=3, thresh=(0, 90),
                                 degrees=True):
        if degrees:
            thresh = (np.deg2rad(thresh[0]), np.deg2rad(thresh[1]))
        grad_dir = self.sobel_dir(kernel_size)
        self.dir_gradient_binary = np.zeros(self.img_orig.shape[:2])
        self.dir_gradient_binary[
            (grad_dir >= thresh[0]) & (grad_dir <= thresh[1])] = 1

    def calc_hls_thresh_binary(self, channel='s', thresh=(0, 255)):
        if channel == 'h':
            img_channel = self.hls()[:, :, 0]
        elif channel == 'l':
            img_channel = self.hls()[:, :, 1]
        elif channel == 's':
            img_channel = self.hls()[:, :, 2]
        else:
            raise ValueError("Color channel must be 'h', 'l' or 's'.")
        self.hls_thresh_binary = np.zeros(self.img_orig.shape[:2])
        self.hls_thresh_binary[
            (img_channel >= thresh[0]) & (img_channel <= thresh[1])] = 1

    def calc_combined_binary(self):
        # and condition for gradient thresholds and then or for color channel
        self.combined_binary = (
                self.abs_gradient_binary * self.dir_gradient_binary
                + self.hls_thresh_binary)
        self.combined_binary[self.combined_binary > 1] = 1

    def color_binary(self):
        return np.dstack((np.zeros_like(self.abs_gradient_binary),
                          self.abs_gradient_binary * self.dir_gradient_binary,
                          self.hls_thresh_binary))

    def warp_perspective_binary(self, perspective_transform_matrix):
        self.combined_binary_warped = cv2.warpPerspective(self.combined_binary,
                                                 perspective_transform_matrix,
                                                 self.combined_binary.shape[
                                                 1::-1],
                                                 flags=cv2.INTER_LINEAR)

    def get_warped_perspective_img_undist(self, perspective_transform_matrix):
        return cv2.warpPerspective(self.img_undist,
                                   perspective_transform_matrix,
                                   self.combined_binary.shape[
                                   1::-1],
                                   flags=cv2.INTER_LINEAR)

    def find_lane_pixels_sliding_window(self, return_image=False):
        binary_warped = self.combined_binary_warped
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :],
                           axis=0)
        if return_image:
            # Create an output image to draw on and visualize the result
            out_img = np.dstack((binary_warped, binary_warped, binary_warped))
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0] // 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # HYPERPARAMETERS
        # Choose the number of sliding windows
        nwindows = 9
        # Set the width of the windows +/- margin
        margin = 100
        # Set minimum number of pixels found to recenter window
        minpix = 50

        # Set height of windows - based on nwindows above and image shape
        window_height = np.int(binary_warped.shape[0] // nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated later for each window in nwindows
        leftx_current = leftx_base
        rightx_current = rightx_base

        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window + 1) * window_height
            win_y_high = binary_warped.shape[0] - window * window_height
            ### TO-DO: Find the four below boundaries of the window ###
            win_xleft_low = leftx_current - margin  # Update this
            win_xleft_high = leftx_current + margin  # Update this
            win_xright_low = rightx_current - margin  # Update this
            win_xright_high = rightx_current + margin  # Update this

            if return_image:
                # Draw the windows on the visualization image
                cv2.rectangle(out_img, (win_xleft_low, win_y_low),
                              (win_xleft_high, win_y_high), (0, 255, 0), 2)
                cv2.rectangle(out_img, (win_xright_low, win_y_low),
                              (win_xright_high, win_y_high), (0, 255, 0), 2)

            # Identify the nonzero pixels in x and y within the window ###
            good_y_ind = (nonzeroy >= win_y_low) & (nonzeroy < win_y_high)
            good_left_inds = \
                ((nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)
                 & good_y_ind).nonzero()[0]
            good_right_inds = \
                ((nonzerox >= win_xright_low) & (nonzerox < win_xright_high)
                 & good_y_ind).nonzero()[0]

            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            # If you found > minpix pixels, recenter next window ###
            if len(good_left_inds) > minpix:
                leftx_current = np.int(nonzerox[good_left_inds].mean())
            if len(good_right_inds) > minpix:
                rightx_current = np.int(nonzerox[good_right_inds].mean())
            ### (`right` or `leftx_current`) on their mean position ###
            # pass # Remove this when you add your function

        # Concatenate the arrays of indices (previously was a list of lists of pixels)
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        pixels_left = np.array([leftx, lefty]).T
        pixels_right = np.array([rightx, righty]).T
        # More intuitive for me to have pixels grouped together like this
        if return_image:
            return pixels_left, pixels_right, out_img
        else:
            return pixels_left, pixels_right
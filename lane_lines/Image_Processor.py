import numpy as np
import cv2
from lane_lines import Line


class Image_Processor():
    def __init__(self, cam_matrix, dist_coeff, perspective_transform_matrix,
                 perspective_transform_matrix_inv):
        self.cam_matrix = cam_matrix
        self.dist_coeff = dist_coeff
        self.perspective_transform_matrix = perspective_transform_matrix
        self.perspective_transform_matrix_inv = perspective_transform_matrix_inv
        self.line_left = Line.Line()
        self.line_right = Line.Line()

    def new_image(self, image):
        # reset everything
        self.img_orig = image
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

    def undistort(self):
        self.img_undist = cv2.undistort(self.img_orig, self.cam_matrix,
                                        self.dist_coeff,
                                        None,
                                        self.cam_matrix)

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

    def calc_dir_gradient_binary(self, kernel_sizes=(3, 5, 7), thresh=(0, 90),
                                 degrees=True):
        if degrees:
            thresh = (np.deg2rad(thresh[0]), np.deg2rad(thresh[1]))
        grad_dir = self.sobel_dir(kernel_sizes)
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

    def warp_perspective_binary(self):
        self.combined_binary_warped = cv2.warpPerspective(self.combined_binary,
                                                          self.perspective_transform_matrix,
                                                          self.combined_binary.shape[
                                                          1::-1],
                                                          flags=cv2.INTER_LINEAR)

    def get_warped_perspective_img_undist(self):
        return cv2.warpPerspective(self.img_undist,
                                   self.perspective_transform_matrix,
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

        self.line_left.fit_polynomial(pixels_left)
        self.line_right.fit_polynomial(pixels_right)
        if return_image:
            out_img[lefty, leftx] = [255, 0, 0]
            out_img[righty, rightx] = [0, 0, 255]
            return out_img

    def find_lane_pixels_poly(self, return_image=False):
        binary_warped = self.combined_binary_warped
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :],
                           axis=0)
        if return_image:
            # Create an output image to draw on and visualize the result
            out_img = np.dstack((binary_warped, binary_warped, binary_warped))

        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # HYPERPARAMETERS
        # Set the width of the windows +/- margin
        margin = 100

        poly_values_left = self.line_left.get_poly_pixel_x_values(nonzeroy)
        poly_values_right = self.line_right.get_poly_pixel_x_values(nonzeroy)
        left_lane_inds = (nonzerox > poly_values_left - margin) & (
                nonzerox < poly_values_left + margin)
        right_lane_inds = (nonzerox > poly_values_right - margin) & (
                nonzerox < poly_values_right + margin)

        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        pixels_left = np.array([leftx, lefty]).T
        pixels_right = np.array([rightx, righty]).T
        # More intuitive for me to have pixels grouped together like this
        self.line_left.fit_polynomial(pixels_left)
        self.line_right.fit_polynomial(pixels_right)

        if return_image:
            ## Visualization ##
            # Create an image to draw on and an image to show the selection window
            out_img = np.dstack(
                (binary_warped, binary_warped, binary_warped)) * 255
            window_img = np.zeros_like(out_img)
            # Color in left and right line pixels
            out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255,
                                                                           0, 0]
            out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0,
                                                                             0,
                                                                             255]

            plot_y = np.linspace(0, self.combined_binary_warped.shape[0],
                                 self.combined_binary_warped.shape[0])
            left_fit_x = self.line_left.get_poly_pixel_x_values(plot_y)
            right_fit_x = self.line_right.get_poly_pixel_x_values(plot_y)

            # Generate a polygon to illustrate the search window area
            # And recast the x and y points into usable format for cv2.fillPoly()
            left_line_window1 = np.array(
                [np.transpose(np.vstack([left_fit_x - margin, plot_y]))])
            left_line_window2 = np.array(
                [np.flipud(np.transpose(np.vstack([left_fit_x + margin,
                                                   plot_y])))])
            left_line_pts = np.hstack((left_line_window1, left_line_window2))
            right_line_window1 = np.array(
                [np.transpose(np.vstack([right_fit_x - margin, plot_y]))])
            right_line_window2 = np.array(
                [np.flipud(np.transpose(np.vstack([right_fit_x + margin,
                                                   plot_y])))])
            right_line_pts = np.hstack((right_line_window1, right_line_window2))

            # Draw the lane onto the warped blank image
            cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
            cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
            result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
            return result

    def find_lane_line_pixels(self):
        if (self.line_left.fail_counter < 10) and (
                self.line_right.fail_counter < 10):
            self.find_lane_pixels_poly()
        else:
            self.find_lane_pixels_sliding_window()

        if self.line_left.sanity_check_other(self.line_right):
            if self.line_left.sanity_check_self():
                self.line_left.accept_fit(True)
            else:
                self.line_left.accept_fit(False)
            if self.line_right.sanity_check_self():
                self.line_right.accept_fit(True)
            else:
                self.line_right.accept_fit(False)
        else:
            self.line_left.accept_fit(False)
            self.line_right.accept_fit(False)

    def draw_lane(self):
        warp_zero = np.zeros_like(self.combined_binary_warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        plot_y = np.linspace(0, self.combined_binary_warped.shape[0],
                             self.combined_binary_warped.shape[0])
        left_fit_x = self.line_left.get_poly_pixel_x_values(plot_y)
        right_fit_x = self.line_right.get_poly_pixel_x_values(plot_y)

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fit_x, plot_y]))])
        pts_right = np.array(
            [np.flipud(np.transpose(np.vstack([right_fit_x, plot_y])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        unwarp = cv2.warpPerspective(color_warp,
                                     self.perspective_transform_matrix_inv,
                                     (self.img_undist.shape[1],
                                      self.img_undist.shape[0]))
        # Combine the result with the original image
        result = cv2.addWeighted(self.img_undist, 1, unwarp, 0.3, 0)

        return result

    def process_image(self, image):
        self.new_image(image)
        self.undistort()
        self.calc_abs_gradient_binary(kernel_sizes=(3, 7, 11),
                                      thresh=(60, 255))
        self.calc_dir_gradient_binary(kernel_sizes=(3, 7, 11),
                                      thresh=(30, 80), degrees=True)
        self.calc_hls_thresh_binary(channel='s', thresh=(170, 255))
        self.calc_combined_binary()
        self.warp_perspective_binary()
        self.find_lane_line_pixels()
        # self.line_left.measure_curvature_real()
        return self.draw_lane()

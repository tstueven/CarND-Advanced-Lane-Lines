import numpy as np


# Define a class to receive the characteristics of each line detection
class Line():
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 50 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        # average x values of the fitted line over the last n iterations
        self.bestx = None
        # polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        # polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        # radius of curvature of the line in some units
        self.radius_of_curvature = None
        # distance in meters of vehicle center from the line
        self.line_base_pos = None
        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0, 0, 0], dtype='float')
        # values for detected line pixels
        self.pixels = None
        self.all_pixels = []
        self.fail_counter = 0

    def fit_polynomial(self, pixels):
        self.pixels = pixels
        px_x, px_y = pixels[:, 0], pixels[:, 1]
        # Fit a second order polynomial to each using `np.polyfit`
        self.current_fit = np.polyfit(px_y,
                                      px_x, 2)

    def fit_polynomial_smoothed(self):
        # TODO : Probably weigh more recent ones more?
        pixels = np.concatenate(self.all_pixels[-10:])
        px_x, px_y = pixels[:, 0], pixels[:, 1]
        # Fit a second order polynomial to each using `np.polyfit`
        self.best_fit = np.polyfit(self.ym_per_pix * px_y,
                                   self.xm_per_pix * px_x, 2)

    def measure_curvature_real(self):
        '''
        Calculates the curvature of polynomial functions in meters.
        '''

        # Define y-value where we want radius of curvature
        # We'll choose the maximum y-value, corresponding to the bottom of the image
        y_eval = 720

        A = self.current_fit[0] * self.xm_per_pix / (self.ym_per_pix ** 2)
        B = self.current_fit[1] * self.xm_per_pix / self.ym_per_pix

        # Implement the calculation of R_curve (radius of curvature) #####
        self.radius_of_curvature = ((1 + (
                2 * A * y_eval * self.ym_per_pix + B) ** 2) ** 1.5) / np.absolute(
            2 * A)

        return self.radius_of_curvature

    def measure_curvature_real_smoothed(self):
        '''
        Calculates the curvature of polynomial functions in meters.
        '''

        # Define y-value where we want radius of curvature
        # We'll choose the maximum y-value, corresponding to the bottom of the image
        y_eval = 720

        A = self.best_fit[0] * self.xm_per_pix / (self.ym_per_pix ** 2)
        B = self.best_fit[1] * self.xm_per_pix / self.ym_per_pix

        # Implement the calculation of R_curve (radius of curvature) #####
        self.radius_of_curvature = ((1 + (
                2 * A * y_eval * self.ym_per_pix + B) ** 2) ** 1.5) / np.absolute(
            2 * A)

        return self.radius_of_curvature

    def distance_current(self, other):
        """
        Returns minimal and maximal distance between two lines.
        """
        this_poly = np.poly1d(self.current_fit)
        other_poly = np.poly1d(other.current_fit)
        y = np.linspace(0, 30, 100)
        diff = other_poly(y) - this_poly(y)
        return diff.min(), diff.max()

    def distance_self_past(self):
        this_poly = np.poly1d(self.current_fit)
        other_poly = np.poly1d(self.best_fit)
        y = np.linspace(0, 30, 100)
        diff = other_poly(y) - this_poly(y)
        return diff.min(), diff.max()

    def sanity_check_other(self, other):
        # Check lines are separated at approximately the same distance
        dist_min, dist_max = self.distance_current(other)
        if dist_min < 3 or dist_max > 4.5:
            return False
        # Check for similar curvature
        radius_quotient = self.radius_of_curvature / other.radius_of_curvature
        if self.radius_of_curvature < 2000 and (
                radius_quotient > 1.25 or radius_quotient < 0.8):
            return False
        # Rough parallelity implicit in other two conditions

        return True

    def sanity_check_self(self):
        # Check if line stays similar to before
        dist_min, dist_max = self.distance_self_past()
        if dist_max > 0.3:
            return False
        return True

    def accept_fit(self, accept):
        if accept:
            self.detected = True
            self.all_pixels.append(self.pixels)
            self.fail_counter = 0
            self.fit_polynomial_smoothed()
            self.measure_curvature_real_smoothed()
        else:
            self.detected = False
            # self.all_pixels.append(np.array([[]]))
            self.fail_counter += 1

    def get_poly_pixel_x_values(self, y_values):
        if self.best_fit is not None:
            poly = np.poly1d(self.best_fit)
        else:
            poly = np.poly1d(self.current_fit)
        return poly(y_values)
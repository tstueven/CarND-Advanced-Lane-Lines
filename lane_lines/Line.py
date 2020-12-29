import numpy as np


# Define a class to receive the characteristics of each line detection
class Line():
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 50 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        # polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        # radius of curvature of the line in some units
        self.radius_of_curvature = None
        # distance in meters of vehicle center from the line
        self.line_base_pos = None
        # values for detected line pixels
        self.pixels = None
        self.all_pixels = []
        self.all_weights = []
        self.fail_counter = 1000

    def fit_polynomial(self, pixels):
        self.pixels = pixels
        px_x, px_y = pixels[:, 0], pixels[:, 1]
        # Fit a second order polynomial to each using `np.polyfit`
        self.current_fit = np.polyfit(px_y, px_x, 2)

    def fit_polynomial_smoothed(self):
        pixels = np.concatenate(self.all_pixels[-15:])
        weights = []
        for i, w in enumerate(self.all_weights[-15:]):
            weights.append((i + 1) / np.sqrt(len(w)) * w)
        weights = np.concatenate(weights)
        px_x, px_y = pixels[:, 0], pixels[:, 1]
        # Fit a second order polynomial to each using `np.polyfit`
        self.best_fit = np.polyfit(px_y, px_x, 2)  # , w=weights)

    def calc_line_based_position(self):
        pix_pos = np.poly1d(self.best_fit)(720)
        self.line_base_pos = self.xm_per_pix * (pix_pos - 1280 / 2)

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

    def distance_other(self, other):
        """
        Returns minimal and maximal distance between two lines.
        """
        this_poly = np.poly1d(self.current_fit)
        other_poly = np.poly1d(other.current_fit)
        y = np.linspace(0, 720, 721)
        diff = np.absolute(other_poly(y) - this_poly(y)) * self.xm_per_pix
        return diff.min(), diff.max()

    def distance_self_past(self):
        this_poly = np.poly1d(self.current_fit)
        other_poly = np.poly1d(self.best_fit)
        y = np.linspace(0, 30, 100)
        diff = np.absolute(other_poly(y) - this_poly(y)) * self.xm_per_pix
        return diff.min(), diff.max()

    def sanity_check_other(self, other):
        # Check lines are separated at approximately the same distance
        dist_min, dist_max = self.distance_other(other)
        if dist_min < 3.2 or dist_max > 4.4:
            return False
        # Check for similar curvature
        radius_quotient = self.measure_curvature_real() / other.measure_curvature_real()
        if self.radius_of_curvature < 2000:
            if radius_quotient > 1.25 or radius_quotient < 0.8:
                return False
            if np.sign(self.best_fit[0]) != np.sign(self.best_fit[0]):
                if (self.radius_of_curvature < 1000
                        or other.radius_of_curvature < 1000):
                    return False
        # Rough parallelity implicit in other two conditions

        return True

    def sanity_check_self(self):
        if self.best_fit is None:
            return True
        # Check if line stays similar to before
        dist_min, dist_max = self.distance_self_past()
        if dist_max > 0.4:
            return False
        return True

    def accept_fit(self, accept):
        if accept:
            self.detected = True
            if self.fail_counter > 1:
                self.all_pixels = []
                self.all_weights = []
            self.all_pixels.append(self.pixels)
            self.all_weights.append(np.ones(len(self.pixels)))
            self.fail_counter = 0
            self.fit_polynomial_smoothed()
            self.calc_line_based_position()
            self.measure_curvature_real_smoothed()
        else:
            self.detected = False
            self.all_pixels.append(self.pixels)
            self.all_weights.append(np.zeros(len(self.pixels)))
            self.fail_counter += 1

    def get_poly_pixel_x_values(self, y_values):
        if self.best_fit is not None:
            poly = np.poly1d(self.best_fit)
        else:
            poly = np.poly1d(self.current_fit)
        return poly(y_values)

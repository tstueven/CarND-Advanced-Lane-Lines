import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import importlib  # import lane_lines.calibration
# importlib.reload(lane_lines.calibration)
from lane_lines.calibration import load_calibration
from lane_lines import gradient
from lane_lines import color
from lane_lines import lane_lines

cam_matrix, dist_coeff = load_calibration('camera_cal')

# %% md

## Undistort Image

# %%

img_orig = mpimg.imread('test_images/test1.jpg')

img_undist = cv2.undistort(img_orig, cam_matrix, dist_coeff, None, cam_matrix)

cv2.imwrite('und.jpg', img_orig)

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(img_orig)  # ,cmap='gray')
ax1.set_title('Original Image', fontsize=50)
ax2.imshow(img_undist, cmap='gray')
ax2.set_title('Undistorted Image', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

# %%
plt.show()
import glob
import os.path
from calibration import *

# Check if calibration parameters are already stored in file
if os.path.isfile('camera_cal/camera_matrix.npy') and os.path.isfile(
        'camera_cal/dist_coeff.npy'):
    camera_matrix = np.load('camera_cal/camera_matrix.npy')
    dist_coeff = np.load('camera_cal/dist_coeff.npy')
# if not, perform the calibration
else:
    # Make a list of calibration images
    images = glob.glob('camera_cal/calibration*.jpg')
    camera_matrix, dist_coeff = calibrate(images)
    np.save('camera_cal/camera_matrix.npy', camera_matrix)
    np.save('camera_cal/dist_coeff.npy', dist_coeff)

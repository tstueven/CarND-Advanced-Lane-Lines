import os.path
import glob
import numpy as np
import cv2


def calibrate(images):
    objp = np.zeros((6 * 9, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d points in real world space
    imgpoints = []  # 2d points in image plane.

    gray_shape = None
    # Step through the list and search for chessboard corners
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        print(gray.shape)
        if gray_shape is None:
            # TODO : Find out about different image shapes
            #    assert (gray_shape[0] == gray.shape[0] and gray_shape[1] ==
            #            gray.shape[1])
            # else:
            gray_shape = gray.shape

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

        # If found, add object points, image points
        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (9, 6), corners, ret)
            cv2.imshow('img', img)
            cv2.waitKey(500)

    cv2.destroyAllWindows()

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints,
                                                       gray_shape[::-1], None,
                                                       None)

    return mtx, dist


def load_calibration(folder):
    # Check if calibration parameters are already stored in file
    if os.path.isfile(folder + '/camera_matrix.npy') and os.path.isfile(
            folder + '/dist_coeff.npy'):
        camera_matrix = np.load(folder + '/camera_matrix.npy')
        dist_coeff = np.load(folder + '/dist_coeff.npy')
    # if not, perform the calibration
    else:
        # Make a list of calibration images
        images = glob.glob(folder + '/calibration*.jpg')
        camera_matrix, dist_coeff = calibrate(images)
        np.save(folder + '/camera_matrix.npy', camera_matrix)
        np.save(folder + 'camer_cal/dist_coeff.npy', dist_coeff)
        
    return camera_matrix, dist_coeff
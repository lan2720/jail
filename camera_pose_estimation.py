import numpy as np
import cv2
import glob

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
# checkerboard Dimensions
cbrow = 2
cbcol = 2
objp = np.zeros((cbrow*cbcol,3), np.float32)
#objp[:,:2] = np.mgrid[0:cbrow,0:cbcol].T.reshape(-1,2)
objp[:,:2] = np.float32([[0, 0], [0, 40], [60, 0], [60, 40]])
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = ['screenshot.jpg']#glob.glob('images_calib_1123/*.jpg')

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    corners = np.array([[75,70],[79,114],[137,55],[141,98]], dtype=np.float32).reshape(-1,1,2) 
    objpoints.append(objp)
    #corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
    imgpoints.append(corners)
    
# Save parameters
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
np.savez("trail_room_calibration_output.npz", ret=ret, mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)



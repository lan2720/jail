import cv2
import numpy as np
import glob
from datetime import datetime


# Load previously saved data
with np.load('trail_room_calibration_output.npz') as X:
    mtx, dist, rvecs, tvecs = [X[i] for i in ('mtx','dist','rvecs','tvecs')]


def rectify():
    img = cv2.imread('/home/jianan/Desktop/20181122_15h00m29s973435.jpg')
    h,  w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

    # undistort
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

    # crop the image
    x,y,w,h = roi
    dst = dst[y:y+h, x:x+w]
    cv2.imwrite('calibresult.png', dst)



def draw(img, corners, imgpts):
    imgpts = np.int32(imgpts).reshape(-1,2)

    # draw ground floor in green
    img = cv2.drawContours(img, [imgpts[:4]],-1,(0,255,0),-3)

    ## draw pillars in blue color
    #for i,j in zip(range(4),range(4,8)):
    #    img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]),(255),3)
    img = cv2.line(img, tuple(imgpts[0]), tuple(imgpts[4]),(0, 0, 255),3)

    ## draw top layer in red color
    #img = cv2.drawContours(img, [imgpts[4:]],-1,(0,0,255),3)

    return img


criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

#axis = np.float32([[0,0,0], [0,3,0], [3,3,0], [3,0,0],
#                   [0,0,-3],[0,3,-3],[3,3,-3],[3,0,-3] ])
axis = np.float32([[0, 0, 0], [0, 40, 0], [60, 40, 0], [60, 0, 0], [0, 0, 50]])

cap = cv2.VideoCapture('给杨博士/法制行为录像/倒地.mp4')
while True:
    # get image from webcam
    _, img = cap.read()
# for fname in glob.glob('*.jpg'):
    # img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners = np.array([[75,70],[79,114],[137,55],[141,98]], dtype=np.float32).reshape(-1,1,2)
    corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)

    # project 3D points to image plane
    imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)

    img = draw(img, corners2, imgpts)
    cv2.imshow('img', img)
    k = cv2.waitKey(0)
    if k == ord('s'):
        filename = datetime.now().strftime('%Y%m%d_%Hh%Mm%Ss%f') + '.jpg'
        cv2.imwrite('3d_sample/'+filename, img)
        print('save an image in %s' % filename)
    elif k == ord('q'):
        break

cv2.destroyAllWindows()

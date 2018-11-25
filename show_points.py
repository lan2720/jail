import numpy as np
import cv2

img = cv2.imread('screenshot.jpg')
points = np.array([[75,70],[79,114],[137,55],[141,98]])
for p in points:
    print(p.tolist())
    cv2.circle(img, tuple(p.tolist()), 2, (0,0,255), -1, 8)

while True:
    cv2.imshow('img', img)
    key = cv2.waitKey(0)
    if key == ord('q'):
        break
cv2.destroyAllWindows()

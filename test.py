import cv2
import numpy as np
x1 = 10
x2 = 20
x3 = 35
x4 = 67
y1 = 23
y2 = 56
y3 = 34
y4 = 23
cnt = np.array([[x1,y1],[x2,y2],[x3,y3],[x4,y4]]) # 必须是array数组的形式

rect = cv2.minAreaRect(cnt) # 得到最小外接矩形的（中心(x,y), (宽,高), 旋转角度）
box = cv2.boxPoints(rect) # cv2.boxPoints(rect) for OpenCV 3.x 获取最小外接矩形的4个顶点
box = np.int0(box)


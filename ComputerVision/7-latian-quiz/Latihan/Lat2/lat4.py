import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread('assets/chessboard2.jpg')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
harris = cv2.cornerHarris(img_gray, 2, 5, 0.04)

no_SubPix = img.copy()
no_SubPix[harris > 0.01 * harris.max()] = [0,0,255]
_, thres = cv2.threshold(harris, 0.01 * harris.max(), 255, 0)
thres = np.uint8(thres)

_, _, _, centroid = cv2.connectedComponentsWithStats(thres)
centroid = np.float32(centroid)
criteria = [cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.01]
enhanced_corner = cv2.cornerSubPix(
  harris, cent
)
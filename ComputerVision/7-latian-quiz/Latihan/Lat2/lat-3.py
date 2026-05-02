import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('assets/chessboard2.jpg')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
harris = cv2.cornerHarris(img_gray, 2, 5, 0.04)

no_SubPix = img.copy()
no_SubPix[harris > 0.01 * harris.max()] = [0,0,255]

_, thres = cv2.threshold(harris, 0.01 * harris.max(), 255, 0)
thres = np.uint8(thres)

_, _, _, centroid = cv2.connectedComponentsWithStats(thres)
centroid = np.float32(centroid)

criteria = [cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 0.01]
enhanced_corner = cv2.cornerSubPix(
  harris, centroid, (5,5), (-1,-1), criteria
)

SubPix = img.copy()
centroid = np.uint(centroid)
for centroid in centroid:
  x = centroid[0]
  y = centroid[1]
  SubPix[y,x] = [0,0,255]

enchanced_corner = np.uint(enhanced_corner)
for corner in enhanced_corner:
  x = corner[0]
  y = corner[1]
  SubPix[y,x] = [0,0,255]

dir = 'assets/images'

for i, image_name in enumerate(os.listdir(dir)):
  image = cv2.imread(dir + '/' + image_name)
  fast = cv2.FastFeatureDetector.create()

  keypoint = fast.detect(image, None)
  fast_res = image.copy()

  cv2.drawKeypoints(image, keypoint, fast_res, cv2.COLOR_BGR2RGB)
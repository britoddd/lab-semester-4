import cv2
import matplotlib.pyplot as plt
import os
import numpy as np

img = cv2.imread('assets/chessboard2.jpg')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
harris = cv2.cornerHarris(img_gray, 2, 5, 0.04)

no_SubPix = img.copy()
no_SubPix[harris > 0.01 * harris.max()] = [0, 0, 255]

_, thres = cv2.threshold(harris, 0.01 * harris.max(), 255, 0)
thres = np.uint8(thres)

_, _, _, centroid = cv2.connectedComponentsWithStats(thres)

centroid = np.float32(centroid)
criteria = [cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 0.01]
enhanced_corner = cv2.cornerSubPix(
  harris, centroid, (5,5), (-1,-1), criteria
)

SubPix = img.copy()

centroid = np.uint16(centroid)
for centroid in centroid:
  x = centroid[0]
  y = centroid[1]
  SubPix[y,x] = [0,0,255]

enhanced_corner = np.uint16(enhanced_corner)
for corner in enhanced_corner:
  x = corner[0]
  y = corner[1]
  SubPix[y,x] = [255,0,0]

dir = 'assets/images'

for i, image_name in enumerate(os.listdir(dir)):
  image = cv2.imread(dir + '/' + image_name)
  fast = cv2.FastFeatureDetector.create()

  keypoint = fast.detect(image, None)
  fast_res = image.copy()
  cv2.drawKeypoints(image, keypoint, fast_res, cv2.COLOR_BGR2RGB)

  plt.subplot(2, 3, i+1)
  plt.imshow(fast_res)
  plt.axis(False)

for i, image_name in enumerate(os.listdir(dir)):
  image = cv2.imread(dir + '/' + image_name)
  orb = cv2.ORB.create()

  keypoint = orb.detect(image, None)
  orb_res = image.copy()
  cv2.drawKeypoints(image, keypoint, orb_res, [0,255,0])

  orb_res = cv2.cvtColor(orb_res, cv2.COLOR_BGR2RGB)

  plt.subplot(2, 3, i+3+1)
  plt.imshow(orb_res)
  plt.axis(False)
plt.show()
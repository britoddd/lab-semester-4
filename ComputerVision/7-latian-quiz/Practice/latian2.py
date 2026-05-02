import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

PATH = 'images/'
IMAGE_PATH = PATH + 'source/'
TARGET_PATH = PATH + 'target/hina.png'

target = cv2.imread(TARGET_PATH)
target_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
target_rgb = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
target_gray = cv2.GaussianBlur(target, (3,3), 0)

akaze = cv2.AKAZE.create()

target_kp, target_dc = akaze.detectAndCompute(target_gray, None)
target_dc = np.float32(target_dc)

best_name = ''
best_match = 0

for img_path in os.listdir(IMAGE_PATH):
  img = cv2.imread(IMAGE_PATH + img_path)
  img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  img_gray = cv2.GaussianBlur(img, (3,3), 0)

  img_kp, img_dc = akaze.detectAndCompute(img_gray, None)
  img_dc = np.float32(img_dc)

  idx_param = dict(algorithm=1, trees=5)
  search_param = dict(checks=50)
  flann = cv2.FlannBasedMatcher(idx_param, search_param)

  matches = flann.knnMatch(target_dc, img_dc, 2)
  matches_mask = [[0,0] for _ in range(len(matches))]

  current_match = 0

  for i, (fm, sm) in enumerate(matches):
    if fm.distance < 0.7 * sm.distance:
      matches_mask[i] = [1,0]
      current_match += 1

  if current_match > best_match:
    best_match = current_match
    best_name = img_path
    best_img = cv2.drawMatchesKnn(target_rgb, target_kp,
                                  img_rgb, img_kp,
                                  matches, None,
                                  matchColor=[0,0,255],
                                  matchesMask=matches_mask,
                                  singlePointColor=[255,0,0])
                                  
plt.imshow(best_img)
plt.show()
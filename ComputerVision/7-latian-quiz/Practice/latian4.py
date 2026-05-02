import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

PATH = 'images/'
TARGET_PATH = PATH + 'target/hina.png'
SOURCE_PATH = PATH + 'source/'

akaze = cv2.AKAZE.create()

def preprocess(img_path):
  img = cv2.imread(img_path)
  img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
  img_gray = cv2.GaussianBlur(img_gray, (3,3), 0)
  return (img_gray, img_rgb)

target_gray, target_rgb = preprocess(TARGET_PATH)
target_kp, target_dc = akaze.detectAndCompute(target_gray, None)
target_dc = np.float32(target_dc)

idx = dict(algorithm=1, trees=5)
search = dict(checks=50)

best_name = ''
best_match = 0
for img_path in os.listdir(SOURCE_PATH):
  img_gray, img_rgb = preprocess(SOURCE_PATH + img_path)
  img_kp, img_dc = akaze.detectAndCompute(img_gray, None)
  img_dc = np.float32(img_dc)

  flann = cv2.FlannBasedMatcher(idx, search)
  matches = flann.knnMatch(target_dc, img_dc, 2)
  macthes_mask = [[0,0] for _ in range(len(matches))]

  curr_match = 0
  for i, (fm, sm) in enumerate(matches):
    if fm.distance < 0.7 * sm.distance:
      macthes_mask[i] = [1,0]
      curr_match += 1
  
  if best_match < curr_match:
    best_match = curr_match
    best_name = img_path
    best_result = cv2.drawMatchesKnn(target_rgb, target_kp,
                                     img_rgb, img_kp,
                                     matches, None,
                                     matchColor=[255,0,0],
                                     singlePointColor=[0,0,255],
                                     matchesMask=macthes_mask)
    
plt.imshow(best_result)
plt.show()
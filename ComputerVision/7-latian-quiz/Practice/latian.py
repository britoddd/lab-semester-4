import cv2
import numpy as np
import matplotlib.pyplot as plt 
import os

PATH = 'images/'
IMAGE_PATH = PATH + 'source/'
TARGET_PATH = PATH + 'target/hina.png'

target_rgb = cv2.cvtColor(cv2.imread(TARGET_PATH), cv2.COLOR_BGR2RGB)

def preprocess(image):
  image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  image = cv2.GaussianBlur(image, (3,3), 0)
  # image = cv2.equalizeHist(image)
  return image

target = cv2.imread(TARGET_PATH)
if target is None:
    print("Error: Could not load image. Check the file path!")
grey_target = preprocess(target)

sift = cv2.SIFT_create()
orb = cv2.ORB_create()
akaze = cv2.AKAZE.create()

target_kp, target_dc = akaze.detectAndCompute(grey_target, None)
target_dc = np.float32(target_dc)

best_name = ''
best_match = 0
for image_path in os.listdir(IMAGE_PATH):
  img = cv2.imread(IMAGE_PATH + image_path)
  img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  img = preprocess(img)
  img_kp, img_dc = akaze.detectAndCompute(img, None)
  img_dc = np.float32(img_dc)

  idx_param = dict(algorithm=1, trees=5)
  search_param = dict(checks=50)
  flann = cv2.FlannBasedMatcher(idx_param, search_param)

  # Kalkulasi match dari descriptornya
  match = flann.knnMatch(target_dc, img_dc, 2)
  matches_mask = [[0,0] for _ in range(len(match))]
  current_match = 0

  for i, (fm, sm) in enumerate(match):
    if fm.distance < 0.7 * sm.distance:
      matches_mask[i] = [1, 0]
      current_match += 1

  if current_match > best_match:
    best_match = current_match
    best_name = image_path
    best_match_data = {
      'image_data': img,
      'keypoint': img_kp,
      'image_rgb': img_rgb,
      'description': img_dc,
      'match': match,
      'matchesMask': matches_mask,
    }

  print(image_path, current_match)

result = cv2.drawMatchesKnn(
  target_rgb,
  target_kp,
  best_match_data['image_rgb'],
  best_match_data['keypoint'],
  best_match_data['match'],
  None,
  matchesMask=best_match_data['matchesMask'],
  matchColor=[0,0,255],
  singlePointColor=[255,0,0]
)

plt.figure()
plt.title(f"Best Match: {best_name} | score: {best_match}")
plt.imshow(result)
plt.show()
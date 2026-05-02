import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

PATH = 'Dataset/'
object = cv2.imread(PATH + 'Object.jpg')
object = cv2.cvtColor(object, cv2.COLOR_BGR2RGB)

DATA_PATH = PATH + 'Data/'
data = []
for image_path in os.listdir(DATA_PATH):
  image_path = DATA_PATH + image_path
  image_data = cv2.imread(image_path)
  data.append(image_data)

gray = cv2.cvtColor(object, cv2.COLOR_RGB2GRAY)
gray = cv2.medianBlur(gray, 3)
gray = cv2.equalizeHist(gray)

sift = cv2.SIFT_create()
orb = cv2.ORB_create()
akaze = cv2.AKAZE_create()

target_keypoint, target_desc = sift.detectAndComputer(gray, None)
target_desc = np.float32(target_desc)

for i, img in enumerate(data):
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
  img = cv2.medianBlur(img, 3)
  img = cv2.equalizeHist(img)

  img_keypoint, img_desc = sift.detectAndComputer(img, None)
  img_desc = np.float32(img_desc)

index_param = dict(algorithm=1)
search_param = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_param, search_param)
match = flann.knnMatch(target_desc, img_desc, 2)
matches_mask = [[0,0] for _ in range(len(match))]

best_match = 0
curr_match = 0

for i, (fm, sm) in enumerate(match):
  if fm.distance < 0.7 * sm.distance:
    matches_mask[i] = [1,0]
    curr_match += 1

  if best_match < curr_match:
    best_match = curr_match
    best_match
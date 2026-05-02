import cv2
import numpy as np
import matplotlib.pyplot as plt 
import os

PATH = 'images/'
IMAGE_PATH = PATH + 'source/'
TARGET_PATH = PATH + 'target/hina.png'

def preprocess(image):
  image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  image = cv2.medianBlur(image, 3)
  image = cv2.equalizeHist(image)
  return image

target = cv2.imread('images/target/hina.png')
if target is None:
    print("Error: Could not load image. Check the file path!")
grey_target = preprocess(target)

sift = cv2.SIFT_create()
orb = cv2.ORB_create()
akaze = cv2.AKAZE_create()

data = []
for image_path in os.listdir(IMAGE_PATH):
  image = cv2.imread(image_path)
  data.append(image)

target_kp, target_dc = akaze.detectAndCompute(grey_target, None)
target_dc = np.float32(target_dc)

idx_param = dict(algorithm=1)
search_param = dict(checks=50)
flann = cv2.FlannBasedMatcher(idx_param, search_param)

best_match = 0
for idx, img in enumerate(data):
  img = preprocess(img)
  img_kp, img_dc = akaze.detectAndComputer(img, None)
  img_dc = np.float32(img_dc)

  # Kalkulasi match dari descriptornya
  match = flann.knnMatch(target_dc, img_dc, 2)
  print(match)
  # matches_mask = [[0,0] for _ in range(len(match))]

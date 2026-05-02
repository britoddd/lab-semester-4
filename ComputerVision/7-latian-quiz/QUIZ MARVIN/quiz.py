import os 
import cv2
import matplotlib.pyplot as plt 
import numpy as np 

#LISTING
source_file = os.listdir('source')
target_file = os.listdir('target')

#LOAD IMAGE
img_target = cv2.imread('target/hina.png')

#SMOOTHING
gray_target = cv2.cvtColor(img_target, cv2.COLOR_BGR2GRAY)
blur_target = cv2.GaussianBlur(gray_target, (3,3), 0)

for i, filename in enumerate(source_file):
    img_source = cv2.imread('source/' + filename)
    gray_source = cv2.cvtColor(img_source, cv2.COLOR_BGR2GRAY)
    blur_source = cv2.GaussianBlur(gray_source, (3,3), 0)

    #FEATURE DETECTION
    #HARRIS
    gray = np.float32(blur_source)
    harris = cv2.cornerHarris(gray, 2, 3, 0.04)
    harrisRes = img_source.copy()
    harrisRes[harris < 0.01*harris.max()] = [0,255,0]
    harrisRes = cv2.cvtColor(harrisRes, cv2.COLOR_BGR2RGB)

    #FAST
    fast = cv2.FastFeatureDetector_create()
    kp_fast = fast.detect(img_source, None)
    fastRes = img_source.copy()
    cv2.drawKeypoints(img_source, kp_fast, fastRes, [0,255,0])
    fastRes = cv2.cvtColor(fastRes, cv2.COLOR_BGR2RGB)

    #ORB
    orb = cv2.ORB_create()
    kp_orb, desc_orb = orb.detectAndCompute(img_source, None)
    orbRes = img_source.copy()
    cv2.drawKeypoints(img_source, kp_orb, orbRes, [255,0,0])
    orbRes = cv2.cvtColor(orbRes, cv2.COLOR_BGR2RGB)
    
    #SIFT
    sift = cv2.SIFT_create()
    kp_target, desc_target = sift.detectAndCompute(img_target, None)
    kp_source, desc_source = sift.detectAndCompute(img_source, None)
    siftRes = img_source.copy()
    cv2.drawKeypoints(img_source, kp_source, siftRes, [255,0,0])
    siftRes = cv2.cvtColor(siftRes, cv2.COLOR_BGR2RGB)

    #AKAZE
    akaze = cv2.AKAZE_create()
    kp_akaze, desc_akaze = akaze.detectAndCompute(img_source, None)
    akazeRes = img_source.copy()
    cv2.drawKeypoints(img_source, kp_akaze, akazeRes, [255,0,0])
    akazeRes = cv2.cvtColor(akazeRes, cv2.COLOR_BGR2RGB)

    #KAZE
    kaze = cv2.KAZE_create()
    kp_kaze, desc_kaze = kaze.detectAndCompute(img_source, None)
    kazeRes = img_source.copy()
    cv2.drawKeypoints(img_source, kp_kaze, kazeRes, [255,0,0])
    kazeRes = cv2.cvtColor(kazeRes, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(20,10))
    plt.subplot(1,1,1); plt.imshow(siftRes); plt.title("SIFT"); plt.axis(False)
    plt.show()
    






















    #kp_target, desc_target = sift/akaze/orb.detectAndCompute(img_target,None)
    #kp_source, desc_source = sift/akaze/orb.detectAndCompute(img_source,None)
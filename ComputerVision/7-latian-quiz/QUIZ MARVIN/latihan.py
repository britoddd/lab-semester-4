import numpy as np
import os
import matplotlib.pyplot as plt
import cv2

# LISTING
source_file = os.listdir('source')
target_file = os.listdir('target')

# LOAD IMAGE
img_target = cv2.imread('target/hina.png')

# SMOOTHING
gray_target = cv2.cvtColor(img_target, cv2.COLOR_BGR2GRAY)
blur_target = cv2.GaussianBlur(gray_target, (3,3),0)

for i, filename in enumerate(source_file):
    img_source = cv2.imread('source/'+filename)
    gray_source = cv2.cvtColor(img_source, cv2.COLOR_BGR2GRAY)
    blur_source = cv2.GaussianBlur(gray_source, (3,3), 0)

    # FEATURE DETECTION
    # HARRIS
    gray = np.float32(blur_source)
    harris = cv2.cornerHarris(gray, 2,3,0.04)
    harrisRes = img_source.copy()
    harrisRes[harris<0.01*harris.max()] = [0,255,0]
    harrisRes = cv2.cvtColor(harrisRes, cv2.COLOR_BGR2RGB)

    # FAST
    fast = cv2.FastFeatureDetector_create()
    kp_fast = fast.detect(img_source, None)
    fastRes = img_source.copy()
    cv2.drawKeypoints(img_source, kp_fast, fastRes, [0,255,0])
    fastRes = cv2.cvtColor(fastRes, cv2.COLOR_BGR2RGB)

    # ORB
    orb = cv2.ORB_create()
    kp_orb, desc_orb = orb.detectAndCompute(img_source, None)
    orbRes = img_source.copy()
    cv2.drawKeypoints(img_source, kp_orb, orbRes, [255,0,0])
    orbRes = cv2.cvtColor(orbRes, cv2.COLOR_BGR2RGB)

    # SIFT
    sift = cv2.SIFT_create()
    kp_target, desc_target = sift.detectAndCompute(img_target, None)
    kp_source, desc_source = sift.detectAndCompute(img_source, None)
    siftRes = img_source.copy()
    cv2.drawKeypoints(img_source, kp_source, siftRes, [255,0,0])
    siftRes = cv2.cvtColor(siftRes, cv2.COLOR_BGR2RGB)

    # AKAZE
    akaze = cv2.AKAZE_create()
    kp_akaze, desc_akaze = akaze.detectAndCompute(img_source, None)
    akazeRes = img_source.copy()
    cv2.drawKeypoints(img_source, kp_akaze, akazeRes, [255,0,0])
    akazeRes = cv2.cvtColor(akazeRes, cv2.COLOR_BGR2RGB)

    # KAZE
    kaze = cv2.KAZE_create()
    kp_kaze, desc_kaze = kaze.detectAndCompute(img_source, None)
    kazeRes = img_source.copy()
    cv2.drawKeypoints(img_source, kp_kaze, kazeRes, [255,0,0])
    kazeRes = cv2.cvtColor(kazeRes, cv2.COLOR_BGR2RGB)

    # FEATURE MATCHING
    # FLANN
    FLANN_INDEX = 1
    index_params = dict(algorithm = FLANN_INDEX, trees = 5)
    search_params = dict(checks = 50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches_flann = flann.knnMatch(desc_target, desc_source, k=2)

    good_flann = []
    matches_mask_flann = []

    for _ in range(0, len(matches_flann)):
        matches_mask_flann.append([0,0])
    
    for i, (m,n) in enumerate(matches_flann):
        if m.distance < 0.7*n.distance:
            good_flann.append([m])
            matches_mask_flann[i] = [1,0]
    
    flannResult = cv2.drawMatchesKnn(
        img_target, kp_target,
        img_source, kp_source,
        good_flann, None,
        matchesMask = matches_mask_flann,
        matchColor = [0,255,0],
        singlePointColor = [255,0,0]
    )
    
    flannResult = cv2.cvtColor(flannResult, cv2.COLOR_BGR2RGB)


    # BRUTE FORCE
    bf = cv2.BFMatcher()
    matches_bf = bf.knnMatch(desc_target, desc_source, k=2)

    good_bf = []
    matches_mask_bf = []

    for _ in range(0, len(matches_mask_bf)):
        matches_mask_bf.append([0,0])
    
    for m,n in matches_bf:
        if m.distance < 0.7*n.distance:
            good_bf.append([m])
            matches_mask_bf[i] = [1,0]
    
    bfResult = cv2.drawMatchesKnn(
        img_target, kp_target,
        img_source, kp_source,
        good_bf, None,
        matchesMask = matches_mask_bf,
        matchColor = [0,255,0],
        singlePointColor = [255,0,0]
    )
    
    bfResult = cv2.cvtColor(bfResult, cv2.COLOR_BGR2RGB)

    # PRINT

    plt.figure(figsize = (15, 10)); plt.subplot(1,1,1); plt.imshow(flannResult); plt.title("Flann : "); plt.axis(False)
    plt.show()
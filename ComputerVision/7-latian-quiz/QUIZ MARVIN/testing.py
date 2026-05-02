import numpy as np
import os
import matplotlib.pyplot as plt
import cv2

# LISTING
source_file = os.listdir('source')

# LOAD IMAGE
img_target = cv2.imread('target/hina.png')

# VALIDASI
if img_target is None:
    print("Target image tidak ditemukan!")
    exit()

# PREPROCESS TARGET
gray_target = cv2.cvtColor(img_target, cv2.COLOR_BGR2GRAY)
blur_target = cv2.GaussianBlur(gray_target, (3,3), 0)

# SIFT TARGET (sekali saja, lebih efisien)
sift = cv2.SIFT_create()
kp_target, desc_target = sift.detectAndCompute(img_target, None)

# CEK descriptor
if desc_target is None:
    print("Descriptor target kosong!")
    exit()

# WAJIB untuk FLANN
desc_target = np.float32(desc_target)

for filename in source_file:
    img_source = cv2.imread('source/' + filename)

    if img_source is None:
        continue

    gray_source = cv2.cvtColor(img_source, cv2.COLOR_BGR2GRAY)
    blur_source = cv2.GaussianBlur(gray_source, (3,3), 0)

    # ========================
    # FEATURE DETECTION
    # ========================

    # HARRIS
    gray = np.float32(blur_source)
    harris = cv2.cornerHarris(gray, 2, 3, 0.04)
    harrisRes = img_source.copy()
    harrisRes[harris > 0.01 * harris.max()] = [0,255,0]

    # FAST
    fast = cv2.FastFeatureDetector_create()
    kp_fast = fast.detect(img_source, None)

    # ORB
    orb = cv2.ORB_create()
    kp_orb, desc_orb = orb.detectAndCompute(img_source, None)

    # SIFT SOURCE
    kp_source, desc_source = sift.detectAndCompute(img_source, None)

    # CEK descriptor
    if desc_source is None:
        continue

    desc_source = np.float32(desc_source)

    # ========================
    # FLANN MATCHING
    # ========================
    FLANN_INDEX = 1
    index_params = dict(algorithm=FLANN_INDEX, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches_flann = flann.knnMatch(desc_target, desc_source, k=2)

    good_flann = []
    matches_mask_flann = [[0,0] for _ in range(len(matches_flann))]

    for idx, (m, n) in enumerate(matches_flann):
        if m.distance < 0.7 * n.distance:
            good_flann.append([m])
            matches_mask_flann[idx] = [1,0]

    flannResult = cv2.drawMatchesKnn(
        img_target, kp_target,
        img_source, kp_source,
        matches_flann, None,
        matchesMask=matches_mask_flann,
        matchColor=[0,255,0],
        singlePointColor=[255,0,0]
    )

    flannResult = cv2.cvtColor(flannResult, cv2.COLOR_BGR2RGB)

    # ========================
    # BRUTE FORCE MATCHING
    # ========================
    bf = cv2.BFMatcher()
    matches_bf = bf.knnMatch(desc_target, desc_source, k=2)

    good_bf = []
    matches_mask_bf = [[0,0] for _ in range(len(matches_bf))]

    for idx, (m, n) in enumerate(matches_bf):
        if m.distance < 0.7 * n.distance:
            good_bf.append([m])
            matches_mask_bf[idx] = [1,0]

    bfResult = cv2.drawMatchesKnn(
        img_target, kp_target,
        img_source, kp_source,
        matches_bf, None,
        matchesMask=matches_mask_bf,
        matchColor=[0,255,0],
        singlePointColor=[255,0,0]
    )

    bfResult = cv2.cvtColor(bfResult, cv2.COLOR_BGR2RGB)

    # ========================
    # DISPLAY
    # ========================
    plt.figure(figsize=(15,5))

    plt.subplot(1,2,1)
    plt.imshow(flannResult)
    plt.title("FLANN")
    plt.axis(False)

    plt.subplot(1,2,2)
    plt.imshow(bfResult)
    plt.title("Brute Force")
    plt.axis(False)

    plt.show()
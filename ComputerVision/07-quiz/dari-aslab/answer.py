import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

best_img = None
best_name = ""
best_score = 0

img = cv2.imread('images/target/hina.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
hina = cv2.GaussianBlur(gray, (3,3), 0)
hina_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

akaze = cv2.AKAZE.create()
kp_hina, desc_hina = akaze.detectAndCompute(hina, None)
desc_hina = np.float32(desc_hina)

path = 'images/source'
for filename in os.listdir(path):
    studentPath = os.path.join(path, filename)
    studentImg = cv2.imread(studentPath)
    studentGray = cv2.cvtColor(studentImg, cv2.COLOR_BGR2GRAY)
    studentBlur = cv2.GaussianBlur(studentGray, (3,3), 0)
    studentRgb = cv2.cvtColor(studentImg, cv2.COLOR_BGR2RGB)

    kp_student, desc_student = akaze.detectAndCompute(studentBlur, None)
    desc_student = np.float32(desc_student)

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)
    score = 0
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(desc_hina, desc_student, k=2)
    matches_mask = []
    for _ in range(0, len(matches)):
        matches_mask.append([0,0])
    for i, (m,n) in enumerate(matches):
        if m.distance < n.distance * 0.7:
            matches_mask[i] = [1,0]
            score += 1
    
    if score > best_score:
        best_name = filename
        best_score = score
        best_img = cv2.drawMatchesKnn(hina_rgb, kp_hina,
                                      studentRgb, kp_student,
                                      matches, None,
                                      matchColor=[0, 255, 0], singlePointColor=[255, 0, 0],
                                      matchesMask = matches_mask)

plt.imshow(best_img)
plt.title(f"Best Match: {best_name} | score: {best_score}")
plt.axis(False)
plt.show()
    

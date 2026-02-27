import cv2
import numpy as np
from matplotlib import pyplot as plt

ori_img = cv2.imread(
    "ComputerVision/2-imgResize-shape-cvt-density-histogram-equalization/apple.png"
)
cv2.imshow("Original", ori_img)
cv2.waitKey(0)

print(ori_img.shape)

cv2.resize(ori_img, (200, 300))

# Resize
scale_percent = 50 / 100
new_h = int(ori_img.shape[0] * scale_percent)
new_w = int(ori_img.shape[1] * scale_percent)
resized_img = cv2.resize(ori_img, (new_w, new_h))
print(f"Resized image shape: {resized_img.shape}")
cv2.imshow("Resized", resized_img)
cv2.waitKey(0)

# Convert to grayscale
gray_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
print(f"Gray image shape: {gray_img.shape}")
cv2.imshow("Gray", gray_img)
cv2.waitKey(0)

# Count pixel density
h = gray_img.shape[0]
w = gray_img.shape[1]
intensity_counter = np.zeros(256, dtype=int)

for i in range(h):
    for j in range(w):
        intensity_counter[gray_img[i][j]] += 1

# plt.figure()
# plt.plot(intensity_counter, 'g', label='Apple')
# plt.xlabel('Quantity')
# plt.ylabel('Intensity')
# plt.show()

# Histogram Equalization -> biar lebih rata histogramnya
equ = cv2.equalizeHist(gray_img)

res = np.hstack((gray_img, equ))
cv2.imshow("Result", res)
cv2.waitKey(0)

h = equ.shape[0]
w = equ.shape[1]
equ_counter = np.zeros(256, dtype=int)

for i in range(h):
    for j in range(w):
        equ_counter[equ[i][j]] += 1

# Plotting before
plt.figure(1, (16, 8))
plt.subplot(1, 2, 1)
plt.plot(intensity_counter, "g", label="Before")
plt.legend(loc="upper right")
plt.xlabel("Quantity")
plt.ylabel("Intensity")

# Plotting after
plt.subplot(1, 2, 2)
plt.plot(equ_counter, "r", label="After")
plt.legend(loc="upper left")
plt.xlabel("Quantity")
plt.ylabel("Intensity")
plt.show()

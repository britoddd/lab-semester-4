import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('images/images.jpg')

plt.figure(1, figsize=(8,8))
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # sebelum diproses, diubah jadi grayscale dulu biar enteng

# Thresholding
_, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)  # 100, 255 -> kurang dari 100 bakal jadi 0, lebih jadi 255
_, binaryinv = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)  # sama aja kayak atas, tapi di invert
_, trunc = cv2.threshold(gray, 100, 255, cv2.THRESH_TRUNC)  # 
_, tozero = cv2.threshold(gray, 100, 255, cv2.THRESH_TOZERO)  # 
_, tozeroinv = cv2.threshold(gray, 100, 255, cv2.THRESH_TOZERO_INV)  # 
_, otsu = cv2.threshold(gray, 100, 255, cv2.THRESH_OTSU)  # dia bakal bikin threshold sendiri??
# print(_)

adaptiveThreshold = cv2.adaptiveThreshold(
    gray,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    11,
    2
)

img_array = [gray, binary, binaryinv, tozero, trunc, tozeroinv, otsu, adaptiveThreshold]
titles = ["Gray", "Binary", "Binary Inverse", "To Zero", "Trunc", "To Zero Inverse", "Otsu", "Adaptive"]

# for i, (image, titles) in enumerate(zip(img_array, titles)):  # ini harus di-zip biar dianggep sebagai kesatuan
#     plt.subplot(3, 3, i+1)
#     plt.imshow(image, 'gray')
#     plt.title(titles)
#     plt.axis(False)
    
# plt.show()


# Filtering
def manualBlur(img, kernel_size):
    offset = kernel_size - 1
    
    img_arr = np.array(img)
    temp = img_arr.copy()
    for i in range(img_arr.shape[0] - offset):
        for j in range(img_arr.shape[1] - offset):
            arr = img_arr[i:(i + kernel_size), j:(j + kernel_size)]
            mean = np.mean(arr)
            temp[i + kernel_size // 2, j + kernel_size // 2] = mean
    return temp

meanBlur = cv2.blur(gray, (10,10))
medianBlur = cv2.medianBlur(gray, 11)
gaussianBlur = cv2.GaussianBlur(gray, (11,11), 5.0)
bilaterBlur = cv2.bilateralFilter(gray, 11, 5.0, 5.0)
manualBlur = manualBlur(gray, 11)

img_array = [gray, meanBlur, medianBlur, gaussianBlur, bilaterBlur, manualBlur]
titles = ["Normal", "Mean", "Median", "Gaussian", "Bilateral", "Manual"]

for i, (image, titles) in enumerate(zip(img_array, titles)):
    plt.subplot(3, 2, i+1)
    plt.imshow(image, 'gray')
    plt.title(titles)
    plt.axis(False)

# plt.imshow(bilaterBlur, 'gray')
# plt.title("Blur")
# plt.axis(False)
plt.show()
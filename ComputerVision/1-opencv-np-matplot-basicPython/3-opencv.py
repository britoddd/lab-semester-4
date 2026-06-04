import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("./images/images.jpg")
print(img.size)
print(img.ndim)
print("Ukuran gambar: ", img.shape[0], "x", img.shape[1], "px")

cv2.imshow('Images', img)
cv2.imwrite('Second Pict.jpg', img)
cv2.waitKey(0)

# Matplotlib
plt.figure(figsize=(10,10))
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert image color to RGB
plt.imshow(img_rgb)
plt.title("Image")
plt.show()
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.vq import *
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler

# Training
train_path = 'Dataset/Dataset/Train/'
train_path_list = os.listdir(train_path)

labels = train_path_list
print(labels)

image_list = []
image_class_id = []

for idx, class_path in enumerate(train_path_list):
    image_path_list = os.listdir(train_path + train_path_list[idx])
    
    for image_path in image_path_list:
        image_list.append(train_path + class_path + '/' + image_path)
        image_class_id.append(idx)
    
sift = cv2.SIFT.create()

descriptor_list = []
for image in image_list:
    _, descriptor = sift.detectAndCompute(cv2.imread(image), None)
    descriptor_list.append(descriptor)

print(descriptor_list)

# Preprocessing
stacked_descriptor = descriptor_list[0]
for descriptor in descriptor_list[1:]:
    stacked_descriptor = np.vstack((stacked_descriptor, descriptor))
    stacked_descriptor = np.float32(stacked_descriptor)

# Clustering, K-Means
centroids, _ = kmeans(stacked_descriptor, 5, 15)
image_features = np.zeros((len(image_list), len(centroids)), "float32")

# Vector Quantization
for i in range(len(image_list)):
    words, _ = vq(descriptor_list[i], centroids)
    
    for w in words:
        image_features[i][w] += 1

print(image_features)

std_scaler = StandardScaler().fit(image_features)
image_features = std_scaler.transform(image_features)

print(image_features)

# Classifying
svc = LinearSVC()
svc.fit(image_features, np.array(image_class_id))

# Testing
test_path = 'Dataset/Dataset/Test/'
test_path_list = os.listdir(test_path)

image_list = []
for image in test_path_list:
    test_image = test_path + image
    image_list.append(test_image)
    
print(image_list)

# Extract Features
descriptor_list = []
for image in image_list:
    _, descriptor = sift.detectAndCompute(cv2.imread(image), None)
    descriptor_list.append(descriptor)
    
test_features = np.zeros((len(image_list), len(centroids)), "float32")
for i in range(len(image_list)):
    words, _ = vq(descriptor_list[i], centroids)
    
    for w in words: 
        test_features[i][w] += 1
        
std_scaler = StandardScaler().fit(test_features)
test_features = std_scaler.transform(test_features)

result = svc.predict(test_features)
print(result)

# Show result
for idx, (class_id, image) in enumerate(zip(result, image_list)):
    plt.subplot(2, 3, idx + 1)
    plt.title(labels[class_id])
    plt.imshow(cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB))
    plt.yticks([])    
    plt.xticks([])    

plt.show()
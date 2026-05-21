import cv2
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
import os

dir = 'images'

def process(img_path):
    img = cv2.imread(img_path)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hist = cv2.calcHist(
        [rgb], [0, 1, 2], None, [8, 8, 8],
        [0,256, 0,256, 0,256]
    )
    normalize = cv2.normalize(hist, None).flatten()
    return normalize

features = []
image_paths = {}

for filename in os.listdir(dir):
    if ".db" in filename: continue
    file_path = os.path.join(dir, filename)
    img_name = filename.split('.')[0]
    data = process(file_path)
    image_paths[img_name] = file_path
    features.append((img_name, data))
    
target = process('images/Shire-001.png')

res = []

for name, data in features:
    dis = euclidean(target, data)
    res.append((dis, name))

sort = sorted(res)

top = 3

target_img = cv2.cvtColor(cv2.imread('images/Shire-001.png'), cv2.COLOR_BGR2RGB)
plt.figure(figsize=(12,4))
plt.subplot(1, top+1, 1)
plt.imshow(target_img)
plt.title('Target')
plt.axis(False)

for i in range(top):
    dis, name = sort[i]
    path = image_paths[name]
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    plt.subplot(1, top+1, i+2)
    plt.imshow(img)
    plt.title(f"{name}, Distance: {dis:.2f}")
    plt.axis(False)
    
plt.tight_layout()
plt.show()
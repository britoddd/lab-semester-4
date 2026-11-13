import os, cv2
import matplotlib.pyplot as plt
import numpy as np

dir = 'images/train'
classes = os.listdir(dir)

face_recognizer = cv2.face.LBPHFaceRecognizer().create()
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

face_list, class_list = [], []
label_map = {index:name for index, name in enumerate(classes)}

def face(full_path):
  img = cv2.imread(full_path)
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  face_detection = face_detector.detectMultiScale(
    gray, scaleFactor=1.2, minNeighbors=5
  )
  return img, gray, face_detection

def preprocess(img):
  resize = cv2.resize(img, (100,100))
  normalize = cv2.equalizeHist(resize)
  return normalize

def train_test_accuracy():
  # Siapin data training
  for index, name in enumerate(classes):
    path = os.path.join(dir, name)
    for img_path in os.listdir(path):
      if img_path in "Thumbs.db":
        continue
      full_path = os.path.join(path, img_path)
      _, gray, data = face(full_path)
      for rect in data:
        x, y, w, h = rect
        face_list.append(preprocess(gray[y:y+h, x:x+w]))
        class_list.append(index)

  # Training model
  face_recognizer.train(face_list, np.array(class_list))

  # Siapin data testing
  test_dir = 'images/test'
  
  # Kalkulasi hasil testing
  total = 0
  correct = 0
  for path in os.listdir(test_dir):
    if path in "Thumbs.db":
      continue
    full_path = os.path.join(test_dir, path)
    _, gray, data = face(full_path)
    for rect in data:
      x, y, w, h = rect
      res, dis = face_recognizer.predict(preprocess(gray[y:y+h, x:x+w]))
      predicted_name = label_map.get(res, "Don't know")
      actual_name = os.path.splitext(path)[0]

      total += 1
      if predicted_name == actual_name:
        correct += 1
        print(f"P: {predicted_name} D: {dis} A: {actual_name}")
  accuracy = correct / total * 100
  print(f"Accuracy: {accuracy}%")

def predict(path):
  img, gray, data = face(path)

  for rect in data:
    x, y, w, h = rect
    res, dis = face_recognizer.predict(preprocess(gray[y:y+h, x:x+w]))
    cv2.rectangle(img, (x,y), (x+w, y+h), [0,255,0], 1)
    text = f"{classes[res]}, dis: {dis}"
    cv2.putText(img, text, (x,y-10), 1, 1.5, [0,255,0], 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis(False)
    plt.title(classes[res])
    plt.show()

while True:
    print("=" * 25)
    print("Select Your Choice:")
    print("1. Train model and see accuracy")
    print("2. Predict")
    print("3. Exit")
    choice = input("Enter your choice: ")
    
    if choice == "1":
        train_test_accuracy()
    elif choice == "2":
        path = input("Enter Image Path: ")
        predict(path)
    elif choice == "3":
        break
    else:
        print("Your choice is not valid")
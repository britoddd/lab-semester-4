import os, cv2
import numpy as np
import matplotlib.pyplot as plt
model_dir = './model.yml'
train_dir = 'Dataset/train'
test_dir = 'Dataset/test'
classes = sorted(os.listdir(train_dir))

label_map = {index: name for index, name in enumerate(classes)}
face_detection = cv2.CascadeClassifier(
  cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

def preprocess(img):
  # img is already a grayscale face crop; just enforce a consistent size
  return cv2.resize(img, (92, 92))

def face(full_path):
  img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
  detected = face_detection.detectMultiScale(img, 1.2, 5)
  # The AT&T/ORL images are already tightly cropped faces, so the detector
  # often finds nothing. Fall back to the whole image as the face region.
  return img, detected

def train_model():
  face_list, class_list = [], []

  for index, name in enumerate(classes):
    path = os.path.join(train_dir, name)
    for img_path in os.listdir(path):
      full_path = os.path.join(path, img_path)
      gray, data = face(full_path)
      for (x, y, w, h) in data:
        face_list.append(preprocess(gray[y:y+h, x:x+w]))
        class_list.append(index)

  face_recognizer = cv2.face.LBPHFaceRecognizer_create()
  face_recognizer.train(face_list, np.array(class_list))
  face_recognizer.save(model_dir)

  total = 0
  correct = 0

  for index, name in enumerate(classes):
    path = os.path.join(test_dir, name)
    for img_path in os.listdir(path):
      full_path = os.path.join(path, img_path)
      gray, data = face(full_path)
      for (x, y, w, h) in data:
        total += 1
        res, dis = face_recognizer.predict(preprocess(gray[y:y+h, x:x+w]))
        actual_name = name
        predict_name = label_map.get(res, "Don't know")
        if actual_name == predict_name:
          correct += 1
          print(f"P: {predict_name} A: {actual_name} D: {dis}")

  accuracy = correct / total * 100 if total else 0
  print(f"Average Accuracy: {accuracy:.2f}%")

def predict():
  if not os.path.exists(model_dir):
    print("Model not found. Please train the model first (menu 1).")
    return

  face_recognizer = cv2.face.LBPHFaceRecognizer_create()
  face_recognizer.read(model_dir)

  path = input("Enter the path to the image for testing (absolute path): ").strip()
  if not os.path.exists(path):
    print("Image not found.")
    return

  gray, data = face(path)
  img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
  for (x, y, w, h) in data:
    res, dis = face_recognizer.predict(preprocess(gray[y:y+h, x:x+w]))
    predict_name = label_map.get(res, "Don't know")
    print(f"{path} detected as {predict_name}")

    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 1)
    text = f"{predict_name} {dis:.2f}"
    cv2.putText(img, text, (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)

  plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
  plt.title("Predict Result")
  plt.axis('off')
  plt.show()

def main():
  while True:
    print("\n1. Train and test model.")
    print("2. Predict.")
    print("3. Exit")
    choice = input("Enter your choice: ").strip()

    if choice == '1':
      train_model()
    elif choice == '2':
      predict()
    elif choice == '3':
      print("Goodbye.")
      break
    else:
      print("Invalid choice.")

if __name__ == '__main__':
  main()

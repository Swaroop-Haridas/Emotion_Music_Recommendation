import os
import cv2
import numpy as np
import random

data_dir = "train"
model_path = "model.yml"
label_map_path = "labels.npy"

emotions = ["angry", "disgusted", "fearful", "happy", "neutral", "sad", "surprised"]

faces = []
labels = []
label_dict = {}

current_label = 0

for emotion in emotions:
    folder_path = os.path.join(data_dir, emotion)
    if not os.path.isdir(folder_path):
        continue

    label_dict[current_label] = emotion
    image_files = os.listdir(folder_path)
    random.shuffle(image_files)
    image_files = image_files[:300] 

    for img_file in image_files:
        img_path = os.path.join(folder_path, img_file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, (100, 100))
        faces.append(img)
        labels.append(current_label)

    current_label += 1

print(f"✅ Loaded {len(faces)} samples. Training model...")

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(faces, np.array(labels))

recognizer.save(model_path)
np.save(label_map_path, label_dict)

print("✅ Training complete. Model and labels saved.")

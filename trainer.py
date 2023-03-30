import cv2
import os

# Set up face recognition classifier
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Collect face images and labels for training
face_samples = []
labels = []
for label, folder_name in enumerate(os.listdir('faces')):
    folder_path = os.path.join('faces', folder_name)
    for filename in os.listdir(folder_path):
        image_path = os.path.join(folder_path, filename)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        face_samples.append(image)
        labels.append(label)

# Train face recognition classifier
recognizer.train(face_samples, np.array(labels))

# Save trained model to file
recognizer.write('trainer.yml')

import cv2
import requests
import json
import tempfile
import os

# Load face detection classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# API endpoint for getting image list
api_endpoint = "http://127.0.0.1:8001/user-list"

# API endpoint for attendance
attendance_api_endpoint = "http://127.0.0.1:8001/attendance"

# Function to download image from URL
def download_image(url):
    response = requests.get(url)
    file = tempfile.NamedTemporaryFile(delete=False)
    file.write(response.content)
    file.close()
    return file.name

# Function to get image list from API
def get_image_list():
    response = requests.get(api_endpoint)
    data = json.loads(response.text)
    image_list = [download_image(image_url) for image_url in data]
    return image_list

# Function to recognize face and call attendance API
def recognize_face():
    # Open camera
    cap = cv2.VideoCapture(0)

    while True:
        # Capture frame from camera
        ret, frame = cap.read()

        # Convert frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in grayscale frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        # Loop through detected faces
        for (x, y, w, h) in faces:
            # Draw rectangle around detected face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Crop face region from frame
            face_roi = gray[y:y+h, x:x+w]

            # Call attendance API if face matches with image list
            if is_matching_face(face_roi):
                response = requests.post(attendance_api_endpoint, data={"status": "present"})
                print(response.text)
                # Release camera and return from function
                cap.release()
                cv2.destroyAllWindows()
                return

        # Display frame with detected faces
        cv2.imshow('frame', frame)

        # Check for key press and exit if 'q' is pressed
        if cv2.waitKey(1) == ord('q'):
            break

    # Release camera
    cap.release()
    # Destroy OpenCV windows
    cv2.destroyAllWindows()

# Function to check if face matches with image list
def is_matching_face(face_roi):
    # Get image list from API
    image_list = get_image_list()

    # Load face detection classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Load face recognition classifier
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('trainer.yml')

    # Loop through image list and try to match face
    for image_path in image_list:
        # Load image from file
        img = cv2.imread(image_path)

        # Convert image to grayscale for face recognition
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect face in grayscale image
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        # If face is detected, try to match with face_roi
        for (x, y, w, h) in faces:
            # Crop face region from image
            roi_gray = gray[y:y+h, x:x+w]

            # Recognize face using face recognition classifier
            label, confidence = recognizer.predict(roi_gray)

            # If confidence is below threshold, return True
            if confidence < 50:
                # Delete temporary image files
                os.remove(image_path)
                return True

    # If no matching face found, create log for visitor and call API
    response = requests.post(api_endpoint, data={"visitor": "unknown"})
    print(response.text)
    return False

# Call recognize_face function to start face
recognize_face()
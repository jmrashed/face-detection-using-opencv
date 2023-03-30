import cv2

# Load the cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# set array of faces
faces = [
    'assets/images/rashed.jpg',
    'assets/images/faysal.jpg',
    'assets/images/abrar.jpg',
    'assets/images/rony.jpg',
    'assets/images/saiful.jpg',
]

# create files for storing results
found_file = open("found.txt", "a")
not_found_file = open("notfound.txt", "a")

# To capture video from webcam. 
cap = cv2.VideoCapture(0)
# To use a video file as input 
# cap = cv2.VideoCapture('filename.mp4')

while True:
    # Read the frame
    _, img = cap.read()

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect the faces
    faces_detected = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Check if the faces in the array exist in the detected faces
    for face in faces:
        face_img = cv2.imread(face)
        if face_img is not None:
            face_gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            _, face_thresh = cv2.threshold(face_gray, 30, 255, cv2.THRESH_BINARY)
            for (x, y, w, h) in faces_detected:
                face_roi = gray[y:y+h, x:x+w]
                _, face_roi_thresh = cv2.threshold(face_roi, 30, 255, cv2.THRESH_BINARY)
                res = cv2.matchTemplate(face_roi_thresh, face_thresh, cv2.TM_CCOEFF_NORMED)
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
              
            else:
                not_found_file.write(face + "\n")
                cv2.rectangle(img, (0, 0), (img.shape[1], img.shape[0]), (0, 0, 255), 2)
        else:
            not_found_file.write(face + "\n")
            cv2.rectangle(img, (0, 0), (img.shape[1], img.shape[0]), (0, 0, 255), 2)

    # Display
    cv2.imshow('img', img)

    # Stop if escape key is pressed
    k = cv2.waitKey(30) & 0xff
    if k==27:
        break

# close files
found_file.close()
not_found_file.close()

# Release the VideoCapture object
cap.release()

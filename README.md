# Face Detection using OpenCV
This project is a simple face detection program using OpenCV in Python. It detects faces in real-time video stream from a webcam or a video file and marks the faces with green rectangles if they match the faces in a pre-defined list, and red rectangles if they don't match.

## Getting Started
To get started with the project, you need to have OpenCV installed on your system. You can install OpenCV using pip:
`pip install opencv-python`
You also need to download the haarcascade_frontalface_default.xml file and save it in the same directory as the Python script. This file contains the pre-trained classifier for detecting frontal faces.

## Usage
To run the program, execute the following command in the terminal:
`python face_detection.py`

The program will open a video stream from your default webcam and start detecting faces in real-time. If you want to use a video file instead, you can modify the cap variable in the code:

`cap = cv2.VideoCapture('path/to/video/file.mp4')`

You can also modify the faces list to match the `faces` you want to detect. The list should contain the file paths of the images of the faces you want to detect.

# Output
When the program detects a face that matches one of the faces in the faces list, it will mark the face with a green rectangle:

<img src="output/Screenshot 2023-03-30 at 12.13.58 PM.png">

If the program detects a face that does not match any of the faces in the faces list, it will mark the face with a red rectangle:

<img src="output/Screenshot 2023-03-30 at 12.13.58 PM.png">

The program also writes the file paths of the faces it detects to two text files: found.txt and notfound.txt. The found.txt file contains the file paths of the faces that were detected, while the notfound.txt file contains the file paths of the faces that were not detected.


# Credits
The `haarcascade_frontalface_default.xml` file used in this project is part of the `OpenCV` library and was created by Rainer Lienhart and Jochen Maydt. The file can be found in the OpenCV GitHub repository.


## License

This project is licensed under the terms of the [MIT](https://choosealicense.com/licenses/mit/) license.


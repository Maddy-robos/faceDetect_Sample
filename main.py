import cv2
import dlib
from facenet_pytorch import MTCNN

# Load the image
img = cv2.imread('face_3.png')

# Create an instance of the MTCNN model
mtcnn = MTCNN()

# Create a dlib detector for face detection
detector = dlib.get_frontal_face_detector()

# Create a dlib predictor for landmark detection
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Detect faces in the image
boxes, _ = mtcnn.detect(img)

print(boxes)

if len(boxes) > 1:
    print("Cannot have more than one face")
else:

    # Iterate through each detected face
    for box in boxes:
        # Get the coordinates of the bounding box
        top_x, top_y, bot_x, bot_y = box.astype(int)

        # Extract the face ROI from the image
        face = img[top_y:bot_y, top_x:bot_x]

        cv2.imshow('ROI', face)
        cv2.waitKey(0)

        # draw bounding box around the face
        cv2.rectangle(img, (top_x, top_y), (bot_x, bot_y), (0, 255, 0), 2)

        # Convert the face ROI to grayscale for landmark detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Use the dlib detector to detect faces in the grayscale image
        rects = detector(gray, 0)
        print(rects)

        # Iterate through each detected face in the grayscale image
        for rect in rects:
            # Use the dlib predictor to get the facial landmarks for the current face
            landmarks = predictor(gray, rect)

            # Iterate through each landmark and draw a circle around it
            for i in range(68):
                x = landmarks.part(i).x
                y = landmarks.part(i).y
                cv2.circle(img, (int(x), int(y)), 2, (0, 255, 0), -1)
                # cv2.circle(img, (int(x) + rect[0], int(y) + rect[1]), 2, (0, 255, 0), -1)

    # Show the image with facial landmarks
    cv2.imshow('Facial Landmarks', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

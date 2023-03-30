import cv2
from mtcnn import MTCNN

# load image
img = cv2.imread("test.jpg")

# initialize MTCNN detector
detector = MTCNN()

# detect faces using MTCNN
results = detector.detect_faces(img)

# iterate over each detected face
for result in results:
    # extract bounding box coordinates
    x1, y1, width, height = result['box']
    x2, y2 = x1 + width, y1 + height

    # draw bounding box around the face
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # extract landmarks of the face
    landmarks = result['keypoints']

    # draw landmarks on the face
    for key, point in landmarks.items():
        x, y = point
        cv2.circle(img, (int(x), int(y)), 2, (0, 255, 0), -1)

# display image with bounding boxes and landmarks
cv2.imshow("Facial Landmarks", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

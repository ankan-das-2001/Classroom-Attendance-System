import cv2
from tensorflow.io.gfile import listdir

from classes.model import Model

# get class list
class_names = listdir("Datasets/train/")

# load model
model = Model(len(class_names))
model.load_model()

# predict live on camera input
cap = cv2.VideoCapture(0)
face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
for i in range(100):
    ret, frame = cap.read()
    faces = face_classifier.detectMultiScale(frame, 1.3, 5)
    if faces is ():
        preds = model.predict_on_cv(frame)
    # Crop all faces found
    else:
        cropped_faces = []
        for (x, y, w, h) in faces:
            x = x - 10
            y = y - 10
            cropped_faces.append(frame[y : y + h + 50, x : x + w + 50])
        for img in cropped_faces:
            preds = model.predict_on_cv(img)
    print(preds)
    if cv2.waitKey(1) == 13:
        break

cap.release()
cv2.destroyAllWindows()

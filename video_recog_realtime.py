import numpy as np
import cv2
from tensorflow.keras import models

index_to_label = {0: 'donaldtrump', 1: 'khacquang', 2: 'messi', 3: 'unknown'}

model = models.load_model('model_face_recog_mobilenet.h5')
face_detector = cv2.CascadeClassifier('haarcascade/haarcascade_frontalface_alt.xml')

camera = cv2.VideoCapture('gettyimages-464058554-640_adpp.mp4')

while True:
    OK, FRAME = camera.read()
    if not OK:
        continue

    faces = face_detector.detectMultiScale(FRAME, 1.2, 5)
    for (x, y, w, h) in faces:
        roi = FRAME[y:y+h, x:x+w]
        roi_resized = cv2.resize(roi, (224, 224))
        roi_norm = roi_resized / 255.0
        roi_reshaped = roi_norm.reshape(1, 224, 224, 3)

        # predict
        proba = model.predict(roi_reshaped)[0]
        max_prob = np.max(proba)
        pred_class = np.argmax(proba)

        # threshold
        if max_prob < 0.6:
            label = "unknown"
        else:
            label = index_to_label[pred_class]

        cv2.rectangle(FRAME, (x,y), (x+w,y+h), (0,255,0), 2)
        cv2.putText(FRAME, label, (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow('cam', FRAME)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()

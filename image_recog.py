import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load face detector
face_detector = cv2.CascadeClassifier('haarcascade/haarcascade_frontalface_alt.xml')

# Load trained model
model = load_model('model_face_recog_mobilenet.h5')

# Label mapping
index_to_label = {0: 'donaldtrump', 1: 'khacquang', 2: 'messi', 3: 'unknown'}

# Load test image
image = cv2.imread("messi_family.jpg")
if image is None:
    print("❌ Không đọc được file ảnh!")
    exit()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
faces = face_detector.detectMultiScale(gray, 1.2, 5)

for (x,y,w,h) in faces:
    roi = image[y:y+h, x:x+w]
    roi_resized = cv2.resize(roi, (224,224))
    roi_norm = roi_resized / 255.0
    roi_reshaped = roi_norm.reshape(1, 224, 224, 3)

    proba = model.predict(roi_reshaped)[0]
    max_prob = np.max(proba)
    pred_idx = np.argmax(proba)

    if max_prob < 0.6:
        label = "unknown"
    else:
        label = index_to_label[pred_idx]

    cv2.rectangle(image, (x,y), (x+w,y+h), (0,255,0), 2)
    cv2.putText(image, f"{label} ({max_prob:.2f})", (x,y-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

cv2.imshow("Face Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

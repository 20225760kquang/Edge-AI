import deeplake
import cv2
import os
import numpy as np

# load LFW từ hub deeplake
ds = deeplake.load('hub://activeloop/lfw')

# prepare folder
os.makedirs('Datasets/unknown', exist_ok=True)

# load face detector
face_cascade = cv2.CascadeClassifier('haarcascade/haarcascade_frontalface_alt.xml')

count = 0

for i in range(1000):
    # lấy image tensor
    img_tensor = ds.images[i].numpy()
    # Deeplake trả về RGB, OpenCV dùng BGR
    img = cv2.cvtColor(img_tensor, cv2.COLOR_RGB2BGR)

    # detect face
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.2, 5)

    for (x,y,w,h) in faces:
        roi = img[y:y+h, x:x+w]
        roi = cv2.resize(roi, (224,224))
        filename = f'Datasets/unknown/roi_{count}.jpg'
        cv2.imwrite(filename, roi)
        count += 1

    if i % 100 == 0:
        print(f"Processed {i} images, saved {count} faces")

print(f"✅ Done, total saved {count} faces into Datasets/unknown")

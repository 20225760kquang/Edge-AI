import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from PIL import Image
import os

face_detector = cv2.CascadeClassifier('haarcascade/haarcascade_frontalface_alt.xml')

TRAIN_DATA = 'Datasets'

x_train = []
y_train = []
x_test = []
y_test = []
index_to_label = {0: 'donaldtrump', 1: 'khacquang', 2: 'messi' , 3 : 'unknown'}

dict = {'donaldtrump' : [1,0,0,0], 'khacquang' : [0,1,0,0], 'messi' : [0,0,1,0], 'unknown' : [0,0,0,1]}


def getData(dir, lst_imgs, lst_labels):
    for whatever in os.listdir(dir):
        whatever_path = os.path.join(dir, whatever)
        for filename in os.listdir(whatever_path):
            filename_path = os.path.join(whatever_path, filename)

            label = filename_path.split('\\')[1]

            # Read the image and resize to a standard size
            img = Image.open(filename_path).resize((100, 100))  # Standardize image size
            img_array = np.array(img)
            lst_imgs.append(img_array)
            lst_labels.append(dict[label])  # Append corresponding label

    return lst_imgs, lst_labels


x_train , y_train = getData(TRAIN_DATA, x_train, y_train )

#train model
# Convert data to NumPy arrays
x_train = np.array(x_train, dtype=np.float32)  # Convert images to NumPy array
y_train = np.array(y_train, dtype=np.float32)  # Convert labels to NumPy array
x_test = np.array(x_test, dtype=np.float32)  # Ensure x_test is an empty NumPy array
y_test = np.array(y_test, dtype=np.float32)  # Ensure y_test is an empty NumPy array

#shuffle datasets
combined = list(zip(x_train, y_train))
np.random.shuffle(combined)
x_train, y_train = zip(*combined)
x_train, y_train = np.array(x_train), np.array(y_train)



# Normalize the image data to the range [0, 1]
x_train = x_train / 255.0
x_test = x_test / 255.0 if x_test.size > 0 else x_test  # If x_test is empty, no division is needed

from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train,
    test_size=0.2,
    random_state=42,
    stratify=y_train
)

model_training_first = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(100,100,3)),
    layers.MaxPooling2D((2,2)),
    layers.Dropout(0.2),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Dropout(0.2),
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Dropout(0.2),

    layers.Flatten(),
    layers.Dense(500, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    layers.Dense(200, activation='relu'),
    layers.Dense(4, activation='softmax')
])
callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True),
    ModelCheckpoint('model_face_recog_10epochs.h5', save_best_only=True)
]

#model_training_first.summary()
model_training_first.compile(optimizer='adam',
                             loss='categorical_crossentropy'
                             ,metrics=['accuracy'])
# model_training_first.fit(x_train, y_train, epochs=10,validation_data=(x_val, y_val), callbacks=callbacks)

model_training_first.save('model_face_recog_10epochs.h5')

camera = cv2.VideoCapture(0)
model = models.load_model('model_face_recog_10epochs.h5')

while True:
    count = 0
    OK , FRAME = camera.read()
    faces = face_detector.detectMultiScale(FRAME, 1.2, 5)
    for (x, y, w, h) in faces:
        roi = cv2.resize(FRAME[y:y + h, x:x + w], (100, 100))
        results = np.argmax(model.predict(roi.reshape(1, 100, 100, 3)))
        print(results)
        label = index_to_label[results]
        cv2.rectangle(FRAME, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(FRAME, label , (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        count += 1
    cv2.imshow('cam', FRAME)

    if cv2.waitKey(1) & 0xFF == ord('q') :
       break




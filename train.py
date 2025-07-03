import os
import numpy as np
import cv2
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split

# Constants
TRAIN_DATA = 'Datasets'
index_to_label = {0: 'donaldtrump', 1: 'khacquang', 2: 'messi', 3: 'unknown'}
label_dict = {
    'donaldtrump': [1,0,0,0],
    'khacquang': [0,1,0,0],
    'messi': [0,0,1,0],
    'unknown': [0,0,0,1]
}

x_data = []
y_data = []

def get_data(root_dir):
    for label_folder in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, label_folder)
        if not os.path.isdir(folder_path):
            continue

        for img_file in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_file)
            img = cv2.imread(img_path)
            if img is None:
                continue
            # resize to match MobileNetV2
            face = cv2.resize(img, (224,224))
            x_data.append(face)
            if label_folder in label_dict:
                y_data.append(label_dict[label_folder])
            else:
                # fallback
                y_data.append([0,0,0,1])

get_data(TRAIN_DATA)

# convert numpy
x_data = np.array(x_data, dtype=np.float32) / 255.0
y_data = np.array(y_data, dtype=np.float32)

# shuffle
idx = np.arange(len(x_data))
np.random.shuffle(idx)
x_data = x_data[idx]
y_data = y_data[idx]

# train/val split
x_train, x_val, y_train, y_val = train_test_split(
    x_data, y_data, test_size=0.2, random_state=42, stratify=y_data
)

# Load MobileNetV2
base_model = MobileNetV2(
    input_shape=(224,224,3),
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False  # freeze feature extractor

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(4, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True),
    ModelCheckpoint('model_face_recog_mobilenet.h5', save_best_only=True)
]

model.fit(
    x_train, y_train,
    epochs=20,
    validation_data=(x_val, y_val),
    callbacks=callbacks
)

print("✅ Model saved as model_face_recog_mobilenet.h5")

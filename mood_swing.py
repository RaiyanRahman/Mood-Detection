# mood_swing.py
# Author: Raiyan Rahman
# Date: March 02, 2020
# Description: Use the device's webcam to detect the faces present, in real
# time, and notify the mood of the faces.

import cv2
import sys
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define the constants.
DEFAULT_CASCADE_PATH = 'haarcascade_frontalface_default.xml'
DEFAULT_RUN_MODE = 'train'
TRAIN_PATH = 'data/train/'
VAL_PATH = 'data/test/'
NUM_TRAIN = 28709
NUM_VAL = 7178
BATCH_SIZE = 64
NUM_EPOCHS = 50
MOOD_DICT = {
    0: 'Angry',
    1: 'Disgusted',
    2: 'Fearful',
    3: 'Happy',
    4: 'Neutral',
    5: 'Sad',
    6: 'Surprised'
}


# Get the arguments for the path to the cascade and the running mode.
mode = DEFAULT_RUN_MODE
cascade_path = DEFAULT_CASCADE_PATH
mode_flag = False
path_flag = False

for arg in sys.argv:
    if mode_flag:
        mode = arg
    elif path_flag:
        cascade_path = arg
    else:
        if arg == '-p':
            path_flag = True
        elif arg == '-m':
            mode_flag = True

# If no mode is given, exit.
if not mode_flag:
    print("Please enter a valid mode flag using '-m' before the mode.")
    exit(0)

# Create the cascade.
face_detection_cascade = cv2.CascadeClassifier(cascade_path)


# Define the model.
# Create the image data generators for the train and test datasets.
train_gen = ImageDataGenerator(rescale=1./255)
val_gen = ImageDataGenerator(rescale=1./255)

train_generator = train_gen.flow_from_directory(
    TRAIN_PATH,
    target_size=(48, 48),
    batch_size=BATCH_SIZE,
    color_mode="grayscale",
    class_mode='categorical'
)
val_generator = val_gen.flow_from_directory(
    VAL_PATH,
    target_size=(48, 48),
    batch_size=BATCH_SIZE,
    color_mode="grayscale",
    class_mode='categorical'
)

# Create the model using sequential layers.
model = Sequential()
# Layers obtained from https://github.com/isseu/emotion-recognition-neural-networks
# and their associated research paper.
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))


# Run the training.
if mode == 'train':
    # Train the model.
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001, decay=1e-6), metrics=['accuracy'])

    model_info = model.fit_generator(
        train_generator,
        steps_per_epoch=NUM_TRAIN // BATCH_SIZE,
        epochs=NUM_EPOCHS,
        validation_data=val_generator,
        validation_steps=NUM_VAL // BATCH_SIZE
    )
    # Save the model for later use.
    model.save_weights('model.h5')

# Display the emotions via the video feed.
elif mode == 'display':
    model.load_weights('model.h5')
    # Open the default webcam using openCV.
    webcam = cv2.VideoCapture(0)

    # Use an infinite loop to keep using the webcam.
    while True:
        # Get the current frame from the webcam.
        ret, frame = webcam.read()
        # If no frame is received from the video feed, exit.
        if not ret:
            break

        # Search for faces in the captured frame.
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)    # Convert to grayscale

        detected_faces = face_detection_cascade.detectMultiScale(
            gray_frame,
            scaleFactor=1.25,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        # Go through the detected faces, identify them, and label their mood.
        for (x, y, width, height) in detected_faces:
            # Draw rectangles around the detected faces in the frame.
            cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 0, 0), 2)
            # Crop the grayscale image to predict it.
            roi_gray = gray_frame[y:y + height, x:x + width]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
            prediction = model.predict(cropped_img)
            max_likelihood = int(np.argmax(prediction))
            cv2.putText(frame, MOOD_DICT[max_likelihood], (x + 20, y - 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255),
                        2, cv2.LINE_AA)

        cv2.imshow('Feed', frame)

        # Allow exiting the program with the q-key.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
else:
    print("Please enter a valid running mode, either 'train' or 'display'.")

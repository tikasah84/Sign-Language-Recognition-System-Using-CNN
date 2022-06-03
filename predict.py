"""
Contains functions : pre_process() and which() that are needed by translator.py for predicting image from webcam
"""
import cv2
import numpy as np

from keras.models import load_model


IMAGE_SIZE = 50  # We'll be working with 50 * 50 pixel images
MODEL_PATH = "word.h5"


LABELS=['Bathroom',
 'Bed',
 'Cry',
 'Drink',
 'Food',
 'Give',
 'Hello',
 'Help',
 'House',
 'Hug',
 'I',
 'Love',
 'me',
 'Namaste',
 'Ok',
 'Shut up',
 'Smile',
 'Thank You',
 'You']

THRESHOLD = 90

# Loads pretrained CNN Model from MODEL_PATH
model = load_model(MODEL_PATH)


def pre_process(img_array):
    img_array = cv2.resize(img_array, (50, 50))
    img_array = img_array.reshape(IMAGE_SIZE, IMAGE_SIZE, 1)

    # Normalize the array
    img_array = img_array / 255.0

    # Expand Dimension of the array as our model expects a 4D array
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def which(img_array):
   
    img_array = pre_process(img_array)
    preds = model.predict(img_array)
    preds *= 100
    most_likely_class_index = int(np.argmax(preds))
   
    return preds.max(), LABELS[most_likely_class_index]

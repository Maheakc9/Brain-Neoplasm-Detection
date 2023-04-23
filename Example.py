#Import all the required libraries

from keras.preprocessing import image
from keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing import image
import os
import pandas as pd


# Load the saved model

model_path="D:/PROJECT/NEOPLASM/model.h5"
model = load_model(model_path)
print(model)

print('Model Loaded Successfully')

# Reading the image
img_path = "D:/PROJECT/NEOPLASM/Dataset/prediction/yes1.jpg"

# convert image to np array and normalize
img = image.load_img(img_path, target_size=(64, 64))

# change dimention 3D to 4D
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)

# Predict the neoplasm
pred = model.predict(img_array)[0]
print(pred)


if pred == 0:
    print('Not detected Neoplasm')
else:
    print('Neoplasm detected')

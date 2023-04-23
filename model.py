# Importing necessary libraries
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.models import Sequential
from sklearn.metrics import confusion_matrix, classification_report

# Setting up the paths and parameters
train_path = 'D:/PROJECT/NEOPLASM/Dataset/train'
test_path = 'D:/PROJECT/NEOPLASM/Dataset/test'
batch_size = 32
epochs = 10

# Preprocessing the data
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

train_set = train_datagen.flow_from_directory(train_path,
                                              target_size = (64, 64),
                                              batch_size = batch_size,
                                              class_mode = 'binary')

test_set = test_datagen.flow_from_directory(test_path,
                                            target_size = (64, 64),
                                            batch_size = batch_size,
                                            class_mode = 'binary')

# Building the model
model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Conv2D(64, (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Conv2D(128, (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Flatten())

model.add(Dense(units = 128, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(units = 1, activation = 'sigmoid'))

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Training the model
model.fit(train_set,
          steps_per_epoch = len(train_set),
          epochs = epochs,
          validation_data = test_set,
          validation_steps = len(test_set))

# Evaluating the model
test_accuracy = model.evaluate(test_set, steps = len(test_set))[1]
print('Test accuracy:', test_accuracy)

model_json=model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
    model.save_weights("my_model_weights.h5")
    model.save("model.h5")
    print("Saved model to disk")

y_pred = model.predict(test_set, steps = len(test_set))
y_pred = np.round(y_pred).astype(int)
print(y_pred[1])
'''
# Predicting the test set results
y_pred = model.predict(test_set, steps = len(test_set))
y_pred = np.round(y_pred)

# Generating the classification report and confusion matrix
y_true = test_set.classes
target_names = ['No Tumor', 'Tumor']
print(classification_report(y_true, y_pred, target_names=target_names))
print(confusion_matrix(y_true, y_pred))
'''

# import pickle as pkl
from __future__ import print_function
import numpy as np
import argparse
import cv2
import pandas as pd
import numpy as np

# import keras
# from keras.models import Model
# from keras.layers import *
# from keras import optimizers

import matplotlib.pyplot as plt

import keras
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

from sklearn.model_selection import train_test_split


#Prepare Data
train_set = np.load('x_train3.npy', allow_pickle=True)
test_set = np.load('x_test3.npy', allow_pickle=True)


train_labels = pd.read_csv('train_max_y.csv')

train_labels = np.asarray(train_labels['Label'])
train_labels = to_categorical(train_labels)
print(train_labels.shape)


x_train, x_val, y_train, y_val = train_test_split(train_set, train_labels, test_size=0.15, random_state=42)

# Reshape data to fit keras Sequential requirements
def keras_reshape(a):
  return a.reshape(a.shape[0], 128, 128, 1)
 
x_train = keras_reshape(x_train)
x_val = keras_reshape(x_val)
x_train = x_train / 255
x_val = x_val / 255

# Define model

from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
input_shape = (128, 128, 1)
num_labels = y_train.shape[1]
print(num_labels)

model = Sequential()

model.add(ZeroPadding2D((1,1),input_shape=input_shape))
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D((2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.5))

model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.5))

# Additional layers
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.5))

model.add(Flatten()) #Flattening the 2D array for fully connected layers
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.summary()


model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


# epochs     - One Epoch is when an ENTIRE dataset is passed forward and backward through the neural network only ONCE
num_epochs = 30
# batch size -Total number of training examples present in a single batch.
batch_size = 200

# Fit model with Model checkpointing
history = model.fit(x_train, y_train, epochs=num_epochs,
                    batch_size=batch_size, 
                    verbose=1,
                    validation_data=(x_val, y_val))

# Predict on test set
x_test = keras_reshape(test_set)
x_test = x_test/255
pred = model.predict_classes(x_test, batch_size=200, verbose=1)

# Save Results
pd.DataFrame(pred).to_csv('results.csv')


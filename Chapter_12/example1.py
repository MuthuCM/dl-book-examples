# Example 12.1 AutoEncoder_ Example_1_ANN
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense
import numpy as np
import h5py
from pathlib import Path
import matplotlib.pyplot as plt
import tensorflow as tf

# Read MNIST data. We won't use the y_train or y_test data
(X_train, _), (X_test, _) = mnist.load_data()

# Cast values into the floating-point type using tf.keras.backend.cast_to_floatx
X_train = tf.keras.backend.cast_to_floatx(X_train)
X_test = tf.keras.backend.cast_to_floatx(X_test)

# Normalize the range from [0,255] to [0,1]
X_train /= 255.
X_test /= 255.

# Reshape the data into a grid with one row per sample, each row 784 (28*28) pixels
X_train = X_train.reshape((len(X_train), 784))
X_test = X_test.reshape((len(X_test), 784))

ann_model = Sequential()
ann_model.add(Dense(20, input_dim=784, activation='relu'))
ann_model.add(Dense(784, activation='sigmoid'))
ann_model.compile(optimizer='adadelta', loss='binary_crossentropy')
history =  ann_model.fit(X_train, X_train,
               epochs=50, batch_size=128, shuffle=True,
               verbose=2,
               validation_data=(X_test, X_test))


predictions = ann_model.predict(X_test)
print(predictions[0])

ann_model_2 = Sequential()
ann_model_2.add(Dense(512, input_dim=784, activation='relu'))
ann_model_2.add(Dense(256, activation='relu'))
ann_model_2.add(Dense(20, activation='relu'))
ann_model_2.add(Dense(256, activation='relu'))
ann_model_2.add(Dense(512, activation='relu'))
ann_model_2.add(Dense(784, activation='sigmoid'))
ann_model_2.compile(optimizer='adadelta', loss='binary_crossentropy')
history2 = ann_model_2.fit(X_train, X_train,
               epochs=50, batch_size=128, shuffle=True,
               verbose=2,
               validation_data=(X_test, X_test))
predictions2 = ann_model_2.predict(X_test)
print(predictions[0])

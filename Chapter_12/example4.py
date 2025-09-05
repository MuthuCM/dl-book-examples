#  Example 12.4 AutoEncoder Example_4_DENOISING 
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, UpSampling2D, MaxPooling2D, Conv2DTranspose # Import directly from keras.layers
import h5py
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf  # Import tensorflow
tf.keras.backend.set_image_data_format('channels_last')

def get_mnist_samples():
    random_seed = 42
    np.random.seed(random_seed)

    # Read MNIST data. We won't be using the y_train or y_test data
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    pixels_per_image = np.prod(X_train.shape[1:])

    # Cast values into the current floating-point type using tf.cast
    X_train = tf.cast(X_train, dtype=tf.keras.backend.floatx()) # Use tf.cast
    X_test = tf.cast(X_test, dtype=tf.keras.backend.floatx())  # Use tf.cast
    
    X_train = np.reshape(X_train, (len(X_train), 28, 28, 1)) 
    X_test = np.reshape(X_test, (len(X_test), 28, 28, 1)) 
    
    X_train = tf.cast(X_train, dtype=tf.keras.backend.floatx())
    X_test = tf.cast(X_test, dtype=tf.keras.backend.floatx())
    # Normalize the range from [0,255] to [0,1]
    X_train /= 255.
    X_test /= 255.

    return (X_train, X_test)

def add_noise_to_mnist(X_train, X_test, noise_factor=0.5): # add noise to the digits
    X_train_noisy = X_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X_train.shape) 
    X_test_noisy = X_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X_test.shape) 

    X_train_noisy = np.clip(X_train_noisy, 0., 1.)
    X_test_noisy = np.clip(X_test_noisy, 0., 1.)
    return (X_train_noisy, X_test_noisy)

def build_autoencoder1():
    # build the autoencoder.
    model = Sequential()
    model.add(Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(28,28,1)))
    model.add(MaxPooling2D((2,2,), padding='same'))
    model.add(Conv2D(32, (3,3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2,2), padding='same'))
    # down to 7, 7, 32 now go back up
    model.add(Conv2D(32, (3,3), activation='relu', padding='same'))
    model.add(UpSampling2D((2,2)))
    model.add(Conv2D(32, (3,3), activation='relu', padding='same'))
    model.add(UpSampling2D((2,2)))
    model.add(Conv2D(1, (3,3), activation='sigmoid', padding='same'))
    
    model.compile(optimizer='adadelta', loss='binary_crossentropy')
    return model

(X_train, X_test) = get_mnist_samples()
(X_train_noisy, X_test_noisy) = add_noise_to_mnist(X_train, X_test, 0.5)

model = build_autoencoder1()
history = model.fit(X_train_noisy, X_train,
                          epochs=100,
                          batch_size=128,
                          shuffle=True,
                          validation_data=(X_test_noisy, X_test))


predictions = model.predict(X_test_noisy)
print(predictions[0])
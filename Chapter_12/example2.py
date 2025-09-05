#  Example 12.2 AutoEncoder_ Example_2_CNN
from keras.models import Sequential, Model
from keras.layers import Conv2D, Dense, Input, MaxPooling2D, UpSampling2D
# Instead of from keras.utils import np_utils
from tensorflow.keras.utils import to_categorical # import to_categorical directly
from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
from keras import backend as keras_backend
from tensorflow.keras import losses, backend as KBE 
keras_backend.set_image_data_format('channels_last')

# Load the MNIST data. We won't use y_train and y_test data
(X_train, y_train), (X_test, y_test) = mnist.load_data()


# cast the sample data to the floating-point type
#Instead of X_train = keras_backend.cast_to_floatx(X_train) use
X_train = tf.keras.backend.cast_to_floatx(X_train) # use tf.keras.backend
X_test = tf.keras.backend.cast_to_floatx(X_test) # use tf.keras.backend

# reshape to 2D grid, one line per image
X_train = X_train.reshape(X_train.shape[0], 784)
X_test = X_test.reshape(X_test.shape[0], 784)

# scale data to range [0, 1]
X_train /= 255.0
X_test /= 255.0

# reshape sample data to 4D tensor using channels_last convention
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

# replace label data with one-hot encoded versions
y_train = to_categorical(y_train, 10) 
y_test = to_categorical(y_test, 10)

encoder_layer_1 = Input(shape=(28,28, 1))
encoder_layer_2 = Conv2D(16, (3, 3), activation='relu', padding='same')
encoder_layer_3 = MaxPooling2D((2,2), padding='same')
encoder_layer_4 = Conv2D(8, (3, 3), activation='relu', padding='same')
encoder_layer_5 = MaxPooling2D((2,2), padding='same')
encoder_layer_6 = Conv2D(3, (3, 3), activation='relu', padding='same')

decoder_layer_1 = UpSampling2D((2,2))
decoder_layer_2 = Conv2D(16, (3, 3), activation='relu', padding='same')
decoder_layer_3 = UpSampling2D((2,2))
decoder_layer_4 = Conv2D(1, (3, 3), activation='sigmoid', padding='same')

encoder_step_1 = encoder_layer_2(encoder_layer_1)
encoder_step_2 = encoder_layer_3(encoder_step_1)
encoder_step_3 = encoder_layer_4(encoder_step_2)
encoder_step_4 = encoder_layer_5(encoder_step_3)
encoder_step_5 = encoder_layer_6(encoder_step_4)

decoder_step_1 = decoder_layer_1(encoder_step_5)
decoder_step_2 = decoder_layer_2(decoder_step_1)
decoder_step_3 = decoder_layer_3(decoder_step_2)
decoder_step_4 = decoder_layer_4(decoder_step_3)


cnn_model = Model(encoder_layer_1, decoder_step_4)
cnn_model.compile(optimizer='adadelta', loss='binary_crossentropy')
history = cnn_model.fit(X_train, X_train,
               epochs=20, batch_size=128, shuffle=True,
               verbose=2,
               validation_data=(X_test, X_test))
predictions = cnn_model.predict(X_test)
print(predictions[0])


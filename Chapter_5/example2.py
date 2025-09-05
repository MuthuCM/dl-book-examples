# Example 5.2 [ Fashion Mnist dataset - CNN ]

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Flatten, Dense, Conv2D, MaxPooling2D
from keras.datasets import fashion_mnist

(training_images, training_labels), (test_images, test_labels) = fashion_mnist.load_data()

training_images=training_images.reshape(60000, 28, 28, 1)
training_images  = training_images / 255.0
test_images = test_images.reshape(10000, 28, 28, 1)
test_images = test_images / 255.0

model = Sequential([
    Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation="relu"),
    Dense(10, activation="softmax")])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

model.fit(training_images, training_labels, epochs=5, batch_size = 128)

model.evaluate(test_images, test_labels)

classifications = model.predict(test_images)


# Displaying test_images[0]
import pylab as plt
plt.imshow(test_images[0:1].reshape(28,28), cmap = 'gray')
plt.show()

# Displaying predicted value
print(classifications[0])
print(test_labels[0])

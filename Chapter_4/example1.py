#Example 4.1 - ANN Model]  [mnist dataset]

# Import Libraries
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Flatten, Dense
from keras. datasets import mnist

# Load Data
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

# Displaying training_images[0]
import pylab as plt
plt.imshow(training_images[7:8].reshape(28,28), cmap = 'gray') # Changed 'grey' to 'gray'
plt.show()

# Do Scaling
training_images  = training_images / 255.0
test_images = test_images / 255.0
training_images.shape
test_images.shape

# Build ANN Model
model = Sequential(
[Flatten(input_shape=(28,28)),
 Dense(128, activation="relu"),
 Dense(10, activation="softmax")])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
# Fit the Model
history = model.fit(training_images, training_labels, epochs=5,batch_size=128)

# Evaluate the Model
model.evaluate(test_images, test_labels)

# Do Prediction
classifications = model.predict(test_images)
print(classifications[0])
print(test_labels[0])




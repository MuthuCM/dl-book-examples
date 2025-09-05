#Example 5.1 [ Fashion Mnist dataset - ANN ]

from keras.models import Sequential
from keras.layers import Flatten, Dense
from keras.datasets import fashion_mnist

(training_images, training_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Displaying training_images[0]
import pylab as plt
plt.imshow(training_images[0:1].reshape(28,28), cmap = 'gray') # Changed 'grey' to 'gray'
plt.show()

training_images  = training_images / 255.0
test_images = test_images / 255.0

model = Sequential([Flatten(input_shape=(28,28)),
                    Dense(128, activation="relu"),
                    Dense(10, activation="softmax")])
adam=Adam(0.0001)

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(training_images, training_labels, epochs=5, batch_size = 128)

model.evaluate(test_images, test_labels)

classifications = model.predict(test_images)
print(classifications[0])
print(test_labels[0])


#Example 6.4 [ CIFAR10 dataset]

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import numpy as np
from keras.utils import to_categorical
from keras.losses import categorical_crossentropy
from keras.optimizers import Adadelta

batch_size = 64
num_classes = 10
epochs = 60
img_rows, img_cols = 32, 32

from keras.datasets import cifar10
(x_train,y_train),(x_test,y_test)= cifar10.load_data()

idx = np.argsort(np.random.random(y_train.shape[0]))
x_train = x_train[idx]
y_train = y_train[idx]
idx = np.argsort(np.random.random(y_test.shape[0]))
x_test = x_test[idx]
y_test = y_test[idx]

x_train = x_train.reshape(x_train.shape[0], 32, 32, 3)
x_test = x_test.reshape(x_test.shape[0], 32, 32, 3)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

y_train = to_categorical(y_train, 10)
y_test =  to_categorical(y_test, 10)

# CHANGE CODE TO OUR USUAL PATTERN
model = Sequential([   

          Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(32,32,3)),
          Conv2D(64, (3, 3), activation='relu'),
          Conv2D(64, (3, 3), activation='relu'),
          Conv2D(64, (3, 3), activation='relu'),
          Conv2D(64, (3, 3), activation='relu'),
          MaxPooling2D(2, 2),
          Flatten(),
          Dense(128, activation='relu'),
          Dropout(0.5),
          Dense(128, activation='relu'),
          Dropout(0.5),
          Dense(10, activation='softmax')
])

model.compile(loss = categorical_crossentropy,
              optimizer= Adadelta(),
              metrics=['accuracy'])

print("Model parameters = %d" % model.count_params())
print(model.summary())

history = model.fit(x_train, y_train,
          batch_size=64,
          epochs=5,
          verbose=1,
          validation_data=(x_test[:1000], y_test[:1000]))

score = model.evaluate(x_test[1000:], y_test[1000:], verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Visualize Accuracy & Loss
import matplotlib.pyplot as plt
def plot_graphs(history, string):
  plt.plot(history.history[string])
  plt.plot(history.history['val_'+string])
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.legend([string, 'val_'+string])
  plt.show()

plot_graphs(history, "accuracy")
plot_graphs(history, "loss")(Dogs vs Cats - Untitled15.ipynb hoddatascience)

# Transfer Learning
conv_base  = keras.applications.vgg16.VGG16(
    weights="imagenet",
    include_top=False)
conv_base.trainable = False

from keras import Sequential
from keras.layers import  RandomFlip, RandomRotation, RandomZoom
data_augmentation = Sequential(
    [
        RandomFlip("horizontal"),
        RandomRotation(0.1),
        RandomZoom(0.2)
    ]
)

from keras.layers import Rescaling, Conv2D, MaxPooling2D, Flatten, Dense
inputs = keras.Input(shape=(180, 180, 3))
x = data_augmentation(inputs)
x = keras.applications.vgg16.preprocess_input(x)
x = conv_base(x)
x = Flatten()(x)
x = Dense(256)(x)
# x = Dropout(0.5)(x)
outputs = Dense(1, activation="sigmoid")(x)
model = keras.Model(inputs, outputs)
model.compile(loss="binary_crossentropy",
              optimizer="rmsprop",
              metrics=["accuracy"])

#callbacks = [
#    keras.callbacks.ModelCheckpoint(
#       filepath="feature_extraction_with_data_augmentation.keras",
#        save_best_only=True,
#        monitor="val_loss")
#]
#history = model.fit(
#    train_dataset,
#    epochs=1,
#    validation_data=validation_dataset,
#    callbacks=callbacks)
history = model.fit(
    train_dataset,
    epochs=5,
    validation_data=validation_dataset
    )

test_loss, test_acc = model.evaluate(test_dataset)
print(f"Test accuracy: {test_acc:.3f}")


import numpy as np
from google.colab import files
from keras.preprocessing import image
from keras.utils import load_img, img_to_array
uploaded = files.upload()
for fn in uploaded.keys():
 
  # predicting images
  path = '/content/' + fn
  img = load_img(path, target_size=(180, 180))
  #img = tf.keras.utils.img_to_array(path, target_size=(180, 180))
  #img = tf.keras.utils.load_img(path, target_size=(180, 180))
  x = img_to_array(img)
  #x = image.img_to_array(img)
  x = np.expand_dims(x, axis=0)

  image_tensor = np.vstack([x])
  classes = model.predict(image_tensor)
  print(classes)
  print(classes[0])
  if classes[0]>0.5:
    print(fn + " is a cat")
  else:
    print(fn + " is a dog")
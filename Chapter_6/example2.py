#Example 6.2(Dogs vs Cats - Untitled15.ipynb hoddatascience)

# DATA AUGMENTATION
import keras
from keras import Sequential
from keras.layers import  RandomFlip, RandomRotation, RandomZoom
data_augmentation = Sequential(
    [
        RandomFlip("horizontal"),
        RandomRotation(0.1),
        RandomZoom(0.2)
    ]
)

from matplotlib import pyplot as plt
plt.figure(figsize=(10, 10))
for images, _ in train_dataset.take(1):
    for i in range(9):
        augmented_images = data_augmentation(images)
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(augmented_images[0].numpy().astype("uint8"))
        plt.axis("off")

from keras.layers import Rescaling, Conv2D, MaxPooling2D, Flatten, Dense
inputs = keras.Input(shape=(180, 180, 3))
x = data_augmentation(inputs)
x = Rescaling(1./255)(x)
x = Conv2D(32, kernel_size=3, activation="relu")(x)
x = MaxPooling2D(pool_size=2)(x)
x = Conv2D(64, kernel_size=3, activation="relu")(x)
x = MaxPooling2D(pool_size=2)(x)
x = Conv2D(128, kernel_size=3, activation="relu")(x)
x = MaxPooling2D(pool_size=2)(x)
x = Conv2D(256, kernel_size=3, activation="relu")(x)
x = MaxPooling2D(pool_size=2)(x)
x = Conv2D(256, kernel_size=3, activation="relu")(x)
x = Flatten()(x)
# x = layers.Dropout(0.5)(x)
outputs = Dense(1, activation="sigmoid")(x)
model = keras.Model(inputs=inputs, outputs=outputs)

model.compile(loss="binary_crossentropy",
              optimizer="rmsprop",
              metrics=["accuracy"])

callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath="convnet_from_scratch_with_augmentation.keras",
        save_best_only=True,
        monitor="val_loss")
]
history = model.fit(
    train_dataset,
    epochs=5,
    validation_data=validation_dataset,
    callbacks=callbacks)

test_model = keras.models.load_model(
    "convnet_from_scratch_with_augmentation.keras")
test_loss, test_acc = test_model.evaluate(test_dataset)
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
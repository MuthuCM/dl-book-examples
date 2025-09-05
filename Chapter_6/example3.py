#Example 6.3(Dogs vs Cats - Untitled15.ipynb hoddatascience)

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
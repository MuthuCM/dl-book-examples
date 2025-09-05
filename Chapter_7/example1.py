#Example 7.1 Leaf Disease Detection
import os

def total_files(folder_path):
    num_files = len([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))])
    return num_files

train_files_healthy = "/content/drive/MyDrive/DATASET/Train/Train/Healthy"
train_files_powdery = "/content/drive/MyDrive/DATASET/Train/Train/Powdery"
train_files_rust = "/content/drive/MyDrive/DATASET/Train/Train/Rust"

test_files_healthy = "/content/drive/MyDrive/DATASET/Test/Test/Healthy"
test_files_powdery = "/content/drive/MyDrive/DATASET/Test/Test/Powdery"
test_files_rust = "/content/drive/MyDrive/DATASET/Test/Test/Rust"

valid_files_healthy = "/content/drive/MyDrive/DATASET/Validation/Validation/Healthy"
valid_files_powdery = "/content/drive/MyDrive/DATASET/Validation/Validation/Powdery"
valid_files_rust = "/content/drive/MyDrive/DATASET/Validation/Validation/Rust"

print("Number of healthy leaf images in training set", total_files(train_files_healthy))
print("Number of powder leaf images in training set", total_files(train_files_powdery))
print("Number of rusty leaf images in training set", total_files(train_files_rust))

print("========================================================")

print("Number of healthy leaf images in test set", total_files(test_files_healthy))
print("Number of powder leaf images in test set", total_files(test_files_powdery))
print("Number of rusty leaf images in test set", total_files(test_files_rust))

print("========================================================")

print("Number of healthy leaf images in validation set", total_files(valid_files_healthy))
print("Number of powder leaf images in validation set", total_files(valid_files_powdery))
print("Number of rusty leaf images in validation set", total_files(valid_files_rust))

from PIL import Image
import IPython.display as display

image_path = '/content/drive/MyDrive/DATASET/Train/Train/Healthy/800edef467d27c15.jpg' 

with open(image_path, 'rb') as f:
    display.display(display.Image(data=f.read(), width=500))
image_path = '/content/drive/MyDrive/DATASET/Train/Train/Powdery/828d2f4a91754ddb.jpg'

with open(image_path, 'rb') as f:
    display.display(display.Image(data=f.read(), width=500))
image_path = '/content/drive/MyDrive/DATASET/Train/Train/Rust/80f09587dfc7988e.jpg'

with open(image_path, 'rb') as f:
    display.display(display.Image(data=f.read(), width=500))
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, 
                                                                                                                                                       horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
train_generator =   
               train_datagen.flow_from_directory('/content/drive/MyDrive/DATASET/Train/Train',
                                              target_size=(225, 225), batch_size=32,   class_mode='categorical')
validation_generator =  
                  test_datagen.flow_from_directory('/content/drive/MyDrive/DATASET/Validation
                                                            /Validation',  target_size=(225, 225),  batch_size=32,
                                                                                                                                     class_mode='categorical')
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(225, 225, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(3, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
history = model.fit(train_generator,
                    batch_size=32,
                    epochs=10,
                    validation_data=validation_generator,
                    validation_batch_size=32
                    )
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
import seaborn as sns
sns.set_theme()
sns.set_context("poster")
figure(figsize=(10, 8), dpi=100)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

model.save("disease_disease_recognition.h5")
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
def preprocess_image(image_path, target_size=(225, 225)):
    img = load_img(image_path, target_size=target_size)
    x = img_to_array(img)
    x = x.astype('float32') / 255.
    x = np.expand_dims(x, axis=0)
    return x
x =  preprocess_image('/content/drive/MyDrive/DATASET/Test/Test/Rust/
                                                                                                                        831abdc76c05e23d.jpg')
predictions = model.predict(x)
predictions[0]

labels = train_generator.class_indices
labels = {v: k for k, v in labels.items()}
labels
 
predicted_label = labels[np.argmax(predictions)]
print(predicted_label)


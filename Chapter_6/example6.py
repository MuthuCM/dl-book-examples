#Example 6.6 SIGN LANGUAGE RECOGNITION[sign_lang_recognition.ipynb]
import os
import cv2
import pickle
import numpy as np
import seaborn as sn
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from keras.layers import Conv2D, Flatten, Dense, AveragePooling2D, Dropout
import matplotlib.pyplot as plt

path = '/content/drive/MyDrive/asl dataset/asl_dataset'
data,label = [],[]
for root, dirs, files in os.walk(path):
    key = os.path.basename(root)
    for file in files:
        full_file_path = os.path.join(root,file)
        img = cv2.imread(full_file_path)
        img = cv2.resize(img,(128,128))
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        data.append(img)
        label.append(key)

data = np.array(data)
label = np.array(label)
x_train, x_test0, y_train, y_test0 = train_test_split(data, label, test_size=0.2)
x_test, x_val, y_test, y_val = train_test_split(x_test0, y_test0, test_size=0.5)
print(x_train.shape)
print(y_train.shape)
print(x_val.shape)
print(y_val.shape)
print(x_test.shape)
print(y_test.shape)

# Normalization
x_train = x_train/255.0
x_val = x_val/255.0
x_test = x_test/255.0
#Encode labels from string to int
le = preprocessing.LabelEncoder()
labelEnc_train = le.fit_transform(y_train)
labelEnc_test = le.fit_transform(y_test)
labelEnc_val = le.fit_transform(y_val)
print(x_val.shape)
print(labelEnc_val.shape)

num_classes = 36

# CNN Model Definition
model = keras.Sequential()

model.add(Conv2D(32, (5,5), activation = 'relu', input_shape = (128,128,3)))
model.add(AveragePooling2D(pool_size=(2, 2))) # Added pool_size

model.add(Conv2D(64, (5,5), activation = 'relu'))
model.add(AveragePooling2D(pool_size=(2, 2))) # Added pool_size

model.add(Flatten())
model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation = 'softmax'))

model.summary()

# compile the neural network
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])
history = model.fit(x_train, labelEnc_train, validation_data=(x_val,labelEnc_val), epochs=6, 
                                                                                                                                                                      batch_size=32)
loss, accuracy = model.evaluate(x_test, labelEnc_test)
print('Test Accuracy =', accuracy)

# Plot the loss value
plt.figure(figsize=(5,3))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.show()

# Plot the accuracy value
plt.figure(figsize=(5,3))
plt.plot(history.history['acc'], label='Train Accuracy')
plt.plot(history.history['val_acc'], label='Validation Accuracy')
plt.legend()
plt.show()

def predict_input_image_type(img):
    image = img.reshape(-1,128,128,3)
    prediction = mod.predict(image)[0]
    confidences = {labels[i]: float(prediction[i]) for i in range(36)}
    return confidences

def predict_and_display(img_path):
    # Load and display the image
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convert to RGB for display
    plt.imshow(img_rgb)
    plt.axis('off')
    plt.show()

    # Resize image for prediction
    img = cv2.resize(img, (128, 128))
    # Predict the sign
    confidences = predict_input_image_type(img)
    predicted_class = max(confidences, key=confidences.get)
    print("Predicted Sign:", predicted_class)
predict_and_display('/content/drive/MyDrive/samplepic.jpeg')

display_and_predict('/content/drive/MyDrive/8.jpg')



#Example 7.2 Paddy Disease Detection
!pip install kaggle
import os
os.environ['KAGGLE_CONFIG_DIR'] = 'kaggle'
import os

# Command to download the Kaggle dataset
os.system('kaggle datasets download -d imbikramsaha/paddy-doctor')
import os

# List files in the current directory
print(os.listdir())

import zipfile
# Define the correct path to your zip file
file_path = 'paddy-doctor.zip'  # The file is in the current directory

# Unzip the file to a specific destination
with zipfile.ZipFile(file_path, 'r') as zip_ref:
    zip_ref.extractall('paddy disease') 

pip install opencv-python-headless
pip install tensorflow
import os
import time
import shutil
import pathlib
import itertools

# import data handling tools
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
sns.set_style('darkgrid')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# import Deep learning Libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam, Adamax
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation,  
                                                                                      Dropout, BatchNormalization,AveragePooling2D
from tensorflow.keras import regularizers

# Ignore Warnings
import warnings
warnings.filterwarnings("ignore")

print ('modules loaded')
import tensorflow as tf
print(tf.__version__)

#generate data paths with labels
def define_paths(data_dir):
    filepaths = []
    labels = []

    folds = os.listdir(data_dir)
    for fold in folds:
        foldpath = os.path.join(data_dir, fold)
        filelist = os.listdir(foldpath)
        for file in filelist:
            fpath = os.path.join(foldpath, file)
            filepaths.append(fpath)
            labels.append(fold)

    return filepaths, labels

# Concatenate data paths & labels into one dataframe, which will be used for fitting model 
def define_df(files, classes):
    Fseries = pd.Series(files, name= 'filepaths')
    Lseries = pd.Series(classes, name='labels')
    return pd.concat([Fseries, Lseries], axis= 1)

# Split dataframe to train, valid, and test
def split_data(data_dir):
    # train dataframe
    files, classes = define_paths(data_dir)
    df = define_df(files, classes)
    strat = df['labels']
    train_df, dummy_df = train_test_split(df,  train_size= 0.8, shuffle= True, 
                                                                                                                 random_state=  123, stratify= strat)
    # valid and test dataframe
    strat = dummy_df['labels']
    valid_df, test_df = train_test_split(dummy_df,  train_size= 0.5, shuffle= True, 
                                                                                                                random_state= 123, stratify= strat)
    return train_df, valid_df, test_df

def create_gens (train_df, valid_df, test_df, batch_size):
    # Image data generator converts images into tensors.
    # define model parameters
    img_size = (224, 224)
    channels = 3 # either BGR or Grayscale
    color = 'rgb'
    img_shape = (img_size[0], img_size[1], channels)

    # Compute test data batch size and number of steps
    ts_length = len(test_df)
    test_batch_size = max(sorted([ts_length // n for n in range(1, ts_length + 1) if ts_length%n 
                                                                                                                                        == 0 and ts_length/n <= 80]))
    test_steps = ts_length // test_batch_size

    # This function will be used in image data generator for data augmentation, 
    # It  takes the image and returns it after transformation
    def scalar(img):
        return img

    tr_gen = ImageDataGenerator(preprocessing_function= scalar, horizontal_flip= True)
    ts_gen = ImageDataGenerator(preprocessing_function= scalar)

    train_gen = tr_gen.flow_from_dataframe( train_df, x_col= 'filepaths', y_col= 'labels', 
                                        target_size= img_size, class_mode= 'categorical',
                                        color_mode= color, shuffle= True, batch_size= batch_size)

    valid_gen = ts_gen.flow_from_dataframe( valid_df, x_col= 'filepaths', y_col= 'labels', 
                                        target_size= img_size, class_mode= 'categorical',
                                        color_mode= color, shuffle= True, batch_size= batch_size)

  
  # We will use custom test_batch_size, and make shuffle= false
    test_gen = ts_gen.flow_from_dataframe( test_df, x_col= 'filepaths', y_col= 'labels', 
                                        target_size= img_size, class_mode= 'categorical',
                                        color_mode= color, shuffle= False, batch_size= test_batch_size)

    return train_gen, valid_gen, test_gen
def get_dataset(path, image_width=224, image_height=224, batch_size=64):

    train_ds=tf.keras.utils.image_dataset_from_directory(
                           path, validation_split=0.2, subset='training',  seed=123,
                           image_size=(image_width, image_height),
                          batch_size=batch_size)

    val_ds=tf.keras.utils.image_dataset_from_directory(
                           path, validation_split=0.2, subset='validation', seed=123,
                           image_size=(image_width, image_height),
                           batch_size=batch_size)
    return train_ds,val_ds

 print("Class names:", train_ds.class_names)

# Create Model Structure
img_size = (224, 224)
channels = 3
img_shape = (img_size[0], img_size[1], channels)
class_count = len(list(train_ds.class_names)) # to define number of classes in dense layer
# Load the pre-trained EfficientNetB4 model without the top classification layer
base_model = tf.keras.applications.efficientnet.EfficientNetB4(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the pre-trained base model layers
base_model.trainable = False

model = Sequential([
    base_model,
    MaxPooling2D(),
    Flatten(),
    Dense(220, activation='relu'),
    Dropout(0.25),
    Dense(class_count, activation= 'softmax')
])
model.compile(optimizer='adam', loss= tf.keras.losses.sparse_categorical_crossentropy , metrics= ['accuracy'])
model.build(input_shape=(None, img_size[0], img_size[1], channels))
model.summary()

model_cnn = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255),
    tf.keras.layers.Conv2D(128, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(16, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.50),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10,activation='softmax')
])

# model.compile(optimizer='adam', loss= 'categorical_crossentropy', metrics= ['accuracy'])
model_cnn.compile(optimizer='adam', loss= tf.keras.losses.sparse_categorical_crossentropy , metrics= ['accuracy'])
model_cnn.build(input_shape=(None, img_size[0], img_size[1], channels))
model_cnn.summary()

batch_size = 64   # set batch size for training
epochs = 5   # number of all epochs in training
history = model.fit(x= train_ds, epochs= epochs, callbacks = callbacks,
                    validation_data= val_ds, verbose = 0)

history_cnn = model_cnn.fit(x= train_ds, epochs= epochs, callbacks = callbacks,
                    validation_data= val_ds, verbose = 0)

plot_training(history)
plot_training(history_cnn)
train_score = model.evaluate(train_ds, steps= test_steps, verbose= 1)
valid_score = model.evaluate(val_ds, steps= test_steps, verbose= 1)
print("Train Loss: ", train_score[0])
print("Train Accuracy: ", train_score[1])
print('-' * 20)
print("Validation Loss: ", valid_score[0])
print("Validation Accuracy: ", valid_score[1])

train_score = model_cnn.evaluate(train_ds, steps= test_steps, verbose= 1)
valid_score = model_cnn.evaluate(val_ds, steps= test_steps, verbose= 1)
print("Train Loss: ", train_score[0])
print("Train Accuracy: ", train_score[1])
print('-' * 20)
print("Validation Loss: ", valid_score[0])
print("Validation Accuracy: ", valid_score[1])


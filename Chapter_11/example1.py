# EXAMPLE 11.1 [Sentiment Analysis - ANN , CNN ] [ Anaconda - Chap8_Example2]
import pandas as pd
sentiments_data = pd.read_csv("D:/2_DL_Material/DataSets/LabelledNewsData.csv",encoding = "ISO-8859-1")
sentiments_data.head(1)

# Keras package for the deep learning model for the sentiment prediction. 
from keras.preprocessing.text import Tokenizer
#from keras.preprocessing.sequence import pad_sequences
from keras_preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM,Bidirectional, Dropout, Activation,GlobalAveragePooling1D
#from keras.layers.embeddings import Embedding
from keras.layers import Embedding

# Load libraries
import statsmodels.api as sm
import seaborn as sns
import pandas as pd
import numpy as np
import datetime
from datetime import date
import matplotlib.pyplot as plt

### Create sequence
vocabulary_size = 20000
tokenizer = Tokenizer(num_words= vocabulary_size)
tokenizer.fit_on_texts(sentiments_data['headline'])
sequences = tokenizer.texts_to_sequences(sentiments_data['headline'])
X = pad_sequences(sequences, maxlen=50)

from sklearn.model_selection import train_test_split
validation_size = 0.3
seed = 7
Y = sentiments_data["sentiment"]
X_train, X_test, Y_train, Y_test = train_test_split(X, \
                       Y, test_size=validation_size, random_state=seed)

# Creating RNN(LSTM) Model
def create_model(input_length=50):
    model = Sequential()
    model.add(Embedding(20000, 64, input_length=50))
    #model.add(GlobalAveragePooling1D())
    model.add(LSTM(100, dropout = 0.2, recurrent_dropout = 0.2))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])    
    return model

from keras.wrappers.scikit_learn import KerasClassifier
model_ANN = KerasClassifier(build_fn=create_model, epochs=10, verbose=1, validation_split=0.4)

import matplotlib.pyplot as plt


def plot_graphs(history, string):
  plt.plot(history.history[string])
  plt.plot(history.history['val_'+string])
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.legend([string, 'val_'+string])
  plt.show()
  
plot_graphs(history, "accuracy")
plot_graphs(history, "loss")

# Creating Bidirectional LSTM Model
def create_model(input_length=50):
    model = Sequential()
    model.add(Embedding(20000, 64, input_length=50))
    #model.add(GlobalAveragePooling1D())
    model.add(Bidirectional(LSTM(100, dropout = 0.2, recurrent_dropout = 0.2)))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])    
    return model 

from keras.wrappers.scikit_learn import KerasClassifier
model_ANN = KerasClassifier(build_fn=create_model, epochs=10, verbose=1, validation_split=0.4)

history = model_ANN.fit(X_train, Y_train)

import matplotlib.pyplot as plt


def plot_graphs(history, string):
  plt.plot(history.history[string])
  plt.plot(history.history['val_'+string])
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.legend([string, 'val_'+string])
  plt.show()
  
plot_graphs(history, "accuracy")
plot_graphs(history, "loss")

# Creating Stacked LSTM Model
def create_model(input_length=50):
    model = Sequential()
    model.add(Embedding(20000, 64, input_length=50))
    #model.add(GlobalAveragePooling1D())
    model.add(Bidirectional(LSTM(100, return_sequences = True)))
    model.add(Bidirectional(LSTM(100, dropout = 0.2, recurrent_dropout = 0.2)))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])    
    return model 

from keras.wrappers.scikit_learn import KerasClassifier
model_ANN = KerasClassifier(build_fn=create_model, epochs=10, verbose=1, validation_split=0.4)

history = model_ANN.fit(X_train, Y_train)

import matplotlib.pyplot as plt


def plot_graphs(history, string):
  plt.plot(history.history[string])
  plt.plot(history.history['val_'+string])
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.legend([string, 'val_'+string])
  plt.show()
  
plot_graphs(history, "accuracy")
plot_graphs(history, "loss")

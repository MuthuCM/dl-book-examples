# EXAMPLE 10.1 [Building ANN for Stock Market News Sentiment Analysis - ANN , CNN ] [ Anaconda - Chap8_Example1]
import pandas as pd
sentiments_data = pd.read_csv("D:/2_DL_Material/DataSets/LabelledNewsData.csv",encoding = "ISO-8859-1")


sentiments_data.head(1)

# Keras package for the deep learning model for the sentiment prediction. 
import tensorflow as tf
from keras.preprocessing.text import Tokenizer

#from keras.preprocessing.sequence import pad_sequences
from keras_preprocessing.sequence import pad_sequences

from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM, Dropout, Activation,GlobalAveragePooling1D

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

def create_model(input_length=50):
    model = Sequential()
    model.add(Embedding(20000, 16, input_length=50))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(24, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])    
    return model 

from keras.wrappers.scikit_learn import KerasClassifier
model_ANN = KerasClassifier(build_fn=create_model, epochs=30, verbose=1, validation_split=0.4)

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

# Reducing Learning Rate
def create_model(input_length=50):
    from keras.optimizers import Adam
    model3 = Sequential()
    model3.add(Embedding(20000, 16, input_length=50))
    model3.add(GlobalAveragePooling1D())
    model3.add(Dense(24, activation='relu'))
    model3.add(Dense(1, activation='sigmoid'))
    adam = Adam(learning_rate = 0.0001,beta_1 = .9,beta_2 = .999, amsgrad = False)
    model3.compile(loss='binary_crossentropy', optimizer= adam, metrics=['accuracy'])    
    return model3

from keras.wrappers.scikit_learn import KerasClassifier
model_ANN_2 = KerasClassifier(build_fn=create_model, epochs=30, verbose=1, validation_split=0.4)

history = model_ANN_2.fit(X_train, Y_train)

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

wc = tokenizer.word_counts
print(wc)

from collections import OrderedDict
newlist = (OrderedDict(sorted(wc.items(),key=lambda t:t[1],reverse=True)))
print(newlist)

xs=[]
ys=[]
curr_x = 1
for item in newlist:
  xs.append(curr_x)
  curr_x=curr_x+1
  ys.append(newlist[item])

print(ys)
plt.plot(xs,ys)
#plt.axis([300,10000,0,100])
plt.show()

plt.plot(xs,ys)
plt.axis([300,10000,0,100])
plt.show()

### REDUCING VOCABULARY SIZE TO 2000
vocabulary_size = 2000
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

def create_model(input_length=50):
    model = Sequential()
    model.add(Embedding(2000, 16, input_length=50))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(24, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])    
    return model 

from keras.wrappers.scikit_learn import KerasClassifier
model_ANN = KerasClassifier(build_fn=create_model, epochs=30, verbose=1, validation_split=0.4)

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

# REDUCING EMBEDDING DIMENTION to 7
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

def create_model(input_length=50):
    model = Sequential()
    model.add(Embedding(20000, 7, input_length=50))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(24, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])    
    return model

from keras.wrappers.scikit_learn import KerasClassifier
model_ANN = KerasClassifier(build_fn=create_model, epochs=30, verbose=1, validation_split=0.4)

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

# REDUCING NUMBER OF NEURONS TO 8
def create_model(input_length=50):
    model = Sequential()
    model.add(Embedding(20000, 16, input_length=50))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])    
    return model 

from keras.wrappers.scikit_learn import KerasClassifier
model_ANN = KerasClassifier(build_fn=create_model, epochs=30, verbose=1, validation_split=0.4)

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

# USING DROPOUT
def create_model(input_length=50):
    model = Sequential()
    model.add(Embedding(20000, 16, input_length=50))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(8, activation='relu'))
    Dropout(0.25)
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])    
    return model

from keras.wrappers.scikit_learn import KerasClassifier
model_ANN = KerasClassifier(build_fn=create_model, epochs=30, verbose=1, validation_split=0.4)

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

# USING REGULARIZATION
from keras.regularizers import l2
def create_model(input_length=50):
    model = Sequential()
    model.add(Embedding(20000, 16, input_length=50))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(8, activation='relu',kernel_regularizer = l2(0.01))) 
    Dropout(0.25)
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])    
    return model

from keras.wrappers.scikit_learn import KerasClassifier
model_ANN = KerasClassifier(build_fn=create_model, epochs=30, verbose=1, validation_split=0.4)

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

from bs4 import BeautifulSoup
import string

stopwords = ["a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at",
             "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could", "did", "do",
             "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have", "having",
             "he", "hed", "hes", "her", "here", "heres", "hers", "herself", "him", "himself", "his", "how",
             "hows", "i", "id", "ill", "im", "ive", "if", "in", "into", "is", "it", "its", "itself",
             "lets", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or", "other", "ought",
             "our", "ours", "ourselves", "out", "over", "own", "same", "she", "shed", "shell", "shes", "should",
             "so", "some", "such", "than", "that", "thats", "the", "their", "theirs", "them", "themselves", "then",
             "there", "theres", "these", "they", "theyd", "theyll", "theyre", "theyve", "this", "those", "through",
             "to", "too", "under", "until", "up", "very", "was", "we", "wed", "well", "were", "weve", "were",
             "what", "whats", "when", "whens", "where", "wheres", "which", "while", "who", "whos", "whom", "why",
             "whys", "with", "would", "you", "youd", "youll", "youre", "youve", "your", "yours", "yourself",
             "yourselves"]

table = str.maketrans('', '', string.punctuation)

#DOING PREDICTION
# CLASSIFYING A HEADLINE
sentences = ["Microsoft has done well in this quarter","Google has performed badly in this quarter","Amazon is struggling in this quarter"]
sequences = tokenizer.texts_to_sequences(sentences)
print(sequences)


max_length = 50
trunc_type='post'
padding_type='post'

padded = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
print(padded)

print(model_ANN.predict(padded))

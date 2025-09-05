#Example 8.1 Classifying News headlines using RNN
# Importing the libraries
import pandas as pd
import numpy as np
from keras.datasets import reuters # a collection of documents with news articles
from keras.utils import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, Activation, LSTM, GRU
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score

# Initializing the values
num_words=None
maxlen=50
test_split=0.3

# Splitting the dataset into train and test sets
(x_train,y_train),(x_test,y_test) = reuters.load_data(num_words = num_words, maxlen = maxlen, test_split = test_split)

from numpy.core.fromnumeric import shape
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

print(x_train[0])
print(y_train[0])

x_train=pad_sequences(x_train,padding="post")
x_test=pad_sequences(x_test,padding="post")
x_train = np.array(x_train).reshape((x_train.shape[0],x_train.shape[1],1))
x_test = np.array(x_test).reshape((x_test.shape[0],x_test.shape[1],1))

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#Simple RNN
model = Sequential([
    SimpleRNN(50, input_shape=(49,1)),
    Dense(46),
    Activation('softmax')
    
])  


#Model compilation
adam = Adam(learning_rate=0.001)
model.compile(loss = 'categorical_crossentropy', optimizer = adam, metrics = ['accuracy'])
model.fit(x_train,y_train, epochs = 200, validation_split=0.3)

#Model Evaluation
y_pred = np.argmax(model.predict(x_test),axis = 1)
y_test = np.argmax(y_test, axis = 1)
print(accuracy_score(y_pred,y_test))

#LSTM Model
model2 = Sequential([
    LSTM(50, input_shape=(49,1)),
    Dense(46),
    Activation('softmax')
    ])  

#Model compilation
adam = Adam(learning_rate = 0.001)
model2.compile(loss = 'categorical_crossentropy', optimizer = adam, metrics = ['accuracy'])
model2.fit(x_train,y_train, epochs = 100, validation_split=0.3)

#Model Evaluation
y_pred = np.argmax(model2.predict(x_test), axis = 1)
print(accuracy_score(y_pred, y_test))

#GRU Model
model3 = Sequential([
    GRU(50, input_shape=(49,1)),
    Dense(46),
    Activation('sigmoid')
    ])

#Model compilation
adam = Adam(learning_rate = 0.001)
model3.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
model3.fit(x_train,y_train, epochs = 100, validation_split = 0.3)

#Model Evaluation
y_pred = np.argmax(model3.predict(x_test), axis = 1)
print(accuracy_score(y_pred, y_test))

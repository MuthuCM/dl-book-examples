#Example 2.2 [ Regression]
# Import Libraries
import pandas as pd
import numpy as np
import sklearn
import tensorflow as tf
from sklearn.datasets import make_regression
from keras.optimizers import Adam
from keras.layers import Dense
from keras.models import Sequential

# Importing the dataset
data=make_regression(200,4,random_state=1)
data

x=data[0]
y=data[1]

from keras import backend as K 
import tensorflow as tf 
from tensorflow.keras.backend import sum,mean, square, epsilon

# define r2 function
def r2(y_true, y_pred):
    ss_res = sum(square(y_true - y_pred)) 
    ss_tot = sum(square(y_true - mean(y_true))) 
    return (1 - ss_res / (ss_tot + epsilon()))

# Model Creation
model = Sequential([
  Dense(1135,activation='tanh',input_dim=4),
  Dense(625,activation='relu'),
  Dense(114,activation='relu'),
  Dense(1,activation='sigmoid')
  ])
adam=Adam(0.0001)

# Model Compilation

model.compile(optimizer=adam,loss='mean_squared_error',metrics=[r2])
print(model.summary())

history=model.fit(x,y,epochs=150,batch_size=5,validation_split=0.2)

#pd.DataFrame(model.history.history)[['r2','val_r2']].reset_index().plot('index',kind='line')

#pd.DataFrame(model.history.history).reset_index().plot('index',kind='line')
df1 = pd.DataFrame(model.history.history)
df2 = df1.reset_index()
df2.plot('index',kind='line')

#pd.DataFrame(model.history.history)[["r2","val_r2"]].reset_index().plot('index',kind='line')
df3 = pd.DataFrame(model.history.history)
df4 = df3[["r2","val_r2"]]
df5 = df4.reset_index()
df5.plot('index',kind='line')

df3.loc[df3["r2"].idxmax()]
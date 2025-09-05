#Example 2.1 [ Classification]

# Import Libraries
import pandas as pd
import numpy as np
import sklearn
import tensorflow as tf
from sklearn.datasets import make_classification
from keras.optimizers import Adam
from keras.layers import Dense
from keras.models import Sequential

# Import the dataset
data=make_classification(200,4,random_state=1)
data

x=data[0]
y=data[1]

# Create Model 
model = Sequential([
  Dense(1135,activation='tanh',input_dim=4),
  Dense(625,activation='relu'),
  Dense(114,activation='relu'),
  Dense(1,activation='sigmoid')
  ])
# Model compilation
adam=Adam(0.001)
model.compile(optimizer=adam,loss='binary_crossentropy',metrics=['accuracy'])
print(model.summary())

history=model.fit(x,y,epochs=150,batch_size=5,validation_split=0.2)
df1 = pd.DataFrame(model.history.history)
df2 = df1.reset_index()
df2.plot('index',kind='line')

df1.loc[data["accuracy"].idxmax()]

# Example 1.2
# Import Libraries
import tensorflow as tf
import numpy as np
from keras import Sequential
from keras.layers import Dense

# Specify Data
xs = np.array([1, 2, 3, 4, 5])
ys = np.array([3, 5, 7, 9, 11])

# Figure out the Mapping Function
layer_0 = Dense(units=1, input_shape=[1])
model = Sequential([layer_0])
#model = Sequential(Dense(units=1, input_shape=[1]))
model.compile(optimizer='sgd', loss='mean_squared_error')
model.fit(xs, ys, epochs = 1000)

# Predict the value of y when x = 6
print(model.predict(np.array([6.0])))
print("Values of W and B are: {}".format(layer_0.get_weights()))

# Display the Regression Equation
parameters = layer_0.get_weights()
W = np.round(parameters[0][0][0])
B = np.round(parameters[1][0])
print(f"Fitted Equation is: Y = {W}X + {B}")

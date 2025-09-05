# Example 1.1
import numpy as np
x = [1,2,3,4,5]
y = [3,5,7,9,11]
n = len(x)
sumx = sumy = sumxx = sumxy = 0
for i in range(0,n):
  sumx = sumx + x[i]
  sumy = sumy + y[i]
  sumxx = sumxx + x[i] * x[i]
  sumxy = sumxy + x[i] * y[i]

W = (n * sumxy - sumx * sumy)/(n*sumxx - sumx * sumx)
B = (sumy - W * sumx)/n
print(f"Fitted Equation is: Y = {W}X + {B}")

# Predict y when x = 6
predicted_value = W * 6 + B
print(f"Value of Y when X = 6 is  {predicted_value}")
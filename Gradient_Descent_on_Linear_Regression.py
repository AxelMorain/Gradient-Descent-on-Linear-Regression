# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 21:05:53 2022

Project Type: Side Project 

File Name: Gradient_Descent_on_Linear_Regression

What:  Optimising a Linear Regression model using Gradient Descent.
    We are going to create our own Gradient Descent algorithm and code every
    line of it. In the process we are going to learn about tensors and 
    how to manipulate them using the TensorFlow library.
    
Why: For fun and learning purposes. 

How: With a smile =)

Function Created:
-
-
-   

TO DO:
- Make up some data 
- Run an approved Linear Regression
- Create my own Gradient Descent model
    -The model on paper to discuss the theory
    -The model in code
- Compare results
- Extra Analysis for even more fun
    
    
notes:
- 
- 
- 
-
-

Summarizing what is going on:
- 
-
-
-
"""


import numpy as np
import random as rd
rd.seed(4)
import matplotlib.pyplot as plt
from scipy import stats
import tensorflow as tf



''' Make up some data 
'''
# Create a line then add noise to it.

X = np.array((list(range(0, 20))))

# y = m*x + b + noise, where m = 1 and b = 5
#m = 1
#b = 5

#Or m and b can be random integers... up to you...
m = rd.randint(-10,10)
b = rd.randint(-10,10)

def noise (minimum, maximum):
    return rd.randint(minimum, maximum)

Y = np.empty(20)
for i in range(0, len(X)):
    Y[i] = m * X[i] + b + noise(-8,8)

# Plotting check
plt.scatter(X,Y)
plt.show()
    
''' Run an aproved Linear Regression
'''
oficial_results = stats.linregress(x = X, y = Y)
print("Benchmark m      =", oficial_results[0])
print("Benchmark b      =", oficial_results[1])


''' Create my own Gradient Descent model
'''
# We are going to use a tensor to store our data.
# Each x, y plane will be a matrix containing information about the model and
#and the values it found for m and b (our 2 unknouwn). Think of this matrix as
#your regular Pandas dataframe.
# For each iteration of your model through the optimizer, a new matrix will be 
# created and stacked on the top previous one, adding layers into the z axis.

# Here will be the columns of each layer/matrix on the x, y plane of the tensor:
# X, Y (for true Y), m, b, Y1 (Extimated Y), MSE (Mean Squared Error)

#--------
# Let's create the first layer, we set the first values of m and b randomly
#--------

m = rd.randint(-5,6)
b = rd.randint(-5,5)


m = np.full(len(X),m)
b = np.full(len(X),b)
Y1 = m * X + b
MSE = np.full(len(X),(1/len(X)) * np.sum((Y - Y1)**2))

# Ploting
# Plot of the original data and the starting point of the algorithm
plt.scatter(X,Y, color = 'b')
plt.scatter(X,Y1, color = 'g')
plt.title("Actual and Estimated Data")
plt.show()
# cool


# Now make our np arrays into a matrix and then a tensor

matrix1 = np.matrix([X, Y, m, b, Y1, MSE])
matrix1 = matrix1.T

matrix1[2,0]

# Since we are dealing with matrices and not Padas df, we can't really call
#data using their column names. 
# To make it easier on ourselves, we are going to give a name to each one of
#the integers that could be use to call a column.
 
jX = 0 # Will call the column on the j's axis of an x, j marix
       #that contains the values for X
jY = 1 # Will call the column on the j's axis of an x, j marix
       #that contains the values for Y
jm = 2 # ...you get the point....
jb = 3
jY1 = 4
jMSE = 5

# making a tensor with TensorFlow

tensor_layer_1 = tf.convert_to_tensor(matrix1, dtype = tf.float64)
tensor = tf.stack([tensor_layer_1, tensor_layer_1], axis = 0)
# As a basis for our tensor, the two first layer would be identical

#--------
# Let's create the iteration function that will improve the model
#--------

# But first let's create the function that will calculate the new values for
#m and for b

def next_m (previous_m, alpha,  X, Y, previous_b):
    return previous_m + alpha * ((1/len(X)) * np.sum( 2*X * (Y - (previous_m*X + previous_b)))) 

def next_b (previous_m,alpha, X, Y, previous_b):
    return previous_b + alpha * ( (1/len(X)) * np.sum( 2 * (Y - (previous_m*X + previous_b)))) 

def Y1 (m, X, b):
    return (m*X + b)

def MSE(true_Y, estimated_Y):
    Y, Y1 = true_Y, estimated_Y
    return (1/len(Y)) * np.sum((Y - Y1)**2)


max_iteration = 4000
threshold = 0.0000001 # for call back function
set_alpha = 0.005 # Learning rate

for k in range(2, max_iteration): # 2 because you start at the new layer(0, 1,..)
    temp_matrix = np.array(tensor[k-1, :, :], dtype = "float64")
    # So tensors are by nature hard to alter. So as a loop hole, we are grabing 
    #the layer we would like to modify (the newest layer), turning it into a
    #np.array (a matrix), calculate the new values of m and b using Gradient
    #Descent, then we calculate Y1 (the estimation of Y by our algorithm)
    #and then the MSE.
    # The np.array we have now is the freshest update of our algorithm.
    # We convert it into a TensorFlow matrix and then add it/layer it on the 
    #top of all the previous itteration (on the z or k axis)
    # This process is been repeated until the maximum number of itteration is
    #reached or a threshold for the change in MSE is met. 

    temp_matrix[:,jm] = next_m( previous_m = tensor[k-1, :, jm],
                               alpha = set_alpha,
                               X = tensor[k-1, :, jX],
                               Y = tensor[k-1, :, jY],
                               previous_b= tensor[k-1, :, jb])
    
    temp_matrix[:,jb] = next_b( previous_m = tensor[k-1, :, jm],
                               alpha = set_alpha,
                               X = tensor[k-1, :, jX],
                               Y = tensor[k-1, :, jY],
                               previous_b= tensor[k-1, :, jb])
    
    temp_matrix[:, jY1] = Y1(m = temp_matrix[:,jm],
                            X = temp_matrix[:,jX],
                            b = temp_matrix[:,jb])
    
    temp_matrix[:, jMSE] = MSE(true_Y = temp_matrix[:,jY],
                               estimated_Y= temp_matrix[:,jY1])

    # Plot the data every 50 itterations
    if (k%50 == 0):
        plt.scatter(temp_matrix[:,jX],temp_matrix[:,jY], color = 'b')
        plt.scatter(temp_matrix[:,jX],temp_matrix[:,jY1], color = 'g')
        title = "Actual and Estimated Data, Run#" + str(k)
        plt.title(title)
        plt.show()
        
        
    temp_matrix = tf.convert_to_tensor(temp_matrix, dtype = tf.float64)
    
    tensor = tf.concat([tensor, temp_matrix[None]], axis = 0)
    
    recent_MSE = tensor[-1,0, jMSE]
    prior_MSE = tensor[-2,0, jMSE]
    
    try:
        
        if (abs(recent_MSE - prior_MSE) <= threshold):
            print('Threshold achived')
            break
    except:
        pass
        
        

'''' Compare results
'''
# Analysing the results

last_run = np.array(tensor[-1, :, :])
all_MSE = np.array(tensor[:, 0, jMSE])
all_m = np.array(tensor[:, 0, jm])
all_b = np.array(tensor[:, 0, jb])

# Print the official results from a while back
print("Benchmark m      =", oficial_results[0])
print("Benchmark b      =", oficial_results[1])
print('----------')
# Print last found values
print("Predicted m =",all_m[-1])
print("Predicted b =",all_b[-1])


''' Testing stuff from a tutorial


data_x = np.linspace(1.0, 10.0, 100)[:, np.newaxis]
print(type (data_x))

data_x = np.hstack((np.ones_like(data_x), data_x))

'''

''' Extra Analysis for Fun
'''


#Plot the MSE

lenght = np.array(list(range(1, tensor.shape[0])))
plt.scatter(lenght,tensor[1:,0,jMSE], color = 'r')
title = "MSE values vs iteration" 
plt.title(title)
plt.show()

# Plot the change in m per iteration

def change_in_m(m_over_itteration):
    temp1 = np.array(m_over_itteration)
    temp2 = np.full(len(temp1),0.0)
    for i in range(0, len(temp1) -1 ):
        temp2[i] = (temp1[i+1] - temp1[i])
    return temp2[:i]

temp_Y = change_in_m(tensor[10:1000, 1, jm])  # Axis name recap: [k, i, j]
# Interesting ranges to try: 1:10, 10:5000, 1:1000
temp_X = list(range(0,len(temp_Y)))
plt.scatter(temp_X,temp_Y, color = 'b')
title = "Change of m vs iteration" 
plt.title(title)
plt.show()
        
# Plot the change in b per iteration

def change_in_b(b_over_itteration):
    temp1 = np.array(b_over_itteration)
    temp2 = np.full(len(temp1),0.0)
    for i in range(0, len(temp1) - 1):
        temp2[i] = (temp1[i+1] - temp1[i])
    return temp2[:i]

temp_Y = change_in_b(tensor[1:1000, 1, jb])
temp_X = list(range(0,len(temp_Y)))
plt.scatter(temp_X,temp_Y, color = 'b')
title = "Change of b vs iteration" 
plt.title(title)
plt.show()
        



# Plot the change in b per iteration with a smooth line
# a little missleading but it looks good
from scipy.interpolate import make_interp_spline

temp_Y = np.array(change_in_b(tensor[1:10, 1, jb]))
temp_X = np.array(list(range(0,len(temp_Y))))

X_Y_Spline = make_interp_spline(temp_X, temp_Y)
 
# Returns evenly spaced numbers
# over a specified interval.
X_ = np.linspace(temp_X.min(), temp_X.max(), 1000)
Y_ = X_Y_Spline(X_)
plt.scatter(X_,Y_, color = 'b')
title = "Change of b vs iteration" 
plt.title(title)
plt.show()











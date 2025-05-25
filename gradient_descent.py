"""
Name:

Please:
    - set breakpoints and walk through this lab.
    - verify the dimensions of the arrays
    - verify values with your hand calculations
    - seek to understand, not to just get it done

Lab: Gradient Descent
  - use genFirstDataSet() to get your Cost and GradientDescent functions to work
  - then use genDataSet() to get a random set of data with a given variance (sigma)

ToDo:
    - plot random data (20 values)
    - plot your predicted line given w0 and w1
    - plot the cost curve
    - answer the questions in the lab in the commented area at the end
    - congratulate yourself - you passed a major milestone!

"""
import numpy as np
import matplotlib.pyplot as plt

# Generates a simple test set of data to work with [x0,x1,y]
def genFirstDataSet():
    return np.array([[1, 0, 0],
                     [1, 1, .5],
                     [1, 2, 1],
                     [1, 3, 1.5]])

# Generates a random set of data with a set variance, slope, offset, min and max x
def genDataSet(nElements, sigma, slope, offset, minX, maxX):
    x0 = np.ones(nElements) # x0 = 1
    x1 = np.random.random(nElements) * (maxX - minX) + minX # x1 = random values between min and max
    y = x1 * slope + offset # y values (w1*x1 +w0)
    y += (np.random.random(nElements) * 2 - 1) * sigma # adds variance to y values "real data"

    data = np.column_stack((x0, x1, y)) # combines into single array
    return data


def costFunc(x, weights, y):
    predicted = np.dot(x,weights)
    cost = np.square(predicted - y)
    return np.sum(cost)


# todo: compute the cost using numpy, not for-loops




def gradDesc(x, weights, y):
    n = len(x)
    predicted = np.dot(x,weights)
    cost = predicted - y
    gradient = np.dot(x.T,cost)/n
    return gradient




# todo: compute the gradient descent using numpy, not for-loops

# Call data generation functions
#data = genFirstDataSet()
data = genDataSet(20, 2, 2, 0, 0, 20)

# todo:  using the split function with data to separate x and y
x = data[:,0:2]
y =data[:,2:3]
print(x.shape)
print(y)

# Initialize weights
weights = np.array([0, 1.5]).reshape(2, 1)
# Initialize LR - learning rate,  change to .005 when using generated data set.
LR = 0.001
# Initialize number of iterations
maxIter = 50
# Create array to store costs per iteration
costArray = []

# todo: Write your main loop
"""
Call the cost function, the gradient function, and then update the weights with the given learning rate.   
Store the value of the cost function into an array so you can display the cost function, 
per iteration, after the loop. 
"""
print(weights.shape)
print(x.shape)
print(gradDesc(x,weights,y).shape)
for i in range(maxIter):
    costArray.append(costFunc(x,weights,y))
    weights -= LR*(gradDesc(x,weights,y))

#print(costArray)
# todo: plot the best fit line
'''
  - use np.argmin() to find the index of the minimum value in x
  - use np.argmax() to find the index of the maximum value in x
  - create an array with the min and max x values
  - create an array with the predicted y values: 2 x values * weights 
'''
print("weights:",weights)
minIndex = np.argmin(x)
maxIndex = np.argmax(x)

maxX = x.flatten()[maxIndex]
minX = x.flatten()[minIndex]
xBoundaries = ([minX,maxX])
yBoundaries = ([weights[0] + weights[1]*minX,weights[0] + weights[1]*maxX])
# plot data
fig1 = plt.figure()
ax1 = fig1.add_axes([0.1, 0.1, 0.8, 0.8])
ax1.plot(xBoundaries,yBoundaries, label = "line", color = 'blue')
ax1.plot(x[:, 1], y, 'rx')
ax1.set(title='Data with Best Fit Line', xlabel='X values', ylabel='Y values')
plt.show()


# plot cost
fig2 = plt.figure()
ax2 = fig2.add_axes([0.1, 0.1, 0.8, 0.8])
ax2.plot(range(maxIter), costArray, color='blue')
ax2.set(title='Cost vs. Iterations', xlabel='iterations', ylabel='cost')
plt.show()

# todo: answer analysis questions in the lab here.
'''
Compare and Contrast

1) Increase the learning rate. At what point does the model fail to train?
It works at 0.01, but fails at 0.02.
2) Decrease the learning rate.  What happens to the Cost Function graph?
It takes much more iterations for the cost to get close to 0
3) Display the final weights and compare with the expected values.  What can 
    you change to make your model more accurate?
I could make more iterations of the main loop so that cost would be closer to 0
'''
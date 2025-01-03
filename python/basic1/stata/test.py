


import numpy as np

from matplotlib import pyplot as plt
from sklearn.datasets import make_blobs

#Step 1 - Define our data

#Input data - Of the form [X value, Y value, Bias term]
# X = np.array([
#     [-2,4,-1],
#     [4,1,-1],
#     [1, 6, -1],
#     [2, 4, -1],
#     [6, 2, -1],
# ])
# #Associated output labels - First 2 examples are labeled '-1' and last 3 are labeled '+1'
# y = np.array([-1,-1,1,1,1])
np.random.seed
x,y=make_blobs(n_samples=100,centers=2,cluster_std=0.8) # random_state=0, y是1 OR 0

X=np.append(x,np.ones((100,1)),axis=1)
for d,item in enumerate(y):
    if y[d]==0:
        y[d]=-1

#lets plot these examples on a 2D graph!
plt.figure()
for  d,sample in enumerate(X):
    # Plot the negative samples (the first 2)
    if y[d]==-1:
        plt.scatter(sample[0], sample[1], s=120, marker='_', linewidths=2)
    # Plot the positive samples (the last 3)
    else:
        plt.scatter(sample[0], sample[1], s=120, marker='+', linewidths=2)
#plt.plot([-2,6],[6,0.5])

plt.show(block=False)

# %%
#lets perform stochastic gradient descent to learn the seperating hyperplane between both classes

def svm_sgd_plot(X, Y):
  #Initialize our SVMs weight vector with zeros (3 values)
  w = np.zeros(len(X[0]))
  #w=np.zeros(X.shape[1])
  #The learning rate
  eta = 1
  #how many iterations to train for
  epochs = 1000#100000
  #store misclassifications so we can plot how they change over time
  errors = []

  #training part, gradient descent part
  for epoch in range(1,epochs):
      error = 0
      for i, x in enumerate(X):
          #misclassification
          if (Y[i]*np.dot(X[i], w)) < 1:
              #misclassified update for ours weights
              w = w + eta * ( (X[i] * Y[i]) + (-2  *(1/epoch)* w) )
              error += 1
          else:
              #correct classification, update our weights
              w = w + eta * (-2  *(1/epoch)* w)
      errors.append(error/100)
        
    
  # #lets plot the rate of classification errors during training for our SVM
  # #plt.plot(errors, '|')
  # plt.figure()
  
  # plt.plot(np.array(errors))
  # #plt.ylim(0.5,1.5)
  # #plt.axes().set_yticklabels([])
  # plt.xlabel('Epoch')
  # plt.ylabel('Misclassified')
  
  # plt.show(block=False)
    
  return w


w = svm_sgd_plot(X,y)
 
#they decrease over time! Our SVM is learning the optimal hyperplane

# Add our test samples
def ybyx(ax):
    #ax*w[0]+ay*w[1]+-1*w[2]=0
    ay=-(ax*w[0]+w[2])/w[1]
    return ay
min2=np.min(X,0)[0:2]
max2=np.max(X,0)[0:2]
px=[min2[0],max2[0]]
py=[]
for d,v in enumerate(px):
    py.append(ybyx(v))
plt.plot(px,py)
plt.show()


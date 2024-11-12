---
title: Linear_Regression_with_Synthetic_Data
description: docker log
weight: 300
---
# Colabs

Machine Learning Crash Course uses Colaboratories (Colabs) for all programming exercises. Colab is Google's implementation of [Jupyter Notebook](https://jupyter.org/). For more information about Colabs and how to use them, go to [Welcome to Colaboratory](https://research.google.com/colaboratory).

# Simple Linear Regression with Synthetic Data

In this first Colab, you'll explore linear regression with a simple database.

## Learning objectives:

After doing this exercise, you'll know how to do the following:

  * Run Colabs.
  * Tune the following [hyperparameters](https://developers.google.com/machine-learning/glossary/#hyperparameter):
    * [learning rate](https://developers.google.com/machine-learning/glossary/#learning_rate)
    * number of [epochs](https://developers.google.com/machine-learning/glossary/#epoch)
    * [batch size](https://developers.google.com/machine-learning/glossary/#batch_size)
  * Interpret different kinds of [loss curves](https://developers.google.com/machine-learning/glossary/#loss_curve).

## Import relevant modules

The following cell imports the packages that the program requires:


```python
import numpy as np
import pandas as pd
import tensorflow as tf
#import tensorflow._api.v2.compat.v1 as tf
from matplotlib import pyplot as plt
```

## Define functions that build and train a model

The following code defines two functions:

  * `build_model(my_learning_rate)`, which builds an empty model.
  * `train_model(model, feature, label, epochs)`, which trains the model from the examples (feature and label) you pass.

Since you don't need to understand model building code right now, we've hidden this code cell.  You may optionally double-click the headline to explore this code.


```python
#@title Define the functions that build and train a model
def build_model(my_learning_rate):
  """Create and compile a simple linear regression model."""
  # Most simple tf.keras models are sequential.
  # A sequential model contains one or more layers.
  model = tf.keras.models.Sequential()

  # Describe the topography of the model.
  # The topography of a simple linear regression model
  # is a single node in a single layer.
  model.add(tf.keras.layers.Dense(units=1,
                                  input_shape=(1,)))

  # Compile the model topography into code that
  # TensorFlow can efficiently execute. Configure
  # training to minimize the model's mean squared error.
  model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=my_learning_rate),
                loss="mean_squared_error",
                metrics=[tf.keras.metrics.RootMeanSquaredError()])

  return model


def train_model(model, feature, label, epochs, batch_size):
  """Train the model by feeding it data."""

  # Feed the feature values and the label values to the
  # model. The model will train for the specified number
  # of epochs, gradually learning how the feature values
  # relate to the label values.
  history = model.fit(x=feature,
                      y=label,
                      batch_size=batch_size,
                      epochs=epochs)

  # Gather the trained model's weight and bias.
  trained_weight = model.get_weights()[0][0]
  trained_bias = model.get_weights()[1]

  # The list of epochs is stored separately from the
  # rest of history.
  epochs = history.epoch

  # Gather the history (a snapshot) of each epoch.
  hist = pd.DataFrame(history.history)

  # Specifically gather the model's root mean
  # squared error at each epoch.
  rmse = hist["root_mean_squared_error"]

  return trained_weight, trained_bias, epochs, rmse

print("Defined build_model and train_model")
```

    Defined build_model and train_model
    

## Define plotting functions

We're using a popular Python library called [Matplotlib](https://developers.google.com/machine-learning/glossary/#matplotlib) to create the following two plots:

*  a plot of the feature values vs. the label values, and a line showing the output of the trained model.
*  a [loss curve](https://developers.google.com/machine-learning/glossary/#loss_curve).

We hid the following code cell because learning Matplotlib is not relevant to the learning objectives. Regardless, you must still run all hidden code cells.


```python
#@title Define the plotting functions
def plot_the_model(trained_weight, trained_bias, feature, label):
  """Plot the trained model against the training feature and label."""

  # Label the axes.
  plt.xlabel("feature")
  plt.ylabel("label")

  # Plot the feature values vs. label values.
  plt.scatter(feature, label)

  # Create a red line representing the model. The red line starts
  # at coordinates (x0, y0) and ends at coordinates (x1, y1).
  x0 = 0
  y0 = trained_bias
  x1 = feature[-1]
  y1 = trained_bias + (trained_weight * x1)
  plt.plot([x0, x1], [y0, y1], c='r')

  # Render the scatter plot and the red line.
  plt.show()

def plot_the_loss_curve(epochs, rmse):
  """Plot the loss curve, which shows loss vs. epoch."""

  plt.figure()
  plt.xlabel("Epoch")
  plt.ylabel("Root Mean Squared Error")

  plt.plot(epochs, rmse, label="Loss")
  plt.legend()
  plt.ylim([rmse.min()*0.97, rmse.max()])
  plt.show()

print("Defined the plot_the_model and plot_the_loss_curve functions.")
```

    Defined the plot_the_model and plot_the_loss_curve functions.
    

## Define the dataset

The dataset consists of 12 [examples](https://developers.google.com/machine-learning/glossary/#example). Each example consists of one [feature](https://developers.google.com/machine-learning/glossary/#feature) and one [label](https://developers.google.com/machine-learning/glossary/#label).



```python
my_feature = np.array([1.0, 2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0, 10.0, 11.0, 12.0])
my_label   = np.array([5.0, 8.8,  9.6, 14.2, 18.8, 19.5, 21.4, 26.8, 28.9, 32.0, 33.8, 38.2])
```

## Specify the hyperparameters

The hyperparameters in this Colab are as follows:

  * [learning rate](https://developers.google.com/machine-learning/glossary/#learning_rate)
  * [epochs](https://developers.google.com/machine-learning/glossary/#epoch)
  * [batch_size](https://developers.google.com/machine-learning/glossary/#batch_size)

The following code cell initializes these hyperparameters and then invokes the functions that build and train the model.


```python
learning_rate=0.01
epochs=10
my_batch_size=12

my_model = build_model(learning_rate)
trained_weight, trained_bias, epochs, rmse = train_model(my_model, my_feature,
                                                         my_label, epochs,
                                                         my_batch_size)
plot_the_model(trained_weight, trained_bias, my_feature, my_label)
plot_the_loss_curve(epochs, rmse)
```

    Epoch 1/10
    

    c:\Users\linchao\AppData\Local\Programs\Python\Python312\Lib\site-packages\keras\src\layers\core\dense.py:87: UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.
      super().__init__(activity_regularizer=activity_regularizer, **kwargs)
    

    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 460ms/step - loss: 408.3693 - root_mean_squared_error: 20.2081
    Epoch 2/10
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 87ms/step - loss: 397.8968 - root_mean_squared_error: 19.9473
    Epoch 3/10
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 35ms/step - loss: 390.4308 - root_mean_squared_error: 19.7593
    Epoch 4/10
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 28ms/step - loss: 384.2571 - root_mean_squared_error: 19.6025
    Epoch 5/10
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 73ms/step - loss: 378.8333 - root_mean_squared_error: 19.4636
    Epoch 6/10
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 26ms/step - loss: 373.9083 - root_mean_squared_error: 19.3367
    Epoch 7/10
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 20ms/step - loss: 369.3424 - root_mean_squared_error: 19.2183
    Epoch 8/10
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 77ms/step - loss: 365.0490 - root_mean_squared_error: 19.1063
    Epoch 9/10
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 23ms/step - loss: 360.9698 - root_mean_squared_error: 18.9992
    Epoch 10/10
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 20ms/step - loss: 357.0640 - root_mean_squared_error: 18.8961
    


    
![png](output_12_3.png)
    



    
![png](output_12_4.png)
    


## Task 1: Examine the graphs

Examine the top graph. The blue dots identify the actual data; the red line identifies the output of the trained model. Ideally, the red line should align nicely with the blue dots.  Does it?  Probably not.

A certain amount of randomness plays into training a model, so you'll get somewhat different results every time you train.  That said, unless you are an extremely lucky person, the red line probably *doesn't* align nicely with the blue dots.  

Examine the bottom graph, which shows the loss curve. Notice that the loss curve decreases but doesn't flatten out, which is a sign that the model hasn't trained sufficiently.

## Task 2: Increase the number of epochs

Training loss should steadily decrease, steeply at first, and then more slowly. Eventually, training loss should stay steady (zero slope or nearly zero slope), which indicates that training has [converged](http://developers.google.com/machine-learning/glossary/#convergence).

In Task 1, the training loss did not converge. One possible solution is to train for more epochs.  Your task is to increase the number of epochs sufficiently to get the model to converge. However, it is inefficient to train past convergence, so don't just set the number of epochs to an arbitrarily high value.

Examine the loss curve. Does the model converge?


```python
learning_rate=0.01
epochs= 20   # Replace ? with an integer.
my_batch_size=12

my_model = build_model(learning_rate)
trained_weight, trained_bias, epochs, rmse = train_model(my_model, my_feature,
                                                        my_label, epochs,
                                                        my_batch_size)
plot_the_model(trained_weight, trained_bias, my_feature, my_label)
plot_the_loss_curve(epochs, rmse)
```

    Epoch 1/20
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 360ms/step - loss: 532.0615 - root_mean_squared_error: 23.0665
    Epoch 2/20
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 24ms/step - loss: 520.0957 - root_mean_squared_error: 22.8056
    Epoch 3/20
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 38ms/step - loss: 511.5465 - root_mean_squared_error: 22.6174
    Epoch 4/20
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 418ms/step - loss: 504.4659 - root_mean_squared_error: 22.4603
    Epoch 5/20
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 42ms/step - loss: 498.2372 - root_mean_squared_error: 22.3212
    Epoch 6/20
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 48ms/step - loss: 492.5747 - root_mean_squared_error: 22.1940
    Epoch 7/20
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 62ms/step - loss: 487.3195 - root_mean_squared_error: 22.0753
    Epoch 8/20
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 56ms/step - loss: 482.3729 - root_mean_squared_error: 21.9630
    Epoch 9/20
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 94ms/step - loss: 477.6687 - root_mean_squared_error: 21.8556
    Epoch 10/20
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 89ms/step - loss: 473.1603 - root_mean_squared_error: 21.7522
    Epoch 11/20
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 69ms/step - loss: 468.8136 - root_mean_squared_error: 21.6521
    Epoch 12/20
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 182ms/step - loss: 464.6024 - root_mean_squared_error: 21.5546
    Epoch 13/20
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 63ms/step - loss: 460.5070 - root_mean_squared_error: 21.4594
    Epoch 14/20
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 81ms/step - loss: 456.5111 - root_mean_squared_error: 21.3661
    Epoch 15/20
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 63ms/step - loss: 452.6021 - root_mean_squared_error: 21.2744
    Epoch 16/20
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 34ms/step - loss: 448.7694 - root_mean_squared_error: 21.1842
    Epoch 17/20
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 56ms/step - loss: 445.0043 - root_mean_squared_error: 21.0951
    Epoch 18/20
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 62ms/step - loss: 441.2996 - root_mean_squared_error: 21.0071
    Epoch 19/20
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 54ms/step - loss: 437.6492 - root_mean_squared_error: 20.9201
    Epoch 20/20
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 34ms/step - loss: 434.0479 - root_mean_squared_error: 20.8338
    


    
![png](output_15_1.png)
    



    
![png](output_15_2.png)
    



```python
#@title Double-click to view a possible solution
learning_rate=0.01
epochs=450
my_batch_size=12

my_model = build_model(learning_rate)
trained_weight, trained_bias, epochs, rmse = train_model(my_model, my_feature,
                                                         my_label, epochs,
                                                         my_batch_size)
plot_the_model(trained_weight, trained_bias, my_feature, my_label)
plot_the_loss_curve(epochs, rmse)

# The loss curve suggests that the model does converge.
```

    Epoch 1/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 283ms/step - loss: 286.7389 - root_mean_squared_error: 16.9334
    Epoch 2/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 24ms/step - loss: 277.9785 - root_mean_squared_error: 16.6727
    Epoch 3/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 22ms/step - loss: 271.7545 - root_mean_squared_error: 16.4850
    Epoch 4/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 64ms/step - loss: 266.6205 - root_mean_squared_error: 16.3285
    Epoch 5/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 20ms/step - loss: 262.1194 - root_mean_squared_error: 16.1901
    Epoch 6/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 23ms/step - loss: 258.0399 - root_mean_squared_error: 16.0636
    Epoch 7/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 64ms/step - loss: 254.2642 - root_mean_squared_error: 15.9457
    Epoch 8/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 25ms/step - loss: 250.7194 - root_mean_squared_error: 15.8341
    Epoch 9/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 24ms/step - loss: 247.3566 - root_mean_squared_error: 15.7276
    Epoch 10/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 22ms/step - loss: 244.1414 - root_mean_squared_error: 15.6250
    Epoch 11/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 20ms/step - loss: 241.0485 - root_mean_squared_error: 15.5257
    Epoch 12/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 75ms/step - loss: 238.0587 - root_mean_squared_error: 15.4291
    Epoch 13/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 22ms/step - loss: 235.1572 - root_mean_squared_error: 15.3348
    Epoch 14/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 20ms/step - loss: 232.3323 - root_mean_squared_error: 15.2425
    Epoch 15/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 87ms/step - loss: 229.5745 - root_mean_squared_error: 15.1517
    Epoch 16/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 30ms/step - loss: 226.8761 - root_mean_squared_error: 15.0624
    Epoch 17/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 30ms/step - loss: 224.2305 - root_mean_squared_error: 14.9743
    Epoch 18/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 89ms/step - loss: 221.6327 - root_mean_squared_error: 14.8873
    Epoch 19/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 35ms/step - loss: 219.0780 - root_mean_squared_error: 14.8013
    Epoch 20/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 36ms/step - loss: 216.5626 - root_mean_squared_error: 14.7161
    Epoch 21/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 65ms/step - loss: 214.0833 - root_mean_squared_error: 14.6316
    Epoch 22/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 31ms/step - loss: 211.6372 - root_mean_squared_error: 14.5478
    Epoch 23/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 31ms/step - loss: 209.2220 - root_mean_squared_error: 14.4645
    Epoch 24/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 35ms/step - loss: 206.8356 - root_mean_squared_error: 14.3818
    Epoch 25/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 28ms/step - loss: 204.4761 - root_mean_squared_error: 14.2995
    Epoch 26/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 85ms/step - loss: 202.1420 - root_mean_squared_error: 14.2177
    Epoch 27/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 35ms/step - loss: 199.8319 - root_mean_squared_error: 14.1362
    Epoch 28/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 68ms/step - loss: 197.5444 - root_mean_squared_error: 14.0551
    Epoch 29/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 28ms/step - loss: 195.2787 - root_mean_squared_error: 13.9742
    Epoch 30/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 24ms/step - loss: 193.0337 - root_mean_squared_error: 13.8937
    Epoch 31/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 68ms/step - loss: 190.8085 - root_mean_squared_error: 13.8133
    Epoch 32/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 31ms/step - loss: 188.6024 - root_mean_squared_error: 13.7333
    Epoch 33/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 36ms/step - loss: 186.4147 - root_mean_squared_error: 13.6534
    Epoch 34/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 69ms/step - loss: 184.2449 - root_mean_squared_error: 13.5737
    Epoch 35/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 34ms/step - loss: 182.0925 - root_mean_squared_error: 13.4942
    Epoch 36/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 31ms/step - loss: 179.9569 - root_mean_squared_error: 13.4148
    Epoch 37/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 287ms/step - loss: 177.8378 - root_mean_squared_error: 13.3356
    Epoch 38/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 33ms/step - loss: 175.7347 - root_mean_squared_error: 13.2565
    Epoch 39/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 48ms/step - loss: 173.6473 - root_mean_squared_error: 13.1775
    Epoch 40/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 31ms/step - loss: 171.5754 - root_mean_squared_error: 13.0987
    Epoch 41/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 31ms/step - loss: 169.5186 - root_mean_squared_error: 13.0199
    Epoch 42/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 26ms/step - loss: 167.4768 - root_mean_squared_error: 12.9413
    Epoch 43/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 25ms/step - loss: 165.4497 - root_mean_squared_error: 12.8627
    Epoch 44/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 62ms/step - loss: 163.4371 - root_mean_squared_error: 12.7843
    Epoch 45/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 25ms/step - loss: 161.4388 - root_mean_squared_error: 12.7059
    Epoch 46/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 18ms/step - loss: 159.4547 - root_mean_squared_error: 12.6275
    Epoch 47/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 76ms/step - loss: 157.4846 - root_mean_squared_error: 12.5493
    Epoch 48/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 24ms/step - loss: 155.5285 - root_mean_squared_error: 12.4711
    Epoch 49/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 20ms/step - loss: 153.5862 - root_mean_squared_error: 12.3930
    Epoch 50/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 75ms/step - loss: 151.6576 - root_mean_squared_error: 12.3149
    Epoch 51/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 25ms/step - loss: 149.7426 - root_mean_squared_error: 12.2369
    Epoch 52/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 18ms/step - loss: 147.8410 - root_mean_squared_error: 12.1590
    Epoch 53/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 81ms/step - loss: 145.9530 - root_mean_squared_error: 12.0811
    Epoch 54/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 35ms/step - loss: 144.0783 - root_mean_squared_error: 12.0033
    Epoch 55/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 36ms/step - loss: 142.2169 - root_mean_squared_error: 11.9255
    Epoch 56/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 77ms/step - loss: 140.3688 - root_mean_squared_error: 11.8477
    Epoch 57/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 33ms/step - loss: 138.5339 - root_mean_squared_error: 11.7700
    Epoch 58/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 32ms/step - loss: 136.7121 - root_mean_squared_error: 11.6924
    Epoch 59/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 48ms/step - loss: 134.9035 - root_mean_squared_error: 11.6148
    Epoch 60/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 23ms/step - loss: 133.1079 - root_mean_squared_error: 11.5372
    Epoch 61/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 38ms/step - loss: 131.3253 - root_mean_squared_error: 11.4597
    Epoch 62/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 29ms/step - loss: 129.5558 - root_mean_squared_error: 11.3823
    Epoch 63/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 73ms/step - loss: 127.7992 - root_mean_squared_error: 11.3048
    Epoch 64/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 29ms/step - loss: 126.0556 - root_mean_squared_error: 11.2274
    Epoch 65/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 29ms/step - loss: 124.3249 - root_mean_squared_error: 11.1501
    Epoch 66/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 79ms/step - loss: 122.6071 - root_mean_squared_error: 11.0728
    Epoch 67/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 30ms/step - loss: 120.9021 - root_mean_squared_error: 10.9956
    Epoch 68/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 25ms/step - loss: 119.2100 - root_mean_squared_error: 10.9183
    Epoch 69/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 78ms/step - loss: 117.5308 - root_mean_squared_error: 10.8412
    Epoch 70/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 23ms/step - loss: 115.8643 - root_mean_squared_error: 10.7640
    Epoch 71/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 23ms/step - loss: 114.2107 - root_mean_squared_error: 10.6869
    Epoch 72/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 22ms/step - loss: 112.5698 - root_mean_squared_error: 10.6099
    Epoch 73/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 25ms/step - loss: 110.9417 - root_mean_squared_error: 10.5329
    Epoch 74/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 98ms/step - loss: 109.3264 - root_mean_squared_error: 10.4559
    Epoch 75/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 37ms/step - loss: 107.7237 - root_mean_squared_error: 10.3790
    Epoch 76/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 91ms/step - loss: 106.1339 - root_mean_squared_error: 10.3021
    Epoch 77/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 32ms/step - loss: 104.5567 - root_mean_squared_error: 10.2253
    Epoch 78/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 38ms/step - loss: 102.9922 - root_mean_squared_error: 10.1485
    Epoch 79/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 77ms/step - loss: 101.4404 - root_mean_squared_error: 10.0718
    Epoch 80/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 46ms/step - loss: 99.9013 - root_mean_squared_error: 9.9951
    Epoch 81/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 41ms/step - loss: 98.3748 - root_mean_squared_error: 9.9184
    Epoch 82/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 70ms/step - loss: 96.8611 - root_mean_squared_error: 9.8418
    Epoch 83/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 231ms/step - loss: 95.3599 - root_mean_squared_error: 9.7652
    Epoch 84/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 18ms/step - loss: 93.8714 - root_mean_squared_error: 9.6887
    Epoch 85/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 32ms/step - loss: 92.3955 - root_mean_squared_error: 9.6123
    Epoch 86/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 76ms/step - loss: 90.9323 - root_mean_squared_error: 9.5358
    Epoch 87/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 24ms/step - loss: 89.4816 - root_mean_squared_error: 9.4595
    Epoch 88/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 20ms/step - loss: 88.0435 - root_mean_squared_error: 9.3832
    Epoch 89/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 78ms/step - loss: 86.6180 - root_mean_squared_error: 9.3069
    Epoch 90/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 30ms/step - loss: 85.2051 - root_mean_squared_error: 9.2307
    Epoch 91/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 33ms/step - loss: 83.8048 - root_mean_squared_error: 9.1545
    Epoch 92/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 218ms/step - loss: 82.4170 - root_mean_squared_error: 9.0784
    Epoch 93/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 36ms/step - loss: 81.0417 - root_mean_squared_error: 9.0023
    Epoch 94/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 44ms/step - loss: 79.6790 - root_mean_squared_error: 8.9263
    Epoch 95/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 138ms/step - loss: 78.3288 - root_mean_squared_error: 8.8504
    Epoch 96/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 62ms/step - loss: 76.9911 - root_mean_squared_error: 8.7745
    Epoch 97/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 143ms/step - loss: 75.6660 - root_mean_squared_error: 8.6986
    Epoch 98/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 48ms/step - loss: 74.3533 - root_mean_squared_error: 8.6228
    Epoch 99/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 264ms/step - loss: 73.0531 - root_mean_squared_error: 8.5471
    Epoch 100/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 78ms/step - loss: 71.7654 - root_mean_squared_error: 8.4714
    Epoch 101/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 64ms/step - loss: 70.4901 - root_mean_squared_error: 8.3958
    Epoch 102/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 54ms/step - loss: 69.2273 - root_mean_squared_error: 8.3203
    Epoch 103/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 46ms/step - loss: 67.9769 - root_mean_squared_error: 8.2448
    Epoch 104/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 310ms/step - loss: 66.7390 - root_mean_squared_error: 8.1694
    Epoch 105/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 58ms/step - loss: 65.5134 - root_mean_squared_error: 8.0940
    Epoch 106/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 89ms/step - loss: 64.3003 - root_mean_squared_error: 8.0187
    Epoch 107/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 40ms/step - loss: 63.0996 - root_mean_squared_error: 7.9435
    Epoch 108/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 52ms/step - loss: 61.9112 - root_mean_squared_error: 7.8684
    Epoch 109/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 49ms/step - loss: 60.7353 - root_mean_squared_error: 7.7933
    Epoch 110/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 57ms/step - loss: 59.5717 - root_mean_squared_error: 7.7183
    Epoch 111/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 43ms/step - loss: 58.4204 - root_mean_squared_error: 7.6433
    Epoch 112/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 29ms/step - loss: 57.2815 - root_mean_squared_error: 7.5685
    Epoch 113/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 73ms/step - loss: 56.1549 - root_mean_squared_error: 7.4937
    Epoch 114/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 59ms/step - loss: 55.0406 - root_mean_squared_error: 7.4189
    Epoch 115/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 157ms/step - loss: 53.9386 - root_mean_squared_error: 7.3443
    Epoch 116/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 59ms/step - loss: 52.8489 - root_mean_squared_error: 7.2697
    Epoch 117/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 43ms/step - loss: 51.7715 - root_mean_squared_error: 7.1952
    Epoch 118/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 68ms/step - loss: 50.7063 - root_mean_squared_error: 7.1208
    Epoch 119/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 48ms/step - loss: 49.6534 - root_mean_squared_error: 7.0465
    Epoch 120/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 44ms/step - loss: 48.6127 - root_mean_squared_error: 6.9723
    Epoch 121/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 40ms/step - loss: 47.5842 - root_mean_squared_error: 6.8981
    Epoch 122/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 41ms/step - loss: 46.5679 - root_mean_squared_error: 6.8241
    Epoch 123/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 76ms/step - loss: 45.5638 - root_mean_squared_error: 6.7501
    Epoch 124/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 56ms/step - loss: 44.5719 - root_mean_squared_error: 6.6762
    Epoch 125/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 45ms/step - loss: 43.5921 - root_mean_squared_error: 6.6024
    Epoch 126/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 63ms/step - loss: 42.6245 - root_mean_squared_error: 6.5287
    Epoch 127/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 43ms/step - loss: 41.6690 - root_mean_squared_error: 6.4552
    Epoch 128/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 57ms/step - loss: 40.7256 - root_mean_squared_error: 6.3817
    Epoch 129/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 56ms/step - loss: 39.7942 - root_mean_squared_error: 6.3083
    Epoch 130/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 58ms/step - loss: 38.8750 - root_mean_squared_error: 6.2350
    Epoch 131/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 163ms/step - loss: 37.9678 - root_mean_squared_error: 6.1618
    Epoch 132/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 52ms/step - loss: 37.0726 - root_mean_squared_error: 6.0887
    Epoch 133/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 48ms/step - loss: 36.1895 - root_mean_squared_error: 6.0158
    Epoch 134/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 52ms/step - loss: 35.3183 - root_mean_squared_error: 5.9429
    Epoch 135/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 57ms/step - loss: 34.4592 - root_mean_squared_error: 5.8702
    Epoch 136/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 277ms/step - loss: 33.6119 - root_mean_squared_error: 5.7976
    Epoch 137/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 80ms/step - loss: 32.7767 - root_mean_squared_error: 5.7251
    Epoch 138/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 63ms/step - loss: 31.9533 - root_mean_squared_error: 5.6527
    Epoch 139/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 95ms/step - loss: 31.1419 - root_mean_squared_error: 5.5805
    Epoch 140/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 56ms/step - loss: 30.3423 - root_mean_squared_error: 5.5084
    Epoch 141/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 51ms/step - loss: 29.5546 - root_mean_squared_error: 5.4364
    Epoch 142/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 52ms/step - loss: 28.7787 - root_mean_squared_error: 5.3646
    Epoch 143/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 84ms/step - loss: 28.0146 - root_mean_squared_error: 5.2929
    Epoch 144/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 96ms/step - loss: 27.2623 - root_mean_squared_error: 5.2213
    Epoch 145/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 63ms/step - loss: 26.5218 - root_mean_squared_error: 5.1499
    Epoch 146/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 53ms/step - loss: 25.7930 - root_mean_squared_error: 5.0787
    Epoch 147/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 48ms/step - loss: 25.0759 - root_mean_squared_error: 5.0076
    Epoch 148/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 149ms/step - loss: 24.3705 - root_mean_squared_error: 4.9367
    Epoch 149/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 55ms/step - loss: 23.6768 - root_mean_squared_error: 4.8659
    Epoch 150/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 57ms/step - loss: 22.9947 - root_mean_squared_error: 4.7953
    Epoch 151/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 57ms/step - loss: 22.3242 - root_mean_squared_error: 4.7248
    Epoch 152/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 42ms/step - loss: 21.6653 - root_mean_squared_error: 4.6546
    Epoch 153/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 56ms/step - loss: 21.0179 - root_mean_squared_error: 4.5845
    Epoch 154/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 61ms/step - loss: 20.3820 - root_mean_squared_error: 4.5146
    Epoch 155/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 62ms/step - loss: 19.7577 - root_mean_squared_error: 4.4450
    Epoch 156/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 125ms/step - loss: 19.1447 - root_mean_squared_error: 4.3755
    Epoch 157/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 77ms/step - loss: 18.5433 - root_mean_squared_error: 4.3062
    Epoch 158/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 64ms/step - loss: 17.9531 - root_mean_squared_error: 4.2371
    Epoch 159/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 43ms/step - loss: 17.3744 - root_mean_squared_error: 4.1683
    Epoch 160/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 47ms/step - loss: 16.8070 - root_mean_squared_error: 4.0996
    Epoch 161/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 41ms/step - loss: 16.2508 - root_mean_squared_error: 4.0312
    Epoch 162/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 341ms/step - loss: 15.7059 - root_mean_squared_error: 3.9631
    Epoch 163/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 35ms/step - loss: 15.1722 - root_mean_squared_error: 3.8952
    Epoch 164/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 20ms/step - loss: 14.6497 - root_mean_squared_error: 3.8275
    Epoch 165/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 86ms/step - loss: 14.1383 - root_mean_squared_error: 3.7601
    Epoch 166/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 28ms/step - loss: 13.6379 - root_mean_squared_error: 3.6930
    Epoch 167/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 21ms/step - loss: 13.1487 - root_mean_squared_error: 3.6261
    Epoch 168/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 67ms/step - loss: 12.6704 - root_mean_squared_error: 3.5595
    Epoch 169/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 129ms/step - loss: 12.2031 - root_mean_squared_error: 3.4933
    Epoch 170/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 83ms/step - loss: 11.7466 - root_mean_squared_error: 3.4273
    Epoch 171/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 46ms/step - loss: 11.3011 - root_mean_squared_error: 3.3617
    Epoch 172/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 177ms/step - loss: 10.8663 - root_mean_squared_error: 3.2964
    Epoch 173/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 33ms/step - loss: 10.4423 - root_mean_squared_error: 3.2315
    Epoch 174/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 48ms/step - loss: 10.0290 - root_mean_squared_error: 3.1669
    Epoch 175/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 63ms/step - loss: 9.6263 - root_mean_squared_error: 3.1026
    Epoch 176/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 49ms/step - loss: 9.2342 - root_mean_squared_error: 3.0388
    Epoch 177/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 33ms/step - loss: 8.8526 - root_mean_squared_error: 2.9753
    Epoch 178/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 77ms/step - loss: 8.4815 - root_mean_squared_error: 2.9123
    Epoch 179/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 31ms/step - loss: 8.1209 - root_mean_squared_error: 2.8497
    Epoch 180/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 44ms/step - loss: 7.7705 - root_mean_squared_error: 2.7876
    Epoch 181/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 65ms/step - loss: 7.4304 - root_mean_squared_error: 2.7259
    Epoch 182/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 30ms/step - loss: 7.1006 - root_mean_squared_error: 2.6647
    Epoch 183/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 26ms/step - loss: 6.7808 - root_mean_squared_error: 2.6040
    Epoch 184/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 28ms/step - loss: 6.4711 - root_mean_squared_error: 2.5438
    Epoch 185/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 225ms/step - loss: 6.1714 - root_mean_squared_error: 2.4842
    Epoch 186/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 21ms/step - loss: 5.8816 - root_mean_squared_error: 2.4252
    Epoch 187/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 40ms/step - loss: 5.6016 - root_mean_squared_error: 2.3668
    Epoch 188/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 161ms/step - loss: 5.3313 - root_mean_squared_error: 2.3090
    Epoch 189/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 85ms/step - loss: 5.0706 - root_mean_squared_error: 2.2518
    Epoch 190/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 51ms/step - loss: 4.8195 - root_mean_squared_error: 2.1953
    Epoch 191/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 30ms/step - loss: 4.5778 - root_mean_squared_error: 2.1396
    Epoch 192/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 77ms/step - loss: 4.3455 - root_mean_squared_error: 2.0846
    Epoch 193/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 25ms/step - loss: 4.1224 - root_mean_squared_error: 2.0304
    Epoch 194/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 25ms/step - loss: 3.9085 - root_mean_squared_error: 1.9770
    Epoch 195/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 81ms/step - loss: 3.7035 - root_mean_squared_error: 1.9245
    Epoch 196/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 27ms/step - loss: 3.5075 - root_mean_squared_error: 1.8728
    Epoch 197/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 24ms/step - loss: 3.3202 - root_mean_squared_error: 1.8221
    Epoch 198/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 65ms/step - loss: 3.1416 - root_mean_squared_error: 1.7724
    Epoch 199/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 29ms/step - loss: 2.9714 - root_mean_squared_error: 1.7238
    Epoch 200/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 28ms/step - loss: 2.8097 - root_mean_squared_error: 1.6762
    Epoch 201/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 21ms/step - loss: 2.6562 - root_mean_squared_error: 1.6298
    Epoch 202/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 21ms/step - loss: 2.5108 - root_mean_squared_error: 1.5845
    Epoch 203/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 80ms/step - loss: 2.3733 - root_mean_squared_error: 1.5405
    Epoch 204/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 34ms/step - loss: 2.2435 - root_mean_squared_error: 1.4978
    Epoch 205/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 23ms/step - loss: 2.1214 - root_mean_squared_error: 1.4565
    Epoch 206/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 226ms/step - loss: 2.0066 - root_mean_squared_error: 1.4166
    Epoch 207/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 39ms/step - loss: 1.8991 - root_mean_squared_error: 1.3781
    Epoch 208/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 47ms/step - loss: 1.7987 - root_mean_squared_error: 1.3412
    Epoch 209/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 40ms/step - loss: 1.7051 - root_mean_squared_error: 1.3058
    Epoch 210/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 60ms/step - loss: 1.6181 - root_mean_squared_error: 1.2721
    Epoch 211/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 42ms/step - loss: 1.5376 - root_mean_squared_error: 1.2400
    Epoch 212/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 36ms/step - loss: 1.4633 - root_mean_squared_error: 1.2097
    Epoch 213/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 82ms/step - loss: 1.3950 - root_mean_squared_error: 1.1811
    Epoch 214/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 44ms/step - loss: 1.3324 - root_mean_squared_error: 1.1543
    Epoch 215/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 92ms/step - loss: 1.2753 - root_mean_squared_error: 1.1293
    Epoch 216/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 47ms/step - loss: 1.2235 - root_mean_squared_error: 1.1061
    Epoch 217/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 32ms/step - loss: 1.1767 - root_mean_squared_error: 1.0847
    Epoch 218/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 62ms/step - loss: 1.1346 - root_mean_squared_error: 1.0652
    Epoch 219/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 37ms/step - loss: 1.0970 - root_mean_squared_error: 1.0474
    Epoch 220/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 31ms/step - loss: 1.0636 - root_mean_squared_error: 1.0313
    Epoch 221/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 82ms/step - loss: 1.0341 - root_mean_squared_error: 1.0169
    Epoch 222/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 28ms/step - loss: 1.0082 - root_mean_squared_error: 1.0041
    Epoch 223/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 23ms/step - loss: 0.9857 - root_mean_squared_error: 0.9928
    Epoch 224/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 65ms/step - loss: 0.9663 - root_mean_squared_error: 0.9830
    Epoch 225/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 29ms/step - loss: 0.9497 - root_mean_squared_error: 0.9745
    Epoch 226/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 28ms/step - loss: 0.9356 - root_mean_squared_error: 0.9673
    Epoch 227/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 20ms/step - loss: 0.9238 - root_mean_squared_error: 0.9612
    Epoch 228/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 24ms/step - loss: 0.9140 - root_mean_squared_error: 0.9561
    Epoch 229/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 74ms/step - loss: 0.9060 - root_mean_squared_error: 0.9518
    Epoch 230/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 28ms/step - loss: 0.8995 - root_mean_squared_error: 0.9484
    Epoch 231/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 19ms/step - loss: 0.8943 - root_mean_squared_error: 0.9457
    Epoch 232/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 225ms/step - loss: 0.8902 - root_mean_squared_error: 0.9435
    Epoch 233/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 21ms/step - loss: 0.8870 - root_mean_squared_error: 0.9418
    Epoch 234/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 22ms/step - loss: 0.8846 - root_mean_squared_error: 0.9405
    Epoch 235/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 82ms/step - loss: 0.8828 - root_mean_squared_error: 0.9396
    Epoch 236/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 29ms/step - loss: 0.8814 - root_mean_squared_error: 0.9388
    Epoch 237/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 27ms/step - loss: 0.8804 - root_mean_squared_error: 0.9383
    Epoch 238/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 60ms/step - loss: 0.8797 - root_mean_squared_error: 0.9379
    Epoch 239/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 35ms/step - loss: 0.8792 - root_mean_squared_error: 0.9377
    Epoch 240/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 26ms/step - loss: 0.8789 - root_mean_squared_error: 0.9375
    Epoch 241/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 58ms/step - loss: 0.8787 - root_mean_squared_error: 0.9374
    Epoch 242/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 26ms/step - loss: 0.8785 - root_mean_squared_error: 0.9373
    Epoch 243/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 19ms/step - loss: 0.8784 - root_mean_squared_error: 0.9372
    Epoch 244/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 75ms/step - loss: 0.8783 - root_mean_squared_error: 0.9372
    Epoch 245/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 28ms/step - loss: 0.8782 - root_mean_squared_error: 0.9371
    Epoch 246/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 23ms/step - loss: 0.8782 - root_mean_squared_error: 0.9371
    Epoch 247/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 72ms/step - loss: 0.8781 - root_mean_squared_error: 0.9371
    Epoch 248/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 34ms/step - loss: 0.8781 - root_mean_squared_error: 0.9371
    Epoch 249/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 83ms/step - loss: 0.8780 - root_mean_squared_error: 0.9370
    Epoch 250/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 33ms/step - loss: 0.8780 - root_mean_squared_error: 0.9370
    Epoch 251/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 29ms/step - loss: 0.8779 - root_mean_squared_error: 0.9370
    Epoch 252/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 74ms/step - loss: 0.8779 - root_mean_squared_error: 0.9369
    Epoch 253/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 264ms/step - loss: 0.8778 - root_mean_squared_error: 0.9369
    Epoch 254/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 65ms/step - loss: 0.8777 - root_mean_squared_error: 0.9369
    Epoch 255/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 90ms/step - loss: 0.8777 - root_mean_squared_error: 0.9368
    Epoch 256/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 93ms/step - loss: 0.8776 - root_mean_squared_error: 0.9368
    Epoch 257/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 40ms/step - loss: 0.8776 - root_mean_squared_error: 0.9368
    Epoch 258/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 31ms/step - loss: 0.8775 - root_mean_squared_error: 0.9367
    Epoch 259/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 75ms/step - loss: 0.8774 - root_mean_squared_error: 0.9367
    Epoch 260/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 29ms/step - loss: 0.8773 - root_mean_squared_error: 0.9367
    Epoch 261/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 26ms/step - loss: 0.8773 - root_mean_squared_error: 0.9366
    Epoch 262/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 82ms/step - loss: 0.8772 - root_mean_squared_error: 0.9366
    Epoch 263/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 23ms/step - loss: 0.8771 - root_mean_squared_error: 0.9366
    Epoch 264/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 24ms/step - loss: 0.8771 - root_mean_squared_error: 0.9365
    Epoch 265/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 68ms/step - loss: 0.8770 - root_mean_squared_error: 0.9365
    Epoch 266/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 29ms/step - loss: 0.8769 - root_mean_squared_error: 0.9364
    Epoch 267/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 25ms/step - loss: 0.8768 - root_mean_squared_error: 0.9364
    Epoch 268/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 81ms/step - loss: 0.8768 - root_mean_squared_error: 0.9364
    Epoch 269/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 28ms/step - loss: 0.8767 - root_mean_squared_error: 0.9363
    Epoch 270/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 19ms/step - loss: 0.8766 - root_mean_squared_error: 0.9363
    Epoch 271/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 64ms/step - loss: 0.8765 - root_mean_squared_error: 0.9362
    Epoch 272/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 32ms/step - loss: 0.8764 - root_mean_squared_error: 0.9362
    Epoch 273/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 26ms/step - loss: 0.8764 - root_mean_squared_error: 0.9361
    Epoch 274/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 79ms/step - loss: 0.8763 - root_mean_squared_error: 0.9361
    Epoch 275/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 35ms/step - loss: 0.8762 - root_mean_squared_error: 0.9361
    Epoch 276/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 29ms/step - loss: 0.8761 - root_mean_squared_error: 0.9360
    Epoch 277/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 63ms/step - loss: 0.8761 - root_mean_squared_error: 0.9360
    Epoch 278/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 59ms/step - loss: 0.8760 - root_mean_squared_error: 0.9359
    Epoch 279/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 187ms/step - loss: 0.8759 - root_mean_squared_error: 0.9359
    Epoch 280/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 27ms/step - loss: 0.8759 - root_mean_squared_error: 0.9359
    Epoch 281/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 21ms/step - loss: 0.8758 - root_mean_squared_error: 0.9358
    Epoch 282/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 77ms/step - loss: 0.8757 - root_mean_squared_error: 0.9358
    Epoch 283/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 28ms/step - loss: 0.8757 - root_mean_squared_error: 0.9358
    Epoch 284/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 25ms/step - loss: 0.8756 - root_mean_squared_error: 0.9357
    Epoch 285/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 65ms/step - loss: 0.8756 - root_mean_squared_error: 0.9357
    Epoch 286/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 29ms/step - loss: 0.8755 - root_mean_squared_error: 0.9357
    Epoch 287/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 29ms/step - loss: 0.8755 - root_mean_squared_error: 0.9357
    Epoch 288/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 77ms/step - loss: 0.8754 - root_mean_squared_error: 0.9356
    Epoch 289/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 30ms/step - loss: 0.8754 - root_mean_squared_error: 0.9356
    Epoch 290/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 28ms/step - loss: 0.8754 - root_mean_squared_error: 0.9356
    Epoch 291/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 21ms/step - loss: 0.8753 - root_mean_squared_error: 0.9356
    Epoch 292/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 34ms/step - loss: 0.8753 - root_mean_squared_error: 0.9356
    Epoch 293/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 83ms/step - loss: 0.8753 - root_mean_squared_error: 0.9356
    Epoch 294/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 28ms/step - loss: 0.8752 - root_mean_squared_error: 0.9355
    Epoch 295/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 32ms/step - loss: 0.8752 - root_mean_squared_error: 0.9355
    Epoch 296/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 75ms/step - loss: 0.8753 - root_mean_squared_error: 0.9356
    Epoch 297/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 26ms/step - loss: 0.8759 - root_mean_squared_error: 0.9359
    Epoch 298/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 25ms/step - loss: 0.8815 - root_mean_squared_error: 0.9389
    Epoch 299/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 680ms/step - loss: 0.8913 - root_mean_squared_error: 0.9441
    Epoch 300/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 66ms/step - loss: 0.8818 - root_mean_squared_error: 0.9391
    Epoch 301/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 49ms/step - loss: 0.8770 - root_mean_squared_error: 0.9365
    Epoch 302/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 95ms/step - loss: 0.8757 - root_mean_squared_error: 0.9358
    Epoch 303/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 36ms/step - loss: 0.8754 - root_mean_squared_error: 0.9356
    Epoch 304/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 27ms/step - loss: 0.8753 - root_mean_squared_error: 0.9356
    Epoch 305/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 69ms/step - loss: 0.8752 - root_mean_squared_error: 0.9355
    Epoch 306/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 34ms/step - loss: 0.8752 - root_mean_squared_error: 0.9355
    Epoch 307/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 30ms/step - loss: 0.8752 - root_mean_squared_error: 0.9355
    Epoch 308/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 225ms/step - loss: 0.8752 - root_mean_squared_error: 0.9355
    Epoch 309/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 45ms/step - loss: 0.8752 - root_mean_squared_error: 0.9355
    Epoch 310/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 38ms/step - loss: 0.8753 - root_mean_squared_error: 0.9356
    Epoch 311/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 31ms/step - loss: 0.8754 - root_mean_squared_error: 0.9356
    Epoch 312/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 39ms/step - loss: 0.8757 - root_mean_squared_error: 0.9358
    Epoch 313/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 62ms/step - loss: 0.8766 - root_mean_squared_error: 0.9363
    Epoch 314/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 55ms/step - loss: 0.8786 - root_mean_squared_error: 0.9373
    Epoch 315/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 29ms/step - loss: 0.8807 - root_mean_squared_error: 0.9385
    Epoch 316/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 41ms/step - loss: 0.8805 - root_mean_squared_error: 0.9383
    Epoch 317/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 276ms/step - loss: 0.8785 - root_mean_squared_error: 0.9373
    Epoch 318/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 52ms/step - loss: 0.8770 - root_mean_squared_error: 0.9365
    Epoch 319/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 39ms/step - loss: 0.8762 - root_mean_squared_error: 0.9361
    Epoch 320/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 79ms/step - loss: 0.8758 - root_mean_squared_error: 0.9359
    Epoch 321/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 30ms/step - loss: 0.8757 - root_mean_squared_error: 0.9358
    Epoch 322/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 26ms/step - loss: 0.8756 - root_mean_squared_error: 0.9357
    Epoch 323/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 59ms/step - loss: 0.8756 - root_mean_squared_error: 0.9357
    Epoch 324/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 28ms/step - loss: 0.8757 - root_mean_squared_error: 0.9358
    Epoch 325/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 20ms/step - loss: 0.8759 - root_mean_squared_error: 0.9359
    Epoch 326/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 72ms/step - loss: 0.8763 - root_mean_squared_error: 0.9361
    Epoch 327/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 29ms/step - loss: 0.8769 - root_mean_squared_error: 0.9364
    Epoch 328/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 26ms/step - loss: 0.8776 - root_mean_squared_error: 0.9368
    Epoch 329/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 79ms/step - loss: 0.8783 - root_mean_squared_error: 0.9372
    Epoch 330/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 30ms/step - loss: 0.8784 - root_mean_squared_error: 0.9372
    Epoch 331/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 25ms/step - loss: 0.8779 - root_mean_squared_error: 0.9370
    Epoch 332/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 25ms/step - loss: 0.8773 - root_mean_squared_error: 0.9366
    Epoch 333/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 255ms/step - loss: 0.8767 - root_mean_squared_error: 0.9363
    Epoch 334/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 53ms/step - loss: 0.8764 - root_mean_squared_error: 0.9362
    Epoch 335/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 58ms/step - loss: 0.8762 - root_mean_squared_error: 0.9360
    Epoch 336/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 64ms/step - loss: 0.8761 - root_mean_squared_error: 0.9360
    Epoch 337/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 47ms/step - loss: 0.8761 - root_mean_squared_error: 0.9360
    Epoch 338/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 67ms/step - loss: 0.8762 - root_mean_squared_error: 0.9361
    Epoch 339/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 33ms/step - loss: 0.8764 - root_mean_squared_error: 0.9362
    Epoch 340/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 30ms/step - loss: 0.8767 - root_mean_squared_error: 0.9363
    Epoch 341/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 62ms/step - loss: 0.8770 - root_mean_squared_error: 0.9365
    Epoch 342/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 35ms/step - loss: 0.8773 - root_mean_squared_error: 0.9366
    Epoch 343/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 41ms/step - loss: 0.8775 - root_mean_squared_error: 0.9367
    Epoch 344/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 66ms/step - loss: 0.8775 - root_mean_squared_error: 0.9367
    Epoch 345/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 28ms/step - loss: 0.8773 - root_mean_squared_error: 0.9366
    Epoch 346/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 21ms/step - loss: 0.8770 - root_mean_squared_error: 0.9365
    Epoch 347/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 85ms/step - loss: 0.8768 - root_mean_squared_error: 0.9364
    Epoch 348/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 125ms/step - loss: 0.8766 - root_mean_squared_error: 0.9363
    Epoch 349/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 228ms/step - loss: 0.8765 - root_mean_squared_error: 0.9362
    Epoch 350/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 81ms/step - loss: 0.8764 - root_mean_squared_error: 0.9362
    Epoch 351/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 52ms/step - loss: 0.8765 - root_mean_squared_error: 0.9362
    Epoch 352/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 52ms/step - loss: 0.8765 - root_mean_squared_error: 0.9362
    Epoch 353/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 41ms/step - loss: 0.8767 - root_mean_squared_error: 0.9363
    Epoch 354/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 57ms/step - loss: 0.8768 - root_mean_squared_error: 0.9364
    Epoch 355/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 51ms/step - loss: 0.8770 - root_mean_squared_error: 0.9365
    Epoch 356/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 38ms/step - loss: 0.8771 - root_mean_squared_error: 0.9365
    Epoch 357/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 35ms/step - loss: 0.8771 - root_mean_squared_error: 0.9366
    Epoch 358/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 29ms/step - loss: 0.8771 - root_mean_squared_error: 0.9365
    Epoch 359/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 42ms/step - loss: 0.8770 - root_mean_squared_error: 0.9365
    Epoch 360/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 70ms/step - loss: 0.8769 - root_mean_squared_error: 0.9364
    Epoch 361/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 43ms/step - loss: 0.8768 - root_mean_squared_error: 0.9364
    Epoch 362/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 46ms/step - loss: 0.8767 - root_mean_squared_error: 0.9363
    Epoch 363/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 39ms/step - loss: 0.8766 - root_mean_squared_error: 0.9363
    Epoch 364/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 36ms/step - loss: 0.8766 - root_mean_squared_error: 0.9363
    Epoch 365/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 227ms/step - loss: 0.8766 - root_mean_squared_error: 0.9363
    Epoch 366/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 29ms/step - loss: 0.8767 - root_mean_squared_error: 0.9363
    Epoch 367/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 20ms/step - loss: 0.8768 - root_mean_squared_error: 0.9364
    Epoch 368/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 78ms/step - loss: 0.8769 - root_mean_squared_error: 0.9364
    Epoch 369/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 32ms/step - loss: 0.8769 - root_mean_squared_error: 0.9364
    Epoch 370/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 106ms/step - loss: 0.8770 - root_mean_squared_error: 0.9365
    Epoch 371/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 49ms/step - loss: 0.8770 - root_mean_squared_error: 0.9365
    Epoch 372/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 61ms/step - loss: 0.8769 - root_mean_squared_error: 0.9365
    Epoch 373/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 56ms/step - loss: 0.8769 - root_mean_squared_error: 0.9364
    Epoch 374/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 78ms/step - loss: 0.8768 - root_mean_squared_error: 0.9364
    Epoch 375/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 49ms/step - loss: 0.8768 - root_mean_squared_error: 0.9364
    Epoch 376/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 109ms/step - loss: 0.8767 - root_mean_squared_error: 0.9363
    Epoch 377/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 49ms/step - loss: 0.8767 - root_mean_squared_error: 0.9363
    Epoch 378/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 43ms/step - loss: 0.8767 - root_mean_squared_error: 0.9363
    Epoch 379/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 39ms/step - loss: 0.8767 - root_mean_squared_error: 0.9363
    Epoch 380/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 255ms/step - loss: 0.8768 - root_mean_squared_error: 0.9364
    Epoch 381/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 33ms/step - loss: 0.8768 - root_mean_squared_error: 0.9364
    Epoch 382/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 59ms/step - loss: 0.8769 - root_mean_squared_error: 0.9364
    Epoch 383/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 57ms/step - loss: 0.8769 - root_mean_squared_error: 0.9364
    Epoch 384/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 59ms/step - loss: 0.8769 - root_mean_squared_error: 0.9364
    Epoch 385/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 37ms/step - loss: 0.8769 - root_mean_squared_error: 0.9364
    Epoch 386/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 46ms/step - loss: 0.8769 - root_mean_squared_error: 0.9364
    Epoch 387/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 59ms/step - loss: 0.8768 - root_mean_squared_error: 0.9364
    Epoch 388/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 18ms/step - loss: 0.8768 - root_mean_squared_error: 0.9364
    Epoch 389/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 56ms/step - loss: 0.8768 - root_mean_squared_error: 0.9364
    Epoch 390/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 36ms/step - loss: 0.8768 - root_mean_squared_error: 0.9364
    Epoch 391/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 66ms/step - loss: 0.8768 - root_mean_squared_error: 0.9364
    Epoch 392/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 35ms/step - loss: 0.8768 - root_mean_squared_error: 0.9364
    Epoch 393/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 177ms/step - loss: 0.8768 - root_mean_squared_error: 0.9364
    Epoch 394/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 74ms/step - loss: 0.8768 - root_mean_squared_error: 0.9364
    Epoch 395/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 23ms/step - loss: 0.8768 - root_mean_squared_error: 0.9364
    Epoch 396/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 26ms/step - loss: 0.8768 - root_mean_squared_error: 0.9364
    Epoch 397/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 82ms/step - loss: 0.8769 - root_mean_squared_error: 0.9364
    Epoch 398/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 29ms/step - loss: 0.8769 - root_mean_squared_error: 0.9364
    Epoch 399/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 26ms/step - loss: 0.8769 - root_mean_squared_error: 0.9364
    Epoch 400/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 74ms/step - loss: 0.8768 - root_mean_squared_error: 0.9364
    Epoch 401/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 32ms/step - loss: 0.8768 - root_mean_squared_error: 0.9364
    Epoch 402/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 24ms/step - loss: 0.8768 - root_mean_squared_error: 0.9364
    Epoch 403/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 81ms/step - loss: 0.8768 - root_mean_squared_error: 0.9364
    Epoch 404/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 37ms/step - loss: 0.8768 - root_mean_squared_error: 0.9364
    Epoch 405/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 57ms/step - loss: 0.8768 - root_mean_squared_error: 0.9364
    Epoch 406/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 59ms/step - loss: 0.8768 - root_mean_squared_error: 0.9364
    Epoch 407/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 54ms/step - loss: 0.8768 - root_mean_squared_error: 0.9364
    Epoch 408/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 252ms/step - loss: 0.8768 - root_mean_squared_error: 0.9364
    Epoch 409/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 32ms/step - loss: 0.8768 - root_mean_squared_error: 0.9364
    Epoch 410/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 23ms/step - loss: 0.8768 - root_mean_squared_error: 0.9364
    Epoch 411/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 72ms/step - loss: 0.8768 - root_mean_squared_error: 0.9364
    Epoch 412/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 26ms/step - loss: 0.8768 - root_mean_squared_error: 0.9364
    Epoch 413/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 25ms/step - loss: 0.8768 - root_mean_squared_error: 0.9364
    Epoch 414/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 70ms/step - loss: 0.8768 - root_mean_squared_error: 0.9364
    Epoch 415/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 64ms/step - loss: 0.8768 - root_mean_squared_error: 0.9364
    Epoch 416/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 48ms/step - loss: 0.8768 - root_mean_squared_error: 0.9364
    Epoch 417/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 96ms/step - loss: 0.8768 - root_mean_squared_error: 0.9364
    Epoch 418/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 128ms/step - loss: 0.8768 - root_mean_squared_error: 0.9364
    Epoch 419/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 42ms/step - loss: 0.8768 - root_mean_squared_error: 0.9364
    Epoch 420/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 37ms/step - loss: 0.8768 - root_mean_squared_error: 0.9364
    Epoch 421/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 321ms/step - loss: 0.8768 - root_mean_squared_error: 0.9364
    Epoch 422/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 20ms/step - loss: 0.8768 - root_mean_squared_error: 0.9364
    Epoch 423/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 99ms/step - loss: 0.8768 - root_mean_squared_error: 0.9364
    Epoch 424/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 59ms/step - loss: 0.8768 - root_mean_squared_error: 0.9364
    Epoch 425/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 60ms/step - loss: 0.8768 - root_mean_squared_error: 0.9364
    Epoch 426/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 66ms/step - loss: 0.8768 - root_mean_squared_error: 0.9364
    Epoch 427/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 43ms/step - loss: 0.8768 - root_mean_squared_error: 0.9364
    Epoch 428/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 38ms/step - loss: 0.8768 - root_mean_squared_error: 0.9364
    Epoch 429/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 64ms/step - loss: 0.8768 - root_mean_squared_error: 0.9364
    Epoch 430/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 85ms/step - loss: 0.8768 - root_mean_squared_error: 0.9364
    Epoch 431/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 43ms/step - loss: 0.8768 - root_mean_squared_error: 0.9364
    Epoch 432/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 85ms/step - loss: 0.8768 - root_mean_squared_error: 0.9364
    Epoch 433/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 118ms/step - loss: 0.8768 - root_mean_squared_error: 0.9364
    Epoch 434/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 35ms/step - loss: 0.8768 - root_mean_squared_error: 0.9364
    Epoch 435/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 43ms/step - loss: 0.8768 - root_mean_squared_error: 0.9364
    Epoch 436/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 48ms/step - loss: 0.8768 - root_mean_squared_error: 0.9364
    Epoch 437/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 37ms/step - loss: 0.8768 - root_mean_squared_error: 0.9364
    Epoch 438/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 86ms/step - loss: 0.8768 - root_mean_squared_error: 0.9364
    Epoch 439/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 49ms/step - loss: 0.8768 - root_mean_squared_error: 0.9364
    Epoch 440/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 64ms/step - loss: 0.8768 - root_mean_squared_error: 0.9364
    Epoch 441/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 58ms/step - loss: 0.8768 - root_mean_squared_error: 0.9364
    Epoch 442/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 46ms/step - loss: 0.8768 - root_mean_squared_error: 0.9364
    Epoch 443/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 28ms/step - loss: 0.8768 - root_mean_squared_error: 0.9364
    Epoch 444/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 30ms/step - loss: 0.8768 - root_mean_squared_error: 0.9364
    Epoch 445/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 77ms/step - loss: 0.8768 - root_mean_squared_error: 0.9364
    Epoch 446/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 33ms/step - loss: 0.8768 - root_mean_squared_error: 0.9364
    Epoch 447/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 274ms/step - loss: 0.8768 - root_mean_squared_error: 0.9364
    Epoch 448/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 46ms/step - loss: 0.8768 - root_mean_squared_error: 0.9364
    Epoch 449/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 93ms/step - loss: 0.8768 - root_mean_squared_error: 0.9364
    Epoch 450/450
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 59ms/step - loss: 0.8768 - root_mean_squared_error: 0.9364
    


    
![png](output_16_1.png)
    



    
![png](output_16_2.png)
    


## Task 3: Increase the learning rate

In Task 2, you increased the number of epochs to get the model to converge. Sometimes, you can get the model to converge more quickly by increasing the learning rate. However, setting the learning rate too high often makes it impossible for a model to converge. In Task 3, we've intentionally set the learning rate too high. Run the following code cell and see what happens.


```python
# Increase the learning rate and decrease the number of epochs.
learning_rate=100.0
epochs=500

my_model = build_model(learning_rate)
trained_weight, trained_bias, epochs, rmse = train_model(my_model, my_feature,
                                                         my_label, epochs,
                                                         my_batch_size)
plot_the_model(trained_weight, trained_bias, my_feature, my_label)
plot_the_loss_curve(epochs, rmse)
```

    Epoch 1/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 338ms/step - loss: 428.7742 - root_mean_squared_error: 20.7069
    Epoch 2/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 25ms/step - loss: 6709082.0000 - root_mean_squared_error: 2590.1895
    Epoch 3/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 78ms/step - loss: 425.6857 - root_mean_squared_error: 20.6322
    Epoch 4/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 36ms/step - loss: 2.5534 - root_mean_squared_error: 1.5979
    Epoch 5/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 33ms/step - loss: 0.9197 - root_mean_squared_error: 0.9590
    Epoch 6/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 48ms/step - loss: 0.8954 - root_mean_squared_error: 0.9462
    Epoch 7/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 35ms/step - loss: 0.8934 - root_mean_squared_error: 0.9452
    Epoch 8/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 36ms/step - loss: 0.8921 - root_mean_squared_error: 0.9445
    Epoch 9/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 45ms/step - loss: 0.8909 - root_mean_squared_error: 0.9439
    Epoch 10/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 38ms/step - loss: 0.8897 - root_mean_squared_error: 0.9433
    Epoch 11/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 32ms/step - loss: 0.8886 - root_mean_squared_error: 0.9426
    Epoch 12/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 29ms/step - loss: 0.8875 - root_mean_squared_error: 0.9421
    Epoch 13/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 28ms/step - loss: 0.8864 - root_mean_squared_error: 0.9415
    Epoch 14/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 24ms/step - loss: 0.8853 - root_mean_squared_error: 0.9409
    Epoch 15/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 28ms/step - loss: 0.8844 - root_mean_squared_error: 0.9404
    Epoch 16/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 23ms/step - loss: 0.8834 - root_mean_squared_error: 0.9399
    Epoch 17/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 67ms/step - loss: 0.8825 - root_mean_squared_error: 0.9394
    Epoch 18/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 19ms/step - loss: 0.8817 - root_mean_squared_error: 0.9390
    Epoch 19/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 20ms/step - loss: 0.8809 - root_mean_squared_error: 0.9386
    Epoch 20/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 80ms/step - loss: 0.8802 - root_mean_squared_error: 0.9382
    Epoch 21/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 24ms/step - loss: 0.8795 - root_mean_squared_error: 0.9378
    Epoch 22/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 17ms/step - loss: 0.8789 - root_mean_squared_error: 0.9375
    Epoch 23/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 84ms/step - loss: 0.8784 - root_mean_squared_error: 0.9372
    Epoch 24/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 33ms/step - loss: 0.8779 - root_mean_squared_error: 0.9369
    Epoch 25/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 43ms/step - loss: 0.8774 - root_mean_squared_error: 0.9367
    Epoch 26/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 59ms/step - loss: 0.8771 - root_mean_squared_error: 0.9365
    Epoch 27/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 26ms/step - loss: 0.8772 - root_mean_squared_error: 0.9366
    Epoch 28/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 33ms/step - loss: 0.8805 - root_mean_squared_error: 0.9384
    Epoch 29/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 92ms/step - loss: 0.9124 - root_mean_squared_error: 0.9552
    Epoch 30/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 39ms/step - loss: 1.2426 - root_mean_squared_error: 1.1147
    Epoch 31/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 47ms/step - loss: 5.1295 - root_mean_squared_error: 2.2648
    Epoch 32/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 35ms/step - loss: 57.3646 - root_mean_squared_error: 7.5739
    Epoch 33/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 26ms/step - loss: 857.8227 - root_mean_squared_error: 29.2886
    Epoch 34/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 72ms/step - loss: 14767.1484 - root_mean_squared_error: 121.5202
    Epoch 35/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 184ms/step - loss: 268501.4688 - root_mean_squared_error: 518.1713
    Epoch 36/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 220ms/step - loss: 2001478.6250 - root_mean_squared_error: 1414.7362
    Epoch 37/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 38ms/step - loss: 900149.5625 - root_mean_squared_error: 948.7621
    Epoch 38/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 33ms/step - loss: 210245.2344 - root_mean_squared_error: 458.5251
    Epoch 39/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 79ms/step - loss: 54196.5664 - root_mean_squared_error: 232.8016
    Epoch 40/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 23ms/step - loss: 17829.1113 - root_mean_squared_error: 133.5257
    Epoch 41/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 19ms/step - loss: 7596.7739 - root_mean_squared_error: 87.1595
    Epoch 42/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 28ms/step - loss: 4151.3403 - root_mean_squared_error: 64.4309
    Epoch 43/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 23ms/step - loss: 2869.7803 - root_mean_squared_error: 53.5703
    Epoch 44/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 88ms/step - loss: 2476.6042 - root_mean_squared_error: 49.7655
    Epoch 45/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 31ms/step - loss: 2636.4099 - root_mean_squared_error: 51.3460
    Epoch 46/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 65ms/step - loss: 3424.6758 - root_mean_squared_error: 58.5207
    Epoch 47/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 29ms/step - loss: 5373.2417 - root_mean_squared_error: 73.3024
    Epoch 48/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 23ms/step - loss: 10073.0811 - root_mean_squared_error: 100.3647
    Epoch 49/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 24ms/step - loss: 22241.3652 - root_mean_squared_error: 149.1354
    Epoch 50/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 19ms/step - loss: 56338.2969 - root_mean_squared_error: 237.3569
    Epoch 51/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 78ms/step - loss: 153606.9844 - root_mean_squared_error: 391.9273
    Epoch 52/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 35ms/step - loss: 382033.8750 - root_mean_squared_error: 618.0889
    Epoch 53/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 33ms/step - loss: 630614.3750 - root_mean_squared_error: 794.1123
    Epoch 54/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 80ms/step - loss: 560630.3125 - root_mean_squared_error: 748.7525
    Epoch 55/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 31ms/step - loss: 328888.0938 - root_mean_squared_error: 573.4877
    Epoch 56/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 28ms/step - loss: 171914.3906 - root_mean_squared_error: 414.6256
    Epoch 57/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 68ms/step - loss: 95431.3203 - root_mean_squared_error: 308.9196
    Epoch 58/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 26ms/step - loss: 60675.8867 - root_mean_squared_error: 246.3248
    Epoch 59/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 22ms/step - loss: 45403.3555 - root_mean_squared_error: 213.0806
    Epoch 60/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 28ms/step - loss: 40212.1992 - root_mean_squared_error: 200.5298
    Epoch 61/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 36ms/step - loss: 41942.5469 - root_mean_squared_error: 204.7988
    Epoch 62/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 32ms/step - loss: 50886.9375 - root_mean_squared_error: 225.5813
    Epoch 63/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 35ms/step - loss: 70335.0391 - root_mean_squared_error: 265.2076
    Epoch 64/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 68ms/step - loss: 107024.4141 - root_mean_squared_error: 327.1459
    Epoch 65/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 26ms/step - loss: 169414.6250 - root_mean_squared_error: 411.6001
    Epoch 66/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 29ms/step - loss: 255709.8281 - root_mean_squared_error: 505.6776
    Epoch 67/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 20ms/step - loss: 331275.3750 - root_mean_squared_error: 575.5652
    Epoch 68/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 37ms/step - loss: 343610.2188 - root_mean_squared_error: 586.1827
    Epoch 69/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 68ms/step - loss: 290728.8438 - root_mean_squared_error: 539.1927
    Epoch 70/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 36ms/step - loss: 218629.2656 - root_mean_squared_error: 467.5781
    Epoch 71/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 33ms/step - loss: 160289.7344 - root_mean_squared_error: 400.3620
    Epoch 72/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 83ms/step - loss: 122473.8359 - root_mean_squared_error: 349.9626
    Epoch 73/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 26ms/step - loss: 101348.3672 - root_mean_squared_error: 318.3526
    Epoch 74/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 25ms/step - loss: 92422.2734 - root_mean_squared_error: 304.0103
    Epoch 75/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 75ms/step - loss: 93129.5859 - root_mean_squared_error: 305.1714
    Epoch 76/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 24ms/step - loss: 102800.5391 - root_mean_squared_error: 320.6252
    Epoch 77/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 26ms/step - loss: 121960.6953 - root_mean_squared_error: 349.2287
    Epoch 78/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 29ms/step - loss: 150939.7031 - root_mean_squared_error: 388.5096
    Epoch 79/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 33ms/step - loss: 187226.0781 - root_mean_squared_error: 432.6963
    Epoch 80/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 33ms/step - loss: 222496.5000 - root_mean_squared_error: 471.6953
    Epoch 81/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 269ms/step - loss: 243898.8594 - root_mean_squared_error: 493.8612
    Epoch 82/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 33ms/step - loss: 242794.2031 - root_mean_squared_error: 492.7415
    Epoch 83/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 27ms/step - loss: 222325.2969 - root_mean_squared_error: 471.5138
    Epoch 84/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 27ms/step - loss: 193532.3750 - root_mean_squared_error: 439.9232
    Epoch 85/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 29ms/step - loss: 166361.7031 - root_mean_squared_error: 407.8746
    Epoch 86/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 28ms/step - loss: 145821.9531 - root_mean_squared_error: 381.8664
    Epoch 87/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 22ms/step - loss: 133186.8594 - root_mean_squared_error: 364.9478
    Epoch 88/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 69ms/step - loss: 128143.6016 - root_mean_squared_error: 357.9715
    Epoch 89/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 25ms/step - loss: 130045.3672 - root_mean_squared_error: 360.6180
    Epoch 90/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 18ms/step - loss: 138236.8281 - root_mean_squared_error: 371.8021
    Epoch 91/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 77ms/step - loss: 151774.4844 - root_mean_squared_error: 389.5825
    Epoch 92/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 25ms/step - loss: 168814.8906 - root_mean_squared_error: 410.8709
    Epoch 93/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 18ms/step - loss: 186114.5781 - root_mean_squared_error: 431.4100
    Epoch 94/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 114ms/step - loss: 199388.8594 - root_mean_squared_error: 446.5298
    Epoch 95/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 39ms/step - loss: 204960.4531 - root_mean_squared_error: 452.7256
    Epoch 96/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 48ms/step - loss: 201749.5156 - root_mean_squared_error: 449.1653
    Epoch 97/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 101ms/step - loss: 191777.4219 - root_mean_squared_error: 437.9240
    Epoch 98/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 36ms/step - loss: 178746.9375 - root_mean_squared_error: 422.7847
    Epoch 99/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 27ms/step - loss: 166164.0469 - root_mean_squared_error: 407.6322
    Epoch 100/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 64ms/step - loss: 156356.2031 - root_mean_squared_error: 395.4190
    Epoch 101/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 20ms/step - loss: 150453.7812 - root_mean_squared_error: 387.8837
    Epoch 102/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 22ms/step - loss: 148755.3125 - root_mean_squared_error: 385.6881
    Epoch 103/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 80ms/step - loss: 151036.7031 - root_mean_squared_error: 388.6344
    Epoch 104/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 28ms/step - loss: 156664.6719 - root_mean_squared_error: 395.8089
    Epoch 105/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 34ms/step - loss: 164568.8281 - root_mean_squared_error: 405.6708
    Epoch 106/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 69ms/step - loss: 173217.4375 - root_mean_squared_error: 416.1940
    Epoch 107/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 52ms/step - loss: 180781.8281 - root_mean_squared_error: 425.1845
    Epoch 108/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 33ms/step - loss: 185578.0625 - root_mean_squared_error: 430.7877
    Epoch 109/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 79ms/step - loss: 186638.9844 - root_mean_squared_error: 432.0173
    Epoch 110/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 37ms/step - loss: 184065.1094 - root_mean_squared_error: 429.0281
    Epoch 111/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 31ms/step - loss: 178904.8906 - root_mean_squared_error: 422.9715
    Epoch 112/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 64ms/step - loss: 172667.5469 - root_mean_squared_error: 415.5328
    Epoch 113/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 55ms/step - loss: 166800.3906 - root_mean_squared_error: 408.4120
    Epoch 114/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 86ms/step - loss: 162371.2656 - root_mean_squared_error: 402.9532
    Epoch 115/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 28ms/step - loss: 159980.9062 - root_mean_squared_error: 399.9761
    Epoch 116/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 25ms/step - loss: 159801.1094 - root_mean_squared_error: 399.7513
    Epoch 117/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 69ms/step - loss: 161643.6094 - root_mean_squared_error: 402.0493
    Epoch 118/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 221ms/step - loss: 165016.7344 - root_mean_squared_error: 406.2225
    Epoch 119/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 28ms/step - loss: 169186.8281 - root_mean_squared_error: 411.3233
    Epoch 120/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 26ms/step - loss: 173280.2031 - root_mean_squared_error: 416.2694
    Epoch 121/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 62ms/step - loss: 176450.8750 - root_mean_squared_error: 420.0605
    Epoch 122/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 25ms/step - loss: 178083.9531 - root_mean_squared_error: 421.9999
    Epoch 123/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 19ms/step - loss: 177956.5625 - root_mean_squared_error: 421.8490
    Epoch 124/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 71ms/step - loss: 176277.3281 - root_mean_squared_error: 419.8539
    Epoch 125/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 28ms/step - loss: 173585.2031 - root_mean_squared_error: 416.6356
    Epoch 126/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 18ms/step - loss: 170564.2188 - root_mean_squared_error: 412.9942
    Epoch 127/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 75ms/step - loss: 167861.0156 - root_mean_squared_error: 409.7085
    Epoch 128/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 26ms/step - loss: 165959.8906 - root_mean_squared_error: 407.3817
    Epoch 129/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 21ms/step - loss: 165126.9375 - root_mean_squared_error: 406.3582
    Epoch 130/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 70ms/step - loss: 165404.3125 - root_mean_squared_error: 406.6993
    Epoch 131/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 23ms/step - loss: 166632.9688 - root_mean_squared_error: 408.2070
    Epoch 132/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 21ms/step - loss: 168494.8281 - root_mean_squared_error: 410.4812
    Epoch 133/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 78ms/step - loss: 170573.6094 - root_mean_squared_error: 413.0056
    Epoch 134/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 29ms/step - loss: 172434.4219 - root_mean_squared_error: 415.2522
    Epoch 135/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 27ms/step - loss: 173713.7969 - root_mean_squared_error: 416.7899
    Epoch 136/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 77ms/step - loss: 174197.2500 - root_mean_squared_error: 417.3694
    Epoch 137/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 23ms/step - loss: 173860.5469 - root_mean_squared_error: 416.9659
    Epoch 138/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 24ms/step - loss: 172858.7656 - root_mean_squared_error: 415.7629
    Epoch 139/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 71ms/step - loss: 171471.0000 - root_mean_squared_error: 414.0906
    Epoch 140/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 26ms/step - loss: 170021.4375 - root_mean_squared_error: 412.3365
    Epoch 141/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 61ms/step - loss: 168805.6094 - root_mean_squared_error: 410.8596
    Epoch 142/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 45ms/step - loss: 168035.8281 - root_mean_squared_error: 409.9217
    Epoch 143/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 66ms/step - loss: 167813.9844 - root_mean_squared_error: 409.6511
    Epoch 144/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 27ms/step - loss: 168127.5938 - root_mean_squared_error: 410.0337
    Epoch 145/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 18ms/step - loss: 168863.9375 - root_mean_squared_error: 410.9306
    Epoch 146/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 79ms/step - loss: 169839.1406 - root_mean_squared_error: 412.1154
    Epoch 147/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 27ms/step - loss: 170837.6094 - root_mean_squared_error: 413.3251
    Epoch 148/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 28ms/step - loss: 171656.4062 - root_mean_squared_error: 414.3144
    Epoch 149/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 29ms/step - loss: 172146.0156 - root_mean_squared_error: 414.9048
    Epoch 150/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 23ms/step - loss: 172237.8750 - root_mean_squared_error: 415.0155
    Epoch 151/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 69ms/step - loss: 171952.9219 - root_mean_squared_error: 414.6721
    Epoch 152/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 26ms/step - loss: 171387.9844 - root_mean_squared_error: 413.9903
    Epoch 153/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 17ms/step - loss: 170686.7031 - root_mean_squared_error: 413.1425
    Epoch 154/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 77ms/step - loss: 170004.4219 - root_mean_squared_error: 412.3159
    Epoch 155/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 26ms/step - loss: 169474.7344 - root_mean_squared_error: 411.6731
    Epoch 156/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 18ms/step - loss: 169186.4531 - root_mean_squared_error: 411.3228
    Epoch 157/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 74ms/step - loss: 169171.5625 - root_mean_squared_error: 411.3047
    Epoch 158/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 27ms/step - loss: 169405.7344 - root_mean_squared_error: 411.5893
    Epoch 159/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 28ms/step - loss: 169818.3594 - root_mean_squared_error: 412.0902
    Epoch 160/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 214ms/step - loss: 170310.4844 - root_mean_squared_error: 412.6869
    Epoch 161/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 17ms/step - loss: 170776.0938 - root_mean_squared_error: 413.2506
    Epoch 162/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 28ms/step - loss: 171124.1094 - root_mean_squared_error: 413.6715
    Epoch 163/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 66ms/step - loss: 171295.8594 - root_mean_squared_error: 413.8790
    Epoch 164/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 24ms/step - loss: 171273.8750 - root_mean_squared_error: 413.8525
    Epoch 165/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 24ms/step - loss: 171082.0625 - root_mean_squared_error: 413.6207
    Epoch 166/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 69ms/step - loss: 170776.4219 - root_mean_squared_error: 413.2510
    Epoch 167/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 38ms/step - loss: 170430.0469 - root_mean_squared_error: 412.8318
    Epoch 168/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 28ms/step - loss: 170116.4688 - root_mean_squared_error: 412.4518
    Epoch 169/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 67ms/step - loss: 169894.7188 - root_mean_squared_error: 412.1829
    Epoch 170/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 32ms/step - loss: 169799.5469 - root_mean_squared_error: 412.0674
    Epoch 171/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 25ms/step - loss: 169836.9844 - root_mean_squared_error: 412.1128
    Epoch 172/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 77ms/step - loss: 169986.5781 - root_mean_squared_error: 412.2943
    Epoch 173/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 25ms/step - loss: 170207.4375 - root_mean_squared_error: 412.5620
    Epoch 174/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 24ms/step - loss: 170448.3750 - root_mean_squared_error: 412.8539
    Epoch 175/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 91ms/step - loss: 170659.3906 - root_mean_squared_error: 413.1094
    Epoch 176/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 28ms/step - loss: 170801.1562 - root_mean_squared_error: 413.2810
    Epoch 177/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 47ms/step - loss: 170851.8125 - root_mean_squared_error: 413.3423
    Epoch 178/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 87ms/step - loss: 170810.3594 - root_mean_squared_error: 413.2921
    Epoch 179/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 41ms/step - loss: 170694.2656 - root_mean_squared_error: 413.1516
    Epoch 180/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 44ms/step - loss: 170534.6719 - root_mean_squared_error: 412.9584
    Epoch 181/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 28ms/step - loss: 170367.6719 - root_mean_squared_error: 412.7562
    Epoch 182/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 34ms/step - loss: 170227.5000 - root_mean_squared_error: 412.5864
    Epoch 183/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 86ms/step - loss: 170139.3281 - root_mean_squared_error: 412.4795
    Epoch 184/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 34ms/step - loss: 170115.6094 - root_mean_squared_error: 412.4507
    Epoch 185/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 258ms/step - loss: 170154.4531 - root_mean_squared_error: 412.4978
    Epoch 186/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 41ms/step - loss: 170242.0156 - root_mean_squared_error: 412.6039
    Epoch 187/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 64ms/step - loss: 170355.8750 - root_mean_squared_error: 412.7419
    Epoch 188/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 33ms/step - loss: 170470.7656 - root_mean_squared_error: 412.8810
    Epoch 189/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 28ms/step - loss: 170563.6562 - root_mean_squared_error: 412.9935
    Epoch 190/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 26ms/step - loss: 170618.0156 - root_mean_squared_error: 413.0593
    Epoch 191/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 23ms/step - loss: 170626.8281 - root_mean_squared_error: 413.0700
    Epoch 192/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 66ms/step - loss: 170592.8594 - root_mean_squared_error: 413.0289
    Epoch 193/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 26ms/step - loss: 170527.1719 - root_mean_squared_error: 412.9494
    Epoch 194/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 20ms/step - loss: 170446.1719 - root_mean_squared_error: 412.8513
    Epoch 195/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 79ms/step - loss: 170367.5469 - root_mean_squared_error: 412.7560
    Epoch 196/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 27ms/step - loss: 170306.8906 - root_mean_squared_error: 412.6826
    Epoch 197/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 23ms/step - loss: 170274.4844 - root_mean_squared_error: 412.6433
    Epoch 198/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 62ms/step - loss: 170273.8438 - root_mean_squared_error: 412.6425
    Epoch 199/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 26ms/step - loss: 170302.0625 - root_mean_squared_error: 412.6767
    Epoch 200/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 20ms/step - loss: 170350.4688 - root_mean_squared_error: 412.7354
    Epoch 201/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 77ms/step - loss: 170407.5000 - root_mean_squared_error: 412.8044
    Epoch 202/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 27ms/step - loss: 170460.7812 - root_mean_squared_error: 412.8690
    Epoch 203/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 19ms/step - loss: 170500.2031 - root_mean_squared_error: 412.9167
    Epoch 204/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 55ms/step - loss: 170519.0625 - root_mean_squared_error: 412.9395
    Epoch 205/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 35ms/step - loss: 170515.7031 - root_mean_squared_error: 412.9355
    Epoch 206/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 32ms/step - loss: 170493.0625 - root_mean_squared_error: 412.9081
    Epoch 207/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 340ms/step - loss: 170457.5469 - root_mean_squared_error: 412.8651
    Epoch 208/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 35ms/step - loss: 170417.6719 - root_mean_squared_error: 412.8167
    Epoch 209/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 83ms/step - loss: 170381.7969 - root_mean_squared_error: 412.7733
    Epoch 210/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 24ms/step - loss: 170356.6250 - root_mean_squared_error: 412.7428
    Epoch 211/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 23ms/step - loss: 170346.1562 - root_mean_squared_error: 412.7301
    Epoch 212/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 81ms/step - loss: 170351.0781 - root_mean_squared_error: 412.7361
    Epoch 213/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 33ms/step - loss: 170368.7812 - root_mean_squared_error: 412.7575
    Epoch 214/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 31ms/step - loss: 170394.4844 - root_mean_squared_error: 412.7887
    Epoch 215/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 47ms/step - loss: 170422.2031 - root_mean_squared_error: 412.8222
    Epoch 216/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 39ms/step - loss: 170446.1719 - root_mean_squared_error: 412.8513
    Epoch 217/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 36ms/step - loss: 170462.0312 - root_mean_squared_error: 412.8705
    Epoch 218/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 41ms/step - loss: 170467.4531 - root_mean_squared_error: 412.8770
    Epoch 219/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 71ms/step - loss: 170462.3906 - root_mean_squared_error: 412.8709
    Epoch 220/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 28ms/step - loss: 170448.8906 - root_mean_squared_error: 412.8546
    Epoch 221/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 31ms/step - loss: 170430.4844 - root_mean_squared_error: 412.8323
    Epoch 222/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 84ms/step - loss: 170411.2656 - root_mean_squared_error: 412.8090
    Epoch 223/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 30ms/step - loss: 170395.2656 - root_mean_squared_error: 412.7896
    Epoch 224/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 18ms/step - loss: 170385.4219 - root_mean_squared_error: 412.7777
    Epoch 225/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 72ms/step - loss: 170382.9531 - root_mean_squared_error: 412.7747
    Epoch 226/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 28ms/step - loss: 170387.6094 - root_mean_squared_error: 412.7803
    Epoch 227/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 19ms/step - loss: 170397.6719 - root_mean_squared_error: 412.7925
    Epoch 228/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 67ms/step - loss: 170410.8438 - root_mean_squared_error: 412.8085
    Epoch 229/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 29ms/step - loss: 170424.0781 - root_mean_squared_error: 412.8245
    Epoch 230/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 31ms/step - loss: 170434.6719 - root_mean_squared_error: 412.8373
    Epoch 231/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 137ms/step - loss: 170440.7969 - root_mean_squared_error: 412.8448
    Epoch 232/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 36ms/step - loss: 170441.6406 - root_mean_squared_error: 412.8458
    Epoch 233/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 35ms/step - loss: 170437.4688 - root_mean_squared_error: 412.8407
    Epoch 234/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 82ms/step - loss: 170429.8438 - root_mean_squared_error: 412.8315
    Epoch 235/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 29ms/step - loss: 170420.5781 - root_mean_squared_error: 412.8203
    Epoch 236/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 18ms/step - loss: 170411.5938 - root_mean_squared_error: 412.8094
    Epoch 237/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 80ms/step - loss: 170404.6250 - root_mean_squared_error: 412.8010
    Epoch 238/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 56ms/step - loss: 170401.0938 - root_mean_squared_error: 412.7967
    Epoch 239/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 38ms/step - loss: 170401.1406 - root_mean_squared_error: 412.7967
    Epoch 240/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 48ms/step - loss: 170404.5469 - root_mean_squared_error: 412.8008
    Epoch 241/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 34ms/step - loss: 170410.1875 - root_mean_squared_error: 412.8077
    Epoch 242/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 32ms/step - loss: 170416.7656 - root_mean_squared_error: 412.8156
    Epoch 243/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 30ms/step - loss: 170422.7344 - root_mean_squared_error: 412.8229
    Epoch 244/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 70ms/step - loss: 170427.1250 - root_mean_squared_error: 412.8282
    Epoch 245/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 28ms/step - loss: 170429.1719 - root_mean_squared_error: 412.8307
    Epoch 246/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 24ms/step - loss: 170428.6875 - root_mean_squared_error: 412.8301
    Epoch 247/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 66ms/step - loss: 170426.0000 - root_mean_squared_error: 412.8268
    Epoch 248/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 26ms/step - loss: 170421.9375 - root_mean_squared_error: 412.8219
    Epoch 249/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 29ms/step - loss: 170417.4219 - root_mean_squared_error: 412.8164
    Epoch 250/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 29ms/step - loss: 170413.4375 - root_mean_squared_error: 412.8116
    Epoch 251/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 37ms/step - loss: 170410.6094 - root_mean_squared_error: 412.8082
    Epoch 252/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 70ms/step - loss: 170409.5000 - root_mean_squared_error: 412.8069
    Epoch 253/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 246ms/step - loss: 170410.1719 - root_mean_squared_error: 412.8077
    Epoch 254/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 33ms/step - loss: 170412.1875 - root_mean_squared_error: 412.8101
    Epoch 255/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 42ms/step - loss: 170415.0781 - root_mean_squared_error: 412.8136
    Epoch 256/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 62ms/step - loss: 170418.1406 - root_mean_squared_error: 412.8173
    Epoch 257/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 31ms/step - loss: 170420.8281 - root_mean_squared_error: 412.8206
    Epoch 258/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 63ms/step - loss: 170422.5781 - root_mean_squared_error: 412.8227
    Epoch 259/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 29ms/step - loss: 170423.2656 - root_mean_squared_error: 412.8235
    Epoch 260/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 29ms/step - loss: 170422.6875 - root_mean_squared_error: 412.8228
    Epoch 261/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 75ms/step - loss: 170421.0469 - root_mean_squared_error: 412.8208
    Epoch 262/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 30ms/step - loss: 170418.9844 - root_mean_squared_error: 412.8183
    Epoch 263/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 52ms/step - loss: 170416.7969 - root_mean_squared_error: 412.8157
    Epoch 264/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 72ms/step - loss: 170414.9531 - root_mean_squared_error: 412.8134
    Epoch 265/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 50ms/step - loss: 170413.7812 - root_mean_squared_error: 412.8120
    Epoch 266/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 47ms/step - loss: 170413.5781 - root_mean_squared_error: 412.8118
    Epoch 267/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 45ms/step - loss: 170414.2031 - root_mean_squared_error: 412.8126
    Epoch 268/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 34ms/step - loss: 170415.3906 - root_mean_squared_error: 412.8140
    Epoch 269/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 69ms/step - loss: 170416.8281 - root_mean_squared_error: 412.8157
    Epoch 270/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 32ms/step - loss: 170418.3281 - root_mean_squared_error: 412.8175
    Epoch 271/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 23ms/step - loss: 170419.6094 - root_mean_squared_error: 412.8191
    Epoch 272/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 69ms/step - loss: 170420.3594 - root_mean_squared_error: 412.8200
    Epoch 273/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 34ms/step - loss: 170420.4844 - root_mean_squared_error: 412.8202
    Epoch 274/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 52ms/step - loss: 170419.9062 - root_mean_squared_error: 412.8195
    Epoch 275/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 219ms/step - loss: 170418.9688 - root_mean_squared_error: 412.8183
    Epoch 276/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 42ms/step - loss: 170417.7969 - root_mean_squared_error: 412.8169
    Epoch 277/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 69ms/step - loss: 170416.7656 - root_mean_squared_error: 412.8156
    Epoch 278/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 29ms/step - loss: 170416.0781 - root_mean_squared_error: 412.8148
    Epoch 279/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 19ms/step - loss: 170415.5938 - root_mean_squared_error: 412.8142
    Epoch 280/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 88ms/step - loss: 170415.7031 - root_mean_squared_error: 412.8144
    Epoch 281/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 26ms/step - loss: 170416.1719 - root_mean_squared_error: 412.8149
    Epoch 282/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 24ms/step - loss: 170416.8594 - root_mean_squared_error: 412.8158
    Epoch 283/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 77ms/step - loss: 170417.6875 - root_mean_squared_error: 412.8168
    Epoch 284/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 26ms/step - loss: 170418.3750 - root_mean_squared_error: 412.8176
    Epoch 285/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 25ms/step - loss: 170418.8906 - root_mean_squared_error: 412.8182
    Epoch 286/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 81ms/step - loss: 170419.0156 - root_mean_squared_error: 412.8184
    Epoch 287/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 29ms/step - loss: 170418.8281 - root_mean_squared_error: 412.8181
    Epoch 288/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 28ms/step - loss: 170418.4375 - root_mean_squared_error: 412.8177
    Epoch 289/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 63ms/step - loss: 170417.9531 - root_mean_squared_error: 412.8171
    Epoch 290/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 28ms/step - loss: 170417.4531 - root_mean_squared_error: 412.8165
    Epoch 291/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 19ms/step - loss: 170417.0156 - root_mean_squared_error: 412.8159
    Epoch 292/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 71ms/step - loss: 170416.7031 - root_mean_squared_error: 412.8156
    Epoch 293/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 28ms/step - loss: 170416.5625 - root_mean_squared_error: 412.8154
    Epoch 294/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 220ms/step - loss: 170416.7031 - root_mean_squared_error: 412.8156
    Epoch 295/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 28ms/step - loss: 170416.8906 - root_mean_squared_error: 412.8158
    Epoch 296/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 29ms/step - loss: 170417.2344 - root_mean_squared_error: 412.8162
    Epoch 297/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 28ms/step - loss: 170417.6719 - root_mean_squared_error: 412.8167
    Epoch 298/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 27ms/step - loss: 170418.0156 - root_mean_squared_error: 412.8172
    Epoch 299/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 64ms/step - loss: 170418.2344 - root_mean_squared_error: 412.8174
    Epoch 300/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 28ms/step - loss: 170418.3125 - root_mean_squared_error: 412.8175
    Epoch 301/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 25ms/step - loss: 170418.3281 - root_mean_squared_error: 412.8175
    Epoch 302/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 66ms/step - loss: 170418.1406 - root_mean_squared_error: 412.8173
    Epoch 303/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 29ms/step - loss: 170417.8125 - root_mean_squared_error: 412.8169
    Epoch 304/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 22ms/step - loss: 170417.4688 - root_mean_squared_error: 412.8165
    Epoch 305/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 70ms/step - loss: 170417.1875 - root_mean_squared_error: 412.8162
    Epoch 306/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 28ms/step - loss: 170417.0000 - root_mean_squared_error: 412.8159
    Epoch 307/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 19ms/step - loss: 170417.0312 - root_mean_squared_error: 412.8160
    Epoch 308/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 71ms/step - loss: 170417.0781 - root_mean_squared_error: 412.8160
    Epoch 309/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 28ms/step - loss: 170417.2344 - root_mean_squared_error: 412.8162
    Epoch 310/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 24ms/step - loss: 170417.4844 - root_mean_squared_error: 412.8165
    Epoch 311/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 66ms/step - loss: 170417.7656 - root_mean_squared_error: 412.8169
    Epoch 312/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 28ms/step - loss: 170417.9688 - root_mean_squared_error: 412.8171
    Epoch 313/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 246ms/step - loss: 170417.9531 - root_mean_squared_error: 412.8171
    Epoch 314/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 31ms/step - loss: 170418.0156 - root_mean_squared_error: 412.8172
    Epoch 315/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 19ms/step - loss: 170417.9531 - root_mean_squared_error: 412.8171
    Epoch 316/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 72ms/step - loss: 170417.8125 - root_mean_squared_error: 412.8169
    Epoch 317/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 30ms/step - loss: 170417.6406 - root_mean_squared_error: 412.8167
    Epoch 318/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 25ms/step - loss: 170417.5000 - root_mean_squared_error: 412.8166
    Epoch 319/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 109ms/step - loss: 170417.3750 - root_mean_squared_error: 412.8164
    Epoch 320/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 40ms/step - loss: 170417.2969 - root_mean_squared_error: 412.8163
    Epoch 321/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 66ms/step - loss: 170417.2188 - root_mean_squared_error: 412.8162
    Epoch 322/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 29ms/step - loss: 170417.2812 - root_mean_squared_error: 412.8163
    Epoch 323/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 23ms/step - loss: 170417.2969 - root_mean_squared_error: 412.8163
    Epoch 324/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 75ms/step - loss: 170417.4219 - root_mean_squared_error: 412.8164
    Epoch 325/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 27ms/step - loss: 170417.5469 - root_mean_squared_error: 412.8166
    Epoch 326/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 29ms/step - loss: 170417.6094 - root_mean_squared_error: 412.8167
    Epoch 327/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 72ms/step - loss: 170417.7344 - root_mean_squared_error: 412.8168
    Epoch 328/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 26ms/step - loss: 170417.8281 - root_mean_squared_error: 412.8170
    Epoch 329/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 31ms/step - loss: 170417.8906 - root_mean_squared_error: 412.8170
    Epoch 330/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 79ms/step - loss: 170417.9219 - root_mean_squared_error: 412.8170
    Epoch 331/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 293ms/step - loss: 170417.8906 - root_mean_squared_error: 412.8170
    Epoch 332/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 84ms/step - loss: 170417.7031 - root_mean_squared_error: 412.8168
    Epoch 333/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 60ms/step - loss: 170417.5469 - root_mean_squared_error: 412.8166
    Epoch 334/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 38ms/step - loss: 170417.3594 - root_mean_squared_error: 412.8164
    Epoch 335/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 42ms/step - loss: 170417.2969 - root_mean_squared_error: 412.8163
    Epoch 336/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 29ms/step - loss: 170417.3125 - root_mean_squared_error: 412.8163
    Epoch 337/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 67ms/step - loss: 170417.5000 - root_mean_squared_error: 412.8166
    Epoch 338/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 30ms/step - loss: 170417.5781 - root_mean_squared_error: 412.8167
    Epoch 339/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 25ms/step - loss: 170417.6719 - root_mean_squared_error: 412.8167
    Epoch 340/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 76ms/step - loss: 170417.6875 - root_mean_squared_error: 412.8168
    Epoch 341/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 24ms/step - loss: 170417.7188 - root_mean_squared_error: 412.8168
    Epoch 342/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 25ms/step - loss: 170417.6719 - root_mean_squared_error: 412.8167
    Epoch 343/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 83ms/step - loss: 170417.6250 - root_mean_squared_error: 412.8167
    Epoch 344/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 26ms/step - loss: 170417.6094 - root_mean_squared_error: 412.8167
    Epoch 345/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 56ms/step - loss: 170417.6250 - root_mean_squared_error: 412.8167
    Epoch 346/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 192ms/step - loss: 170417.6094 - root_mean_squared_error: 412.8167
    Epoch 347/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 28ms/step - loss: 170417.6250 - root_mean_squared_error: 412.8167
    Epoch 348/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 21ms/step - loss: 170417.6094 - root_mean_squared_error: 412.8167
    Epoch 349/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 146ms/step - loss: 170417.5781 - root_mean_squared_error: 412.8167
    Epoch 350/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 51ms/step - loss: 170417.4531 - root_mean_squared_error: 412.8165
    Epoch 351/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 72ms/step - loss: 170417.3594 - root_mean_squared_error: 412.8164
    Epoch 352/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 85ms/step - loss: 170417.3594 - root_mean_squared_error: 412.8164
    Epoch 353/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 64ms/step - loss: 170417.4375 - root_mean_squared_error: 412.8165
    Epoch 354/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 38ms/step - loss: 170417.4531 - root_mean_squared_error: 412.8165
    Epoch 355/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 52ms/step - loss: 170417.5469 - root_mean_squared_error: 412.8166
    Epoch 356/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 48ms/step - loss: 170417.5781 - root_mean_squared_error: 412.8167
    Epoch 357/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 69ms/step - loss: 170417.6719 - root_mean_squared_error: 412.8167
    Epoch 358/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 31ms/step - loss: 170417.6875 - root_mean_squared_error: 412.8168
    Epoch 359/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 378ms/step - loss: 170417.7344 - root_mean_squared_error: 412.8168
    Epoch 360/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 48ms/step - loss: 170417.6875 - root_mean_squared_error: 412.8168
    Epoch 361/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 90ms/step - loss: 170417.6250 - root_mean_squared_error: 412.8167
    Epoch 362/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 38ms/step - loss: 170417.5469 - root_mean_squared_error: 412.8166
    Epoch 363/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 84ms/step - loss: 170417.5469 - root_mean_squared_error: 412.8166
    Epoch 364/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 45ms/step - loss: 170417.4531 - root_mean_squared_error: 412.8165
    Epoch 365/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 32ms/step - loss: 170417.4219 - root_mean_squared_error: 412.8164
    Epoch 366/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 62ms/step - loss: 170417.4531 - root_mean_squared_error: 412.8165
    Epoch 367/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 29ms/step - loss: 170417.4688 - root_mean_squared_error: 412.8165
    Epoch 368/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 20ms/step - loss: 170417.5469 - root_mean_squared_error: 412.8166
    Epoch 369/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 89ms/step - loss: 170417.6719 - root_mean_squared_error: 412.8167
    Epoch 370/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 66ms/step - loss: 170417.6875 - root_mean_squared_error: 412.8168
    Epoch 371/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 288ms/step - loss: 170417.6719 - root_mean_squared_error: 412.8167
    Epoch 372/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 48ms/step - loss: 170417.7500 - root_mean_squared_error: 412.8168
    Epoch 373/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 27ms/step - loss: 170417.6406 - root_mean_squared_error: 412.8167
    Epoch 374/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 23ms/step - loss: 170417.6094 - root_mean_squared_error: 412.8167
    Epoch 375/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 23ms/step - loss: 170417.4844 - root_mean_squared_error: 412.8165
    Epoch 376/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 75ms/step - loss: 170417.5156 - root_mean_squared_error: 412.8166
    Epoch 377/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 29ms/step - loss: 170417.5469 - root_mean_squared_error: 412.8166
    Epoch 378/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 19ms/step - loss: 170417.5000 - root_mean_squared_error: 412.8166
    Epoch 379/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 70ms/step - loss: 170417.5469 - root_mean_squared_error: 412.8166
    Epoch 380/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 29ms/step - loss: 170417.5469 - root_mean_squared_error: 412.8166
    Epoch 381/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 24ms/step - loss: 170417.6719 - root_mean_squared_error: 412.8167
    Epoch 382/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 67ms/step - loss: 170417.7031 - root_mean_squared_error: 412.8168
    Epoch 383/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 30ms/step - loss: 170417.6875 - root_mean_squared_error: 412.8168
    Epoch 384/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 27ms/step - loss: 170417.5469 - root_mean_squared_error: 412.8166
    Epoch 385/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 75ms/step - loss: 170417.4844 - root_mean_squared_error: 412.8165
    Epoch 386/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 54ms/step - loss: 170417.3906 - root_mean_squared_error: 412.8164
    Epoch 387/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 71ms/step - loss: 170417.5469 - root_mean_squared_error: 412.8166
    Epoch 388/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 64ms/step - loss: 170417.6094 - root_mean_squared_error: 412.8167
    Epoch 389/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 38ms/step - loss: 170417.6719 - root_mean_squared_error: 412.8167
    Epoch 390/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 37ms/step - loss: 170417.6406 - root_mean_squared_error: 412.8167
    Epoch 391/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 74ms/step - loss: 170417.6719 - root_mean_squared_error: 412.8167
    Epoch 392/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 70ms/step - loss: 170417.6562 - root_mean_squared_error: 412.8167
    Epoch 393/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 46ms/step - loss: 170417.6719 - root_mean_squared_error: 412.8167
    Epoch 394/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 272ms/step - loss: 170417.6875 - root_mean_squared_error: 412.8168
    Epoch 395/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 23ms/step - loss: 170417.6094 - root_mean_squared_error: 412.8167
    Epoch 396/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 33ms/step - loss: 170417.6406 - root_mean_squared_error: 412.8167
    Epoch 397/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 28ms/step - loss: 170417.6719 - root_mean_squared_error: 412.8167
    Epoch 398/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 74ms/step - loss: 170417.6406 - root_mean_squared_error: 412.8167
    Epoch 399/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 30ms/step - loss: 170417.6250 - root_mean_squared_error: 412.8167
    Epoch 400/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 24ms/step - loss: 170417.6406 - root_mean_squared_error: 412.8167
    Epoch 401/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 79ms/step - loss: 170417.6250 - root_mean_squared_error: 412.8167
    Epoch 402/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 32ms/step - loss: 170417.6406 - root_mean_squared_error: 412.8167
    Epoch 403/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 26ms/step - loss: 170417.6250 - root_mean_squared_error: 412.8167
    Epoch 404/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 73ms/step - loss: 170417.6406 - root_mean_squared_error: 412.8167
    Epoch 405/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 31ms/step - loss: 170417.6406 - root_mean_squared_error: 412.8167
    Epoch 406/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 76ms/step - loss: 170417.6406 - root_mean_squared_error: 412.8167
    Epoch 407/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 31ms/step - loss: 170417.6406 - root_mean_squared_error: 412.8167
    Epoch 408/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 26ms/step - loss: 170417.6094 - root_mean_squared_error: 412.8167
    Epoch 409/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 69ms/step - loss: 170417.6406 - root_mean_squared_error: 412.8167
    Epoch 410/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 32ms/step - loss: 170417.6406 - root_mean_squared_error: 412.8167
    Epoch 411/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 29ms/step - loss: 170417.6250 - root_mean_squared_error: 412.8167
    Epoch 412/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 69ms/step - loss: 170417.6406 - root_mean_squared_error: 412.8167
    Epoch 413/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 30ms/step - loss: 170417.6406 - root_mean_squared_error: 412.8167
    Epoch 414/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 22ms/step - loss: 170417.6406 - root_mean_squared_error: 412.8167
    Epoch 415/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 84ms/step - loss: 170417.6406 - root_mean_squared_error: 412.8167
    Epoch 416/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 33ms/step - loss: 170417.6406 - root_mean_squared_error: 412.8167
    Epoch 417/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 26ms/step - loss: 170417.6406 - root_mean_squared_error: 412.8167
    Epoch 418/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 71ms/step - loss: 170417.6406 - root_mean_squared_error: 412.8167
    Epoch 419/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 31ms/step - loss: 170417.6406 - root_mean_squared_error: 412.8167
    Epoch 420/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 23ms/step - loss: 170417.6406 - root_mean_squared_error: 412.8167
    Epoch 421/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 46ms/step - loss: 170417.6406 - root_mean_squared_error: 412.8167
    Epoch 422/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 27ms/step - loss: 170417.6406 - root_mean_squared_error: 412.8167
    Epoch 423/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 29ms/step - loss: 170417.6406 - root_mean_squared_error: 412.8167
    Epoch 424/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 215ms/step - loss: 170417.6406 - root_mean_squared_error: 412.8167
    Epoch 425/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 19ms/step - loss: 170417.6406 - root_mean_squared_error: 412.8167
    Epoch 426/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 76ms/step - loss: 170417.6406 - root_mean_squared_error: 412.8167
    Epoch 427/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 41ms/step - loss: 170417.6406 - root_mean_squared_error: 412.8167
    Epoch 428/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 33ms/step - loss: 170417.6406 - root_mean_squared_error: 412.8167
    Epoch 429/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 33ms/step - loss: 170417.6406 - root_mean_squared_error: 412.8167
    Epoch 430/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 30ms/step - loss: 170417.6406 - root_mean_squared_error: 412.8167
    Epoch 431/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 69ms/step - loss: 170417.6406 - root_mean_squared_error: 412.8167
    Epoch 432/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 32ms/step - loss: 170417.6406 - root_mean_squared_error: 412.8167
    Epoch 433/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 29ms/step - loss: 170417.6406 - root_mean_squared_error: 412.8167
    Epoch 434/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 78ms/step - loss: 170417.6406 - root_mean_squared_error: 412.8167
    Epoch 435/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 30ms/step - loss: 170417.6406 - root_mean_squared_error: 412.8167
    Epoch 436/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 30ms/step - loss: 170417.6406 - root_mean_squared_error: 412.8167
    Epoch 437/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 85ms/step - loss: 170417.6406 - root_mean_squared_error: 412.8167
    Epoch 438/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 36ms/step - loss: 170417.6406 - root_mean_squared_error: 412.8167
    Epoch 439/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 64ms/step - loss: 170417.6406 - root_mean_squared_error: 412.8167
    Epoch 440/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 36ms/step - loss: 170417.7344 - root_mean_squared_error: 412.8168
    Epoch 441/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 30ms/step - loss: 170417.6719 - root_mean_squared_error: 412.8167
    Epoch 442/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 78ms/step - loss: 170417.5781 - root_mean_squared_error: 412.8167
    Epoch 443/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 32ms/step - loss: 170417.5938 - root_mean_squared_error: 412.8167
    Epoch 444/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 26ms/step - loss: 170417.5938 - root_mean_squared_error: 412.8167
    Epoch 445/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 80ms/step - loss: 170417.6406 - root_mean_squared_error: 412.8167
    Epoch 446/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 31ms/step - loss: 170417.5938 - root_mean_squared_error: 412.8167
    Epoch 447/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 23ms/step - loss: 170417.5781 - root_mean_squared_error: 412.8167
    Epoch 448/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 77ms/step - loss: 170417.4531 - root_mean_squared_error: 412.8165
    Epoch 449/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 33ms/step - loss: 170417.4219 - root_mean_squared_error: 412.8164
    Epoch 450/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 71ms/step - loss: 170417.6094 - root_mean_squared_error: 412.8167
    Epoch 451/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 225ms/step - loss: 170417.7344 - root_mean_squared_error: 412.8168
    Epoch 452/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 32ms/step - loss: 170417.7500 - root_mean_squared_error: 412.8168
    Epoch 453/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 20ms/step - loss: 170417.6875 - root_mean_squared_error: 412.8168
    Epoch 454/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 78ms/step - loss: 170417.6094 - root_mean_squared_error: 412.8167
    Epoch 455/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 30ms/step - loss: 170417.6094 - root_mean_squared_error: 412.8167
    Epoch 456/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 27ms/step - loss: 170417.5781 - root_mean_squared_error: 412.8167
    Epoch 457/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 59ms/step - loss: 170417.6094 - root_mean_squared_error: 412.8167
    Epoch 458/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 34ms/step - loss: 170417.5781 - root_mean_squared_error: 412.8167
    Epoch 459/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 27ms/step - loss: 170417.6094 - root_mean_squared_error: 412.8167
    Epoch 460/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 72ms/step - loss: 170417.5781 - root_mean_squared_error: 412.8167
    Epoch 461/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 28ms/step - loss: 170417.5156 - root_mean_squared_error: 412.8166
    Epoch 462/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 26ms/step - loss: 170417.4531 - root_mean_squared_error: 412.8165
    Epoch 463/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 85ms/step - loss: 170417.4219 - root_mean_squared_error: 412.8164
    Epoch 464/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 245ms/step - loss: 170417.4531 - root_mean_squared_error: 412.8165
    Epoch 465/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 35ms/step - loss: 170417.5625 - root_mean_squared_error: 412.8166
    Epoch 466/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 28ms/step - loss: 170417.5781 - root_mean_squared_error: 412.8167
    Epoch 467/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 59ms/step - loss: 170417.6562 - root_mean_squared_error: 412.8167
    Epoch 468/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 34ms/step - loss: 170417.7344 - root_mean_squared_error: 412.8168
    Epoch 469/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 28ms/step - loss: 170417.7188 - root_mean_squared_error: 412.8168
    Epoch 470/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 72ms/step - loss: 170417.6719 - root_mean_squared_error: 412.8167
    Epoch 471/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 32ms/step - loss: 170417.5781 - root_mean_squared_error: 412.8167
    Epoch 472/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 23ms/step - loss: 170417.5156 - root_mean_squared_error: 412.8166
    Epoch 473/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 19ms/step - loss: 170417.4844 - root_mean_squared_error: 412.8165
    Epoch 474/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 31ms/step - loss: 170417.5156 - root_mean_squared_error: 412.8166
    Epoch 475/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 78ms/step - loss: 170417.4688 - root_mean_squared_error: 412.8165
    Epoch 476/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 328ms/step - loss: 170417.4531 - root_mean_squared_error: 412.8165
    Epoch 477/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 61ms/step - loss: 170417.5781 - root_mean_squared_error: 412.8167
    Epoch 478/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 84ms/step - loss: 170417.6719 - root_mean_squared_error: 412.8167
    Epoch 479/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 62ms/step - loss: 170417.6719 - root_mean_squared_error: 412.8167
    Epoch 480/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 64ms/step - loss: 170417.6719 - root_mean_squared_error: 412.8167
    Epoch 481/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 41ms/step - loss: 170417.7031 - root_mean_squared_error: 412.8168
    Epoch 482/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 34ms/step - loss: 170417.6875 - root_mean_squared_error: 412.8168
    Epoch 483/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 74ms/step - loss: 170417.7344 - root_mean_squared_error: 412.8168
    Epoch 484/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 36ms/step - loss: 170417.6406 - root_mean_squared_error: 412.8167
    Epoch 485/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 28ms/step - loss: 170417.5625 - root_mean_squared_error: 412.8166
    Epoch 486/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 74ms/step - loss: 170417.4219 - root_mean_squared_error: 412.8164
    Epoch 487/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 32ms/step - loss: 170417.3594 - root_mean_squared_error: 412.8164
    Epoch 488/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 181ms/step - loss: 170417.3281 - root_mean_squared_error: 412.8163
    Epoch 489/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 79ms/step - loss: 170417.3594 - root_mean_squared_error: 412.8164
    Epoch 490/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 43ms/step - loss: 170417.4531 - root_mean_squared_error: 412.8165
    Epoch 491/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 38ms/step - loss: 170417.5625 - root_mean_squared_error: 412.8166
    Epoch 492/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 48ms/step - loss: 170417.5781 - root_mean_squared_error: 412.8167
    Epoch 493/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 31ms/step - loss: 170417.7188 - root_mean_squared_error: 412.8168
    Epoch 494/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 75ms/step - loss: 170417.8281 - root_mean_squared_error: 412.8170
    Epoch 495/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 32ms/step - loss: 170417.8594 - root_mean_squared_error: 412.8170
    Epoch 496/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 23ms/step - loss: 170417.8281 - root_mean_squared_error: 412.8170
    Epoch 497/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 76ms/step - loss: 170417.7344 - root_mean_squared_error: 412.8168
    Epoch 498/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 33ms/step - loss: 170417.5469 - root_mean_squared_error: 412.8166
    Epoch 499/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 221ms/step - loss: 170417.4375 - root_mean_squared_error: 412.8165
    Epoch 500/500
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 49ms/step - loss: 170417.3594 - root_mean_squared_error: 412.8164
    


    
![png](output_18_1.png)
    



    
![png](output_18_2.png)
    


The resulting model is terrible; the red line doesn't align with the blue dots. Furthermore, the loss curve oscillates like a [roller coaster](https://www.wikipedia.org/wiki/Roller_coaster).  An oscillating loss curve strongly suggests that the learning rate is too high.

## Task 4: Find the ideal combination of epochs and learning rate

Assign values to the following two hyperparameters to make training converge as efficiently as possible:

*  learning_rate
*  epochs


```python
# Set the learning rate and number of epochs
learning_rate= ?  # Replace ? with a floating-point number
epochs= ?   # Replace ? with an integer

my_model = build_model(learning_rate)
trained_weight, trained_bias, epochs, rmse = train_model(my_model, my_feature,
                                                         my_label, epochs,
                                                         my_batch_size)
plot_the_model(trained_weight, trained_bias, my_feature, my_label)
plot_the_loss_curve(epochs, rmse)
```


      Cell In[10], line 2
        learning_rate= ?  # Replace ? with a floating-point number
                       ^
    SyntaxError: invalid syntax
    



```python
#@title Double-click to view a possible solution

learning_rate=0.14
epochs=70

my_model = build_model(learning_rate)
trained_weight, trained_bias, epochs, rmse = train_model(my_model, my_feature,
                                                         my_label, epochs,
                                                         my_batch_size)
plot_the_model(trained_weight, trained_bias, my_feature, my_label)
plot_the_loss_curve(epochs, rmse)
```

## Task 5: Adjust the batch size

The system recalculates the model's loss value and adjusts the model's weights and bias after each **iteration**.  Each iteration is the span in which the system processes one batch. For example, if the **batch size** is 6, then the system recalculates the model's loss value and adjusts the model's weights and bias after processing every 6 examples.  

One **epoch** spans sufficient iterations to process every example in the dataset. For example, if the batch size is 12, then each epoch lasts one iteration. However, if the batch size is 6, then each epoch consumes two iterations.  

It is tempting to simply set the batch size to the number of examples in the dataset (12, in this case). However, the model might actually train faster on smaller batches. Conversely, very small batches might not contain enough information to help the model converge.

Experiment with `batch_size` in the following code cell. What's the smallest integer you can set for `batch_size` and still have the model converge in a hundred epochs?


```python
learning_rate=0.05
epochs=100
my_batch_size= ?  # Replace ? with an integer.

my_model = build_model(learning_rate)
trained_weight, trained_bias, epochs, rmse = train_model(my_model, my_feature,
                                                        my_label, epochs,
                                                        my_batch_size)
plot_the_model(trained_weight, trained_bias, my_feature, my_label)
plot_the_loss_curve(epochs, rmse)
```


```python
#@title Double-click to view a possible solution

learning_rate=0.05
epochs=100
my_batch_size=1 # Wow, a batch size of 1 works!

my_model = build_model(learning_rate)
trained_weight, trained_bias, epochs, rmse = train_model(my_model, my_feature,
                                                         my_label, epochs,
                                                         my_batch_size)
plot_the_model(trained_weight, trained_bias, my_feature, my_label)
plot_the_loss_curve(epochs, rmse)

```

## Summary of hyperparameter tuning

Most machine learning problems require a lot of hyperparameter tuning.  Unfortunately, we can't provide concrete tuning rules for every model. Lowering the learning rate can help one model converge efficiently but make another model converge much too slowly.  You must experiment to find the best set of hyperparameters for your dataset. That said, here are a few rules of thumb:

 * Training loss should steadily decrease, steeply at first, and then more slowly until the slope of the curve reaches or approaches zero.
 * If the training loss does not converge, train for more epochs.
 * If the training loss decreases too slowly, increase the learning rate. Note that setting the learning rate too high may also prevent training loss from converging.
 * If the training loss varies wildly (that is, the training loss jumps around), decrease the learning rate.
 * Lowering the learning rate while increasing the number of epochs or the batch size is often a good combination.
 * Setting the batch size to a *very* small batch number can also cause instability. First, try large batch size values. Then, decrease the batch size until you see degradation.
 * For real-world datasets consisting of a very large number of examples, the entire dataset might not fit into memory. In such cases, you'll need to reduce the batch size to enable a batch to fit into memory.

Remember: the ideal combination of hyperparameters is data dependent, so you must always experiment and verify.

#Author : Sandip Pal
#File : local_mnist
#Purpose : Train a model with MNIST training sample
#          Classify our own image
#          Next classify an entire page with numericals


# Get the working directory path
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import tensorflow.keras.models
import matplotlib.image as mpimg
import numpy as np
import os
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

imgmp = mpimg.imread('myhandwritennums/number.png')
print(imgmp.shape)
imgmp = 1 - imgmp  # Done to match black to white and vice versa with Keras db
imgmp = imgmp.reshape(1, 28, 28, 1)


rec_model = tensorflow.keras.models.load_model('mnistmodel')


pred = rec_model.predict(imgmp)
print(pred.argmax())
plt.imshow(imgmp.reshape(28, 28) , cmap='Greys')
plt.show()





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




#Not used
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])


def normalize_img(x):
    """Normalizes images: `uint8` -> `float32`."""
    x = x.astype('float32')
    x = x / 255.0
    return x

# load  dataset
def load_dataset_and_normalize():
    current_dir = os.getcwd()
    # load dataset
    (trainX, trainY), (testX, testY) = mnist.load_data(path=current_dir+'/../db/mnist.npz')
    # reshape dataset to have a single channel
    trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
    testX = testX.reshape((testX.shape[0], 28, 28, 1))
    trainX = normalize_img(trainX)
    testX = normalize_img(testX)
    return trainX, trainY, testX, testY


#define cnn model2
def define_model2():
    model = Sequential()
    model.add(Conv2D(28, kernel_size=(3,3), input_shape=(28, 28, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten()) # Flattening the 2D arrays for fully connected layers
    model.add(Dense(128, activation=tf.nn.relu))
    model.add(Dropout(0.2))
    model.add(Dense(10,activation=tf.nn.softmax))
    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])
    return model


# define a simple cnn model
def define_model():
    model = Sequential([
      Flatten(input_shape=(28, 28, 1)),
      Dense(128,activation='relu'),
      Dense(10, activation='softmax')
    ])

    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(0.001),
        metrics=['accuracy'],
    )
    return model


# define cnn model1
def define_model1():
    model = Sequential()
    model.add(Conv2D(24,kernel_size= (5,5),padding='same',activation='relu',
            input_shape=(28,28,1)))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model
    
    
def define_model5():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(10, activation='softmax'))
    # compile model
    opt = SGD(lr=0.01, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model
    



model = define_model2()
X_train, Y_train, X_test, Y_test = load_dataset_and_normalize()

#Y_train = to_categorical(Y_train);
#Y_test  = to_categorical(Y_test);


model.fit( x=X_train, y=Y_train, epochs=7, batch_size=32, validation_data=(X_test, Y_test) )

model.save('mnistmodel')





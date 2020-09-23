#Author : Sandip Pal
#File : local_Fmnist_training
#Purpose : Train a model with Fashion MNIST training sample
#          Train the model with 60,000 images
#          Next classify an entire page with photos of apparels

# TensorFlow and tf.keras
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import tensorflow.keras.models
import matplotlib.image as mpimg
import os
# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical


def normalize_img(x):
    ''' Normalizes images: uint8 -> float32 '''
    x = x.astype('float32')
    x = x / 255.0
    return x


# load  dataset
def load_dataset_and_normalize1():
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    train_images = train_images / 255.0
    test_images = test_images / 255.0
    return train_images, train_labels, test_images, test_labels


# load  dataset
def load_dataset_and_normalize2():
    #current_dir = os.getcwd()
    fashion_mnist = keras.datasets.fashion_mnist
    # load dataset
    (trainX, trainY), (testX, testY) = fashion_mnist.load_data()
    # reshape dataset to have a single channel
    trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
    testX = testX.reshape((testX.shape[0], 28, 28, 1))
    trainX = normalize_img(trainX)
    testX = normalize_img(testX)
    return trainX, trainY, testX, testY



def define_model1():

    model = Sequential()
    model.add(Flatten(input_shape=(28, 28))) # Flattening the 2D arrays for fully connected layers
    model.add(Dense(128, activation=tf.nn.relu))
    model.add(Dropout(0.2))
    model.add(Dense(10))
    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    return model


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

# define a cnn model
def define_model3():
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



print(tf.__version__)

X_train, Y_train, X_test, Y_test = load_dataset_and_normalize2()



'''
model = define_model1();
model.fit(train_images, train_labels, epochs=15)
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print('\nTest accuracy:', test_acc)

'''

model = define_model2();
model.fit(X_train, Y_train, epochs=12)
test_loss, test_acc = model.evaluate(X_test,  Y_test, verbose=2)
print('\nTest accuracy:', test_acc)



model.save('fmnistmodel')







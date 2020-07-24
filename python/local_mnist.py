#Author : Sandip Pal
#File : local_mnist
#Purpose : Train a model with MNIST training sample
#          Classify our own image
#          Next classify an entire page with numericals


# Get the working directory path
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D


current_dir = os.getcwd()
# Import mnist data stored in the following path: current directory -> mnist.npz

from tensorflow.keras.datasets import mnist
(X_train, Y_train), (X_test, Y_test) = mnist.load_data(path=current_dir+'/../db/mnist.npz')

#Not used
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])


def normalize_img(x):
    """Normalizes images: `uint8` -> `float32`."""
    return tf.cast(x, tf.float32) / 255.



print('MNIST Dataset Shape:')
print('X_train: ' + str(X_train.shape))
print('Y_train: ' + str(Y_train.shape))
print('X_test:  '  + str(X_test.shape))
print('Y_test:  '  + str(Y_test.shape))


X_train_n = normalize_img(X_train)
X_test_n = normalize_img(X_test)

input_shape = (28, 28, 1)

print('========Just A DEBUG Print===============')
image_index = 239
print(X_train_n[image_index])
print(Y_train[image_index])
plt.imshow(X_train_n[image_index] , cmap='Greys')
print('=======================')
plt.show()


# define cnn model1
def define_model1():
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


#define cnn model2
def define_model2():
    model = Sequential()
    model.add(Conv2D(28, kernel_size=(3,3), input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten()) # Flattening the 2D arrays for fully connected layers
    model.add(Dense(128, activation=tf.nn.relu))
    model.add(Dropout(0.2))
    model.add(Dense(10,activation=tf.nn.softmax))
    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])
    return model


# define cnn model simple
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


model = define_model2()

model.fit(
    x=X_train_n,
    y=Y_train,
    epochs=9,
    validation_data=(X_test_n, Y_test),
)

imgmp = mpimg.imread('myhandwritennums/number.png')
print(imgmp.shape)
imgmp = 1 - imgmp  # Done to match black to white and vice versa with Keras db

pred = model.predict(imgmp.reshape(1, 28, 28, 1))

print(imgmp)
print(pred.argmax())
plt.imshow(imgmp.reshape(28, 28) , cmap='Greys')
plt.show()





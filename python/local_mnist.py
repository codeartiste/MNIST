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
current_dir = os.getcwd()



# Import mnist data stored in the following path: current directory -> mnist.npz

from tensorflow.keras.datasets import mnist
(X_train, Y_train), (X_test, Y_test) = mnist.load_data(path=current_dir+'/../db/mnist.npz')

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])






imgmp = mpimg.imread('myhandwritennums/number.png')
print(imgmp.shape)
imgmp = 1 - imgmp



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


print('=======================')
image_index = 56
print(X_train_n[image_index])
print(Y_train[image_index])
plt.imshow(X_train_n[image_index] , cmap='Greys')
print('=======================')
plt.show()

""""
# define cnn model1
def define_model1():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(100, activation='relu', kernel_initializer='he_uniform'))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))
    # compile model
    opt = tf.keras.optimizers.SGD(lr=0.01, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

"""
# define cnn model simple
def define_model():
    model = tf.keras.models.Sequential([
      tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
      tf.keras.layers.Dense(128,activation='relu'),
      tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(0.001),
        metrics=['accuracy'],
    )
    return model


model = define_model()


model.fit(
    x=X_train_n,
    y=Y_train,
    epochs=9,
    validation_data=(X_test_n, Y_test),
)




pred = model.predict(imgmp.reshape(1, 28, 28, 1))

print(imgmp)
print(pred.argmax())
plt.imshow(imgmp.reshape(28, 28) , cmap='Greys')
plt.show()





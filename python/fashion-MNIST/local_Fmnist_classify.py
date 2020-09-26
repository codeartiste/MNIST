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

'''
imgmp = mpimg.imread('myhandwritennums/number.png')
print(imgmp.shape)
imgmp = 1 - imgmp  # Done to match black to white and vice versa with Keras db
imgmp = imgmp.reshape(1, 28, 28, 1)
'''
def normalize_img(x):
    ''' Normalizes images: uint8 -> float32 '''
    x = x.astype('float32')
    x = x / 255.0
    return x


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
    
    



model = tensorflow.keras.models.load_model('fmnistmodel')


X_train, Y_train, X_test, Y_test = load_dataset_and_normalize2()


class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
print(X_train.shape)
print(len(Y_train))
print(Y_train)
print(X_test.shape)
print(len(Y_test))
plt.figure()
plt.imshow(X_train[0].reshape(28,28))
plt.colorbar()
plt.grid(False)
plt.show()


plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(X_train[i].reshape(28,28), cmap=plt.cm.binary)
    plt.xlabel(class_names[Y_train[i]])
plt.show()



predictions = model.predict(X_test)
print(predictions[0])
print(np.argmax(predictions[0]))
print(Y_test[0])

def plot_image(i, predictions_array, true_label, img):
  true_label, img = true_label[i], img[i].reshape(28,28)
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  true_label = true_label[i]
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')


i = 1
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], Y_test, X_test)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i],  Y_test)
plt.show()

i = 22
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], Y_test, X_test)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i],  Y_test)
plt.show()

# Plot the first X test images, their predicted labels, and the true labels.
# Color correct predictions in blue and incorrect predictions in red.
num_rows = 5
num_cols = 3
offset = 50
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i+offset, predictions[i+offset ], Y_test, X_test)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i+offset, predictions[i+offset], Y_test)
plt.tight_layout()
plt.show()


'''
# Grab an image from the test dataset.
img = X_test[11]

print(img.shape)


# Add the image to a batch where it's the only member.
img = (np.expand_dims(img,0))

print(img.shape)


predictions_single = model.predict(img)

print('-------Prediction------')
print(predictions_single)

plot_value_array(11, predictions_single[0], Y_test)
_ = plt.xticks(range(10), class_names, rotation=45)
plt.show()

print(np.argmax(predictions_single[0]))
'''

imgmp = mpimg.imread('randomimages/shirt1.png')
print(imgmp.shape)
imgmp = 1 - imgmp  # Done to match black to white and vice versa with Keras db
imgmp = imgmp.reshape(1, 28, 28, 1)
pred = model.predict(imgmp)
print(pred.argmax())
plt.imshow(imgmp.reshape(28, 28) , cmap='Greys')
plt.xlabel(class_names[pred.argmax()])
plt.show()






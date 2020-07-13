# Get the working directory path
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds
import os
current_dir = os.getcwd()

# Import mnist data stored in the following path: current directory -> mnist.npz

from tensorflow.keras.datasets import mnist
(X_train, Y_train), (X_test, Y_test) = mnist.load_data(path=current_dir+'/../db/mnist.npz')


(ds_train, ds_test), ds_info = tfds.load(
    current_dir+'/../db/mnist.npz',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)

print('MNIST Dataset Shape:')
print('X_train: ' + str(X_train.shape))
print('Y_train: ' + str(Y_train.shape))
print('X_test:  '  + str(X_test.shape))
print('Y_test:  '  + str(Y_test.shape))

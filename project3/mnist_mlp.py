'''Trains a simple deep NN on the MNIST dataset.
Gets to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.
'''

from __future__ import print_function

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
import sklearn.metrics as skm
import numpy as np


batch_size = 128
num_classes = 10
epochs = 20

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


def convert(arr):
    result = np.empty([10000])
    for i in range(10000):
        mx = arr[i][0]
        index = 0
        for j in range(1, 10):
            if arr[i][j] > mx:
                mx = arr[i][j]
                index = j
        result[i] = index
    return result


model = Sequential()
# # one layer of softmax
# model.add(Dense(10, activation='softmax', input_shape=(784,)))

#two hidden layer of 16 neurons
model.add(Dense(256, activation='relu', input_shape=(784,)))
model.add(Dropout(0.2))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))

#original
# model.add(Dense(512, activation='relu', input_shape=(784,)))
# model.add(Dropout(0.2))
# model.add(Dense(512, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(10, activation='softmax'))

model.summary()
model.compile(loss='mean_squared_error',
              optimizer=RMSprop(),
              metrics=['accuracy'])#L2
# model.compile(loss='categorical_crossentropy',
#               optimizer=RMSprop(),
#               metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=batch_size, epochs=epochs,
                    verbose=1, validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
y_pred = model.predict(x_test, batch_size=batch_size, verbose=1)
# print(convert(y_pred).shape)
# print(y_test.shape)
print("confusion matrix: \n")
confusion = skm.confusion_matrix(convert(y_test), convert(y_pred), labels=None, sample_weight=None)
# print(confusion)
np.set_printoptions(precision=2,suppress=True)
confusion_normalized = confusion.astype('float') / confusion.sum(axis=1)[:, np.newaxis]
print(confusion_normalized)

print('Test loss:', score[0])
print('Test accuracy:', score[1])

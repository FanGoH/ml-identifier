from __future__ import print_function
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D,Lambda
from keras import backend as K
from keras.utils import np_utils
import numpy as np
import tensorflow as tf
import json
from datetime import date,datetime
from keras.callbacks import ModelCheckpoint
import os


batch_size = 128
num_classes = 10
epochs = 1000
img_rows, img_cols = 28, 28


def min_max_pool2d(x):
    max_x =  K.pool2d(x, pool_size=(2, 2), strides=(2, 2))
    min_x = -K.pool2d(-x, pool_size=(2, 2), strides=(2, 2))
    return K.concatenate([max_x, min_x], axis=1) # concatenate on channel

def min_max_pool2d_output_shape(input_shape):
    shape = list(input_shape)
    shape[1] *= 2
    return tuple(shape)


def load(f):
    return np.load(f"E:\Manmade\gitHubRepo\ml-identifier\kmnist\{f}",  allow_pickle=True)['arr_0']

x_train = load('kmnist-train-imgs.npz')
x_test = load('kmnist-test-imgs.npz')
y_train = load('kmnist-train-labels.npz')
y_test = load('kmnist-test-labels.npz')

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)



# input image dimensions


x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('{} train samples, {} test samples'.format(len(x_train), len(x_test)))

# Convert class vectors to binary class matrices
y_train = keras.utils.np_utils.to_categorical(y_train, num_classes)
y_test = keras.utils.np_utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3),
                 activation='relu'))
model.add(Lambda(min_max_pool2d, output_shape=min_max_pool2d_output_shape))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=tf.keras.optimizers.Adadelta(),
              metrics=['accuracy'])

filepath = "./ModelCheckPoints/saved-model-{epoch:04d}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=False, save_freq=4690)

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test),callbacks=[checkpoint])

model_json = model.to_json()
name = f"./ModelCheckPoints/model - {date.today()}"
name1 = name + ".json"
name2 = name + ".h5"
with open(name1, "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5



model.save_weights(name2)
print("Saved model to disk")

train_score = model.evaluate(x_train, y_train, verbose=0)
test_score = model.evaluate(x_test, y_test, verbose=0)
print('Train loss:', train_score[0])
print('Train accuracy:', train_score[1])
print('Test loss:', test_score[0])
print('Test accuracy:', test_score[1])

os.system("shutdown /s")
"""imports jijo."""
from functools import total_ordering
from PIL import Image
from keras.engine import training

from keras.models import Sequential

from keras.models import model_from_json
import keras
import numpy as np
import tensorflow as tf
import json
from keras import backend as K

def load(f):
    """Utilities."""
    return np.load(f"E:\Manmade\gitHubRepo\ml-identifier\kmnist\{f}",  allow_pickle=True)['arr_0']

x_train = load('kmnist-train-imgs.npz')
x_test = load('kmnist-test-imgs.npz')
y_train = load('kmnist-train-labels.npz')
y_test = load('kmnist-test-labels.npz')
img_rows, img_cols = 28, 28

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)



# input image dimensions
num_classes = 10

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

y_train = keras.utils.np_utils.to_categorical(y_train, num_classes)
y_test = keras.utils.np_utils.to_categorical(y_test, num_classes)

json_file = open('./ModelCheckPoints/model - 2021-10-29.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load weights into new model
loaded_model.load_weights("./ModelCheckPoints/model - 2021-10-29.h5")
loaded_model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=tf.keras.optimizers.Adadelta(),
              metrics=['accuracy'])

train_score = loaded_model.evaluate(x_train, y_train, verbose=0)
test_score = loaded_model.evaluate(x_test, y_test, verbose=0)
print('Train loss:', train_score[0])
print('Train accuracy:', train_score[1])
print('Test loss:', test_score[0])
print('Test accuracy:', test_score[1])
# y=[]
# x =loaded_model.predict(x_test)
# for i in x:
#     y.append(np.argmax(i))

# total=0
# for x,y in zip(y,y_test):
#     if x==y:
#         total+=1

# total /= len(y_test)

# print(total)
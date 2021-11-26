"""IMPORTS."""
from PIL import Image
from keras.models import model_from_json
from keras import backend as K
import numpy as np
import keras
import json
import tensorflow as tf
from sklearn.metrics import confusion_matrix

print(tf.__version__)

from numba import cuda

def release():
    device = cuda.get_current_device()
    device.reset()





img_rows, img_cols = 28, 28
num_classes = 10

def load(f):
    """Utilitie."""
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


# Convert class vectors to binary class matrices




# input image dimensions



files = ['./ModelCheckPoints/model - 2021-10-29','./ModelCheckPoints3/model - 2021-11-04','./ModelCheckPoints4/model - 2021-11-04','./ModelCheckPoints5/model - 2021-11-05','./ModelCheckPoints6/model - 2021-11-05','./ModelCheckPoints7/model - 2021-11-05','./ModelCheckPoints8/model - 2021-11-05','./ModelCheckPoints9/model - 2021-11-05']

for file in files:
    json_file = open(f"{file}.json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(f"{file}.h5")
    print(f"MODEL : {file}")
    print("TEST PREDICTION")
    y_predicted = [np.argmax(i) for i in loaded_model.predict(x_test)]
    y_test = y_test
    x = confusion_matrix(y_test,y_predicted)
    print("")
    print(x)
    print("TRAIN PREDICTION")
    y_predicted = [np.argmax(i) for i in loaded_model.predict(x_train)]
    x = confusion_matrix(y_train,y_predicted)
    print(x)
    print("")
    print("")
    print("")

        

        
    


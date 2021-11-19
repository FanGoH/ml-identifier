"""IMPORTS."""
from PIL import Image
from keras.models import model_from_json
from keras import backend as K
import numpy as np
import keras
import json
import tensorflow as tf

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
print('{} train samples, {} test samples'.format(len(x_train), len(x_test)))

# Convert class vectors to binary class matrices
y_train = keras.utils.np_utils.to_categorical(y_train, num_classes)
y_test = keras.utils.np_utils.to_categorical(y_test, num_classes)


# input image dimensions


json_file = open('./ModelCheckPoints3/model - 2021-11-04.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)


archivo ='Modelo3.json'
data = open(archivo, 'r')
data = json.load(data)

secuence = [6, 11, 16, 22, 27, 32, 38, 43, 48, 53, 59, 64, 69, 75, 80, 85, 90]

secuence2 = [96,101,106,112,117,122,127,133,138,143,149,154,159,164,170,175]

secuence3 = [180, 186,191,196]

##for i in range(130,201,10):
    
filepath = "./ModelCheckPoints3/model - 2021-11-04.h5" 
    
loaded_model.load_weights(filepath)
loaded_model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=tf.keras.optimizers.Adadelta(),
              metrics=['accuracy'])
train_score = loaded_model.evaluate(x_train, y_train, verbose=0)
test_score = loaded_model.evaluate(x_test, y_test, verbose=0)
print('Train loss:', train_score[0])
print('Train accuracy:', train_score[1])
print('Test loss:', test_score[0])
print('Test accuracy:', test_score[1])
data['epochs'].append(200)
data['train_accuracy'].append(train_score[1])
data['train_loss'].append(train_score[0])
data['test_loss'].append(test_score[0])
data['test_accuracy'].append(test_score[1])
with open(archivo, 'w') as f:
    data={
        'epochs':data['epochs'],
        'train_loss': data['train_loss'],
        'train_accuracy':data['train_accuracy'],
        'test_loss':data['test_loss'],
        'test_accuracy': data['test_accuracy']
    }
    json.dump(data, f)
K.clear_session()
        



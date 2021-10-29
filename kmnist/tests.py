import json
from PIL import Image
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D,Lambda
from keras.models import model_from_json
from keras import backend as K
from keras.utils import np_utils
import numpy as np
import tensorflow as tf
import json


def load(f):
    return np.load(f"E:\Manmade\gitHubRepo\ml-identifier\kmnist\{f}",  allow_pickle=True)['arr_0']

json_file = open('model - 2021-10-28.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model - 2021-10-28.h5")
print("Loaded model from disk")
# evaluate loaded model on test data
image = Image.open('TEST.png')
# convert image to numpy array
images = []
images.append( np.asarray(image))


image = Image.open('Test2.png')
# convert image to numpy array
images.append( np.asarray(image))

image = Image.open('Test3.png')
# convert image to numpy array
images.append( np.asarray(image))

image = Image.open('Test4.png')
image=image.convert('1') 
# convert image to numpy array
images.append( np.asarray(image))

image = Image.open('Test5.png')
# convert image to numpy array
images.append( np.asarray(image))
for i in images:
    print (i.shape)
data = np.array(images)
for i in data:
    print (i.shape)

if K.image_data_format() == 'channels_first':
    data = data.reshape(data.shape[0], 1, 28, 28)
    input_shape = (1, 28, 28)
else:
    data = data.reshape(data.shape[0], 28, 28, 1)
    input_shape = (28, 28, 1)



# input image dimensions


data = data.astype('float32')

data /= 255

probs = loaded_model.predict(data)

for i in probs:
    print( np.argmax(i))
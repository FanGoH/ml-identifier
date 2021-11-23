"""imports jijo."""
from PIL import Image
from keras.models import model_from_json
from keras import backend as K
import numpy as np


def load(f):
    """Load Structure."""
    return np.load(f"E:\Manmade\gitHubRepo\ml-identifier\kmnist\{f}", allow_pickle=True)['arr_0']

file = "./ModelCheckPoints/model - 2021-10-28"
json_file = open(file+".json", 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights(file+".h5")

print(loaded_model.summary())
# print("Loaded model from disk")
# # evaluate loaded model on test data
# # convert image to numpy array
# sources = ['TEST.png', 'Test2.png', 'Test3.png', "Test4.png", 'Test5.png','Test6.png','Test7.png']

# images = []

# for i in sources:
#     image = Image.open(i).convert('L')
#     images.append(np.asarray(image))


# data = np.array(images)


# if K.image_data_format() == 'channels_first':
#     data = data.reshape(data.shape[0], 1, 28, 28)
#     input_shape = (1, 28, 28)
# else:
#     data = data.reshape(data.shape[0], 28, 28, 1)
#     input_shape = (28, 28, 1)


# # input image dimensions


# data = data.astype('float32')

# data /= 255

# probs = loaded_model.predict(data)
# names = ["\u304a", "\u304d", "\u3059", "\u3064", "\u306a",
#          "\u306f", "\u307e", "\u3084", "\u308c", "\u3092"]
# for i in probs:
#     a =  float((i[np.argmax(i)] * 100))
#     b =  f'{a:.2f}'
#     print(f"El caracter corresponde a {names[np.argmax(i)]}, con una probabilidad de { b }")

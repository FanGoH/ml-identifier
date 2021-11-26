from tkinter import *
from PIL import Image

import numpy as np
from PIL import Image, ImageGrab
from keras.models import model_from_json
from keras import backend as K
import io
import win32gui



def load(f):
    """Load Structure."""
    return np.load(f"E:\Manmade\gitHubRepo\ml-identifier\kmnist\{f}", allow_pickle=True)['arr_0']

file = "./ModelCheckPoints/model - 2021-10-29"
json_file = open(file+".json", 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights(file+".h5")
loaded_model.predict()

def getImg(canvas,fileName):

    canvas.update()
    canvas.update_idletasks()

    HWND = canvas.winfo_id()  # get the handle of the canvas
    rect = win32gui.GetWindowRect(HWND)  # get the coordinate of the canvas
    img = ImageGrab.grab(rect)
    # use PIL to convert to PNG 
   
    img =img.resize((28,28))
    img.save(fileName + '.png', 'png')
    return img

root = Tk()
root.title("Paint Application")
root.geometry("280x350")
def paint(event):
    # get x1, y1, x2, y2 co-ordinates
    x1, y1 = (event.x-3), (event.y-3)
    x2, y2 = (event.x+3), (event.y+3)
    color = "white"
    # display the mouse movement inside canvas
    wn.create_oval(x1, y1, x2, y2, fill=color, outline=color)
# create canvas 
wn=Canvas(root, width=280, height=280, bg='black')

# bind mouse event with canvas(wn)
wn.bind('<B1-Motion>', paint)
wn.pack()
def PrintImage():
    img = getImg(wn,"canvas")
    img = img.convert('L')
    img = np.asarray(img)
    data =  [img]
    data = np.array(data)
    if K.image_data_format() == 'channels_first':
        data = data.reshape(data.shape[0], 1, 28, 28)
        input_shape = (1, 28, 28)
    else:
        data = data.reshape(data.shape[0], 28, 28, 1)
        input_shape = (28, 28, 1)

    data = data.astype('float32')

    data /= 255
    probs = loaded_model.predict(data)
    names = ["\u304a", "\u304d", "\u3059", "\u3064", "\u306a",
             "\u306f", "\u307e", "\u3084", "\u308c", "\u3092"]
    for i in probs:
        print(i)
        a =  float((i[np.argmax(i)] * 100))
        b =  f'{a:.2f}'
        print(f"El caracter corresponde a {names[np.argmax(i)]}, con una probabilidad de { b }")

def EraseCanvas():
    wn.delete('all')


boton = Button(text="Predict", command=PrintImage)

Button = Button(text =" Erase", command=EraseCanvas)







Button.place(x=80, y=290)
boton.place(x=180, y=290)
root.mainloop()
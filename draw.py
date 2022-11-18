# TODO let user input drawing of symbol and pass it to trained NN to recognise
from tensorflow import keras
import numpy
import number_recogniser
import os.path
from PIL import ImageTk, Image, ImageDraw
import PIL
from tkinter import *

# Global variables
canvas = None
drawframe = None
image = None
model = None
outputlabel = None
width = 280  # canvas width
height = 280 # canvas height
center = height // 2
white = (255, 255, 255) # canvas back

def paint(event):
    x1, y1 = (event.x - 1), (event.y - 1)
    x2, y2 = (event.x + 1), (event.y + 1)
    canvas.create_oval(x1, y1, x2, y2, fill="black", width=3)
    # drawframe.line([x1, y1, x2, y2], fill="black", width=5)

def recognise():
    smallimage = image.resize((28, 28)) # Compress to size of NN
    smallimage = smallimage.convert('L') # Grayscale
    matrix = numpy.array(smallimage) # Convert drawn layer to matrix
    matrix = 255 - matrix # Express black pixels as higher
    matrix = matrix.astype(numpy.float64) / 255 # Normalise
    pred = model.predict(numpy.reshape(matrix, (1, 28, 28, 1))) # Predict
    print(pred)
    predclass = numpy.argmax(pred)
    outputlabel["text"] = str(predclass)

def clear():
    canvas.delete('all')
    # drawframe.rectangle((0, 0, width, height), fill=(0, 0, 0, 0))

def main():
    # TODO remove global variables and * imports
    global canvas, drawframe, image, model, outputlabel

    # Cache trained model
    pathname = "__pycache__/model"
    if os.path.isdir(pathname):
        model = keras.models.load_model(pathname, compile=True)
    else:
        model = number_recogniser.main() # Train model on training data
        model.save(pathname)

    # This code from https://www.folkstalk.com/2022/10/jupyter-notebook-let-a-user-inputs-a-drawing-with-code-examples.html
    # create a tkinter canvas to draw on
    master = Tk()
    canvas = Canvas(master, width=width, height=height, bg='white')
    canvas.pack()
    # create an empty PIL image and draw object to draw on
    image = PIL.Image.new("RGB", (width, height), white)
    drawframe = ImageDraw.Draw(image)
    canvas.pack(expand=YES, fill=BOTH)
    canvas.bind("<B1-Motion>", paint)
    # Create a label to output the prediction
    outputlabel = Label(master, text='')
    outputlabel.pack()
    # add a button to recognise the image
    recogbutton = Button(text="Recognise", command=recognise)
    recogbutton.pack()
    # add a button to clear the image
    clearbutton = Button(text="Clear", command=clear)
    clearbutton.pack()

    master.mainloop()

# def save():
#     # save image to hard drive
#     filename = "user_input.jpg"
#     output_image.save(filename)


if __name__ == '__main__':
    main()

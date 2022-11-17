# TODO let user input drawing of symbol and pass it to trained NN to recognise

import numpy
import number_recogniser
import pickle
import os.path
from PIL import ImageTk, Image, ImageDraw
import PIL
from tkinter import *

# Global variables
canvas = None
drawframe = None
image = None
model = None
width = 280  # canvas width
height = 280 # canvas height
center = height // 2
white = (255, 255, 255) # canvas back

def paint(event):
    x1, y1 = (event.x - 1), (event.y - 1)
    x2, y2 = (event.x + 1), (event.y + 1)
    canvas.create_oval(x1, y1, x2, y2, fill="black", width=5)
    drawframe.line([x1, y1, x2, y2], fill="black", width=5)

def recognise():
    matrix = numpy.array(image)[1] # Convert drawn layer to matrix
    matrix = 255 - matrix # Express black pixels as higher
    # TODO compress matrix 10 times
    # TODO recognise
    # TODO output prediction

def main():
    global canvas, drawframe, image, model

    # Cache trained model
    filename = "__pycache__/model-v1.cache"
    if os.path.isfile(filename):
        with open(filename, 'rb') as file:
            model = pickle.load(file)
    else:
        model = number_recogniser.main() # Train model on training data
        with open(filename, 'wb') as file:
            pickle.dump(model, file)


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

    # add a button to recognise the image
    button = Button(text="Recognise", command=recognise)
    button.pack()

    # TODO add clear button

    master.mainloop()

# def save():
#     # save image to hard drive
#     filename = "user_input.jpg"
#     output_image.save(filename)


if __name__ == '__main__':
    main()

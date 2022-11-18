import tensorflow
from tensorflow import keras
import numpy
import number_recogniser
import os.path
from PIL import ImageTk, Image, ImageDraw
import PIL
from tkinter import *

def paint(event, canvas, drawframe):
    x1, y1 = (event.x - 1), (event.y - 1)
    x2, y2 = (event.x + 1), (event.y + 1)
    canvas.create_rectangle(x1, y1, x2, y2, fill="black", width=15)
    drawframe.rectangle([x1, y1, x2, y2], fill="black", width=15)

def recognise(image, model, outputlabel):
    smallimage = image.resize((28, 28)) # Compress to size of NN
    smallimage = smallimage.convert('L') # Grayscale
    matrix = numpy.array(smallimage) # Convert drawn layer to matrix
    matrix = 255 - matrix # Express black pixels as higher
    matrix = matrix.astype(numpy.float32) / 255 # Normalise
    pred = model.predict(numpy.reshape(matrix, (1, 28, 28, 1)), verbose=0) # Predict
    prob = tensorflow.nn.softmax(pred).numpy() # Convert prediction activation to probabilities
    predclass = numpy.argmax(pred, axis=-1)[0] # Find label = index of highest activation
    print(max(prob[0]), type(max(prob[0])))
    p = numpy.round(float(max(prob[0])), 3) # Round prediction probability
    outputlabel["text"] = f"{predclass} (p={p})"
    print(prob)

def clear(canvas, drawframe, width, height, outputlabel):
    outputlabel["text"] = ""
    canvas.delete('all')
    drawframe.rectangle((0, 0, width, height), fill=(255, 255, 255, 255))

def main():
    # Cache trained model
    pathname = "__pycache__/model"
    if os.path.isdir(pathname):
        model = keras.models.load_model(pathname, compile=True)
    else:
        model = number_recogniser.main() # Train model on training data
        model.save(pathname)

    # Canvas setup and paint code from:
    # https://www.folkstalk.com/2022/10/jupyter-notebook-let-a-user-inputs-a-drawing-with-code-examples.html
    width = 280  # canvas width
    height = 280 # canvas height
    center = height // 2
    # create a tkinter canvas to draw on
    master = Tk()
    canvas = Canvas(master, width=width, height=height, bg='white')
    canvas.pack()
    # create an empty PIL image and draw object to draw on
    image = PIL.Image.new("RGB", (width, height), (255, 255, 255))
    drawframe = ImageDraw.Draw(image)
    canvas.pack(expand=YES, fill=BOTH)
    canvas.bind("<B1-Motion>", lambda event: paint(event, canvas, drawframe))
    # Create a label to output the prediction
    outputlabel = Label(master, text='')
    outputlabel.pack()
    # add a button to recognise the image
    recogbutton = Button(text="Recognise", command=lambda : recognise(image, model, outputlabel))
    recogbutton.pack()
    # add a button to clear the image
    clearbutton = Button(text="Clear", command=lambda : clear(canvas, drawframe, width, height, outputlabel))
    clearbutton.pack()

    master.mainloop()

if __name__ == '__main__':
    main()

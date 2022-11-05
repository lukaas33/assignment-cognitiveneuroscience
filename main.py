from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy

def main():
    # Get data
    (x_train, y_train), (x_test, y_test) = mnist.load_data() # Handwritten character image dataset

    # Preprocessing
    x_train = x_train / 255.0 # Normalise pixel values
    x_test = x_test / 255.0 # Normalise pixel values
    # OPTIMIZE here

    # Define models
    modelOneLayer = Sequential([
        Flatten(input_shape=(28, 28)), # Flatten 2D image matrix into 1D matrix; input layer with nodes for each pixel
        Dense( # Hidden layer with x nodes, densely connected  TODO change parameters
            128,
            activation='relu'
        ),
        Dense(10) # Output layer with a node for each number
    ])
    modelConv = Sequential([
        Conv2D( # convolutional layer TODO change parameters
            10,
            kernel_size=3,
            input_shape=(28, 28, 1),
            activation="relu",
            padding="same"
        ),
        Flatten(input_shape=(28, 28)), # Flatten 2D image matrix into 1D matrix; input layer with nodes for each pixel
        Dense(10) # Output layer with a node for each number
    ])
    # TODO model three layers
    models = [
        modelOneLayer,
        modelConv
    ]

    # Train models
    for model in models:
        # Set parameters
        model.compile( # TODO cahnge and comment parameters
            optimizer=Adam(0.001),
            loss=SparseCategoricalCrossentropy(from_logits=True),
            metrics=[SparseCategoricalAccuracy()]
        )
        # Train the model
        model.fit(x_train, y_train)
        # Evaluate the model
        model.evaluate(x_test, y_test)

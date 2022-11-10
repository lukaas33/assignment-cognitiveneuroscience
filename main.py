from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.data import Dataset, AUTOTUNE
from tensorflow import cast, float32

# TODO fix CPU
# TODO hyperparameter optimisation of main arguments
# QUESTION does a bigger batch size decrease performance?
# QUESTION what does prefetch do exactly

# Preprocessing functions
normalize_example = lambda image, label: (cast(image, float32) / 255.0, label)

# Main control flow
def main(batchsize = 128, epochs = 2):
    # Get data
    # MNIST Handwritten character image dataset with 60k training examples and 10k testing examples
    # Each example x contains 28x28 'pixels' expressed as integers with range 0-255
    # Each example label y contains a number that the image represents
    (x_train, y_train), (x_test, y_test) = mnist.load_data(
        path='mnist.npz' # Store on disk for improved runtime
    )
    ds_train = Dataset.from_tensor_slices((x_train, y_train))
    ds_test = Dataset.from_tensor_slices((x_test, y_test))

    # Preprocessing
    # Training set
    ds_train = ds_train.map(normalize_example) # Normalise pixel values
    ds_train = ds_train.cache() # Cache (keep in memory) for improved runtime
    ds_train = ds_train.shuffle(len(x_train)) # Shuffle to create random batches (instead of examples with the same label)
    ds_train = ds_train.batch(batchsize) # After each batch of some examples the error is calculated and the model trained; this improves runtime
    # Testing set
    ds_test = ds_test.map(normalize_example)
    ds_test = ds_test.batch(batchsize)
    ds_test = ds_test.cache() # Can cache later as the testing set isn't shuffled

    # Define models
    modelOneLayer = Sequential([ # Simple one layered model
        Flatten(input_shape=(28, 28)), # Flatten 2D image matrix into 1D matrix; input layer with nodes for each pixel
        Dense( # Hidden layer, densely connected with previous layer
            28*28, # same size as input layer; TODO try different sizes
            activation='relu' # TODO change and comment on parameters
        ),
        Dense(10) # Output layer with a node for each number, densely connected with previous
    ])
    modelConv = Sequential([ # Convolutional model
        Conv2D( # convolutional layer TODO change and comment on parameters
            28,
            kernel_size=3,
            input_shape=(28, 28, 1), # QUESTION why the 1?
            activation="relu",
            padding="same" # padding to ensure that the shape of the data stays the same
        ),
        Flatten(input_shape=(28, 28)),
        Dense(10)
    ])
    # TODO model three layers

    # Train and evaluate models
    for model in (modelOneLayer, modelConv):
        # Set parameters
        model.compile( # TODO change and comment on parameters
            optimizer=Adam(0.001),
            loss=SparseCategoricalCrossentropy(from_logits=True),
            metrics=[SparseCategoricalAccuracy()]
        )
        # Train the model
        model.fit(
            ds_train, # Run the examples through the model and update the weights
            epochs=epochs, # Go over dataset 5 times
            use_multiprocessing=True, # Improve runtime by multithreading
            verbose=0 # don't print progress of training
        )
        # Evaluate the model
        model.evaluate(
            ds_test, # Run the testing examples through the model to find the accuracy
            verbose=2 # Print the maximum amount of detail
        )

if __name__ == '__main__':
    main()

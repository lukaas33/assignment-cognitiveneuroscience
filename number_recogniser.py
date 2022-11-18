from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.data import Dataset, AUTOTUNE
from tensorflow import cast, float32

# TODO fix GPU
# TODO hyperparameter optimisation of main arguments
# QUESTION does a bigger batch size decrease performance?
# QUESTION what does prefetch do exactly

# Get dataset
def get_data():
    # MNIST Handwritten character image dataset with 60k training examples and 10k testing examples
    # Each example x contains 28x28 'pixels' expressed as integers with range 0-255
    # Each example label y contains a number that the image represents
    (x_train, y_train), (x_test, y_test) = mnist.load_data( # Load dataset
        path='mnist.npz' # Store on disk for improved runtime
    )
    ds_train = Dataset.from_tensor_slices((x_train, y_train)) # Dataset object allows some useful methods
    ds_test = Dataset.from_tensor_slices((x_test, y_test))
    return ds_train, ds_test

# Preprocess data
def preprocess(dataset, batchsize, shuffle=False, cache=True):
    dataset = dataset.map(lambda image, label: (cast(image, float32) / 255.0, label)) # Normalise pixel values
    if shuffle:
        dataset = dataset.shuffle(len(dataset)) # Shuffle to create random batches (instead of examples with the same label)
    dataset = dataset.batch(batchsize) # After each batch of some examples the error is calculated and the model trained; this improves runtime
    if cache: # Don't want this when storing the model
        dataset = dataset.cache() # Cache (keep in memory) for improved runtime
    return dataset

# Define models
def def_models():
    # TODO add dropout?
    yield Sequential([ # Simple one layered model
        Flatten(input_shape=(28, 28)), # Flatten 2D image matrix into 1D matrix; input layer with nodes for each pixel
        Dense( # Hidden layer, densely connected with previous layer
            28*28, # same size as input layer; TODO try different sizes
            activation='relu' # relu popular, avoid vanishing gradient problem; deactivates some neurons
        ),
        Dense(10) # Output layer with a node for each number, densely connected with previous
    ])
    yield Sequential([ # Convolutional model
        Conv2D( # convolutional layer TODO change and comment on parameters
            28,
            kernel_size=3,
            input_shape=(28, 28, 1), # QUESTION why the 1?
            activation="relu",
            padding="same" # padding to ensure that the shape of the data stays the same
        ),
        # TODO poolen? Reduceert onbelangrijke features
        Flatten(input_shape=(28, 28)),
        Dense(10)
    ])
    # TODO model three layers (depth costs more time and may require more examples)


# Train and evaluate model
def train(ds_train, model, epochs, ds_test=None):
    # Set parameters
    model.compile( # TODO change and comment on parameters
        optimizer=Adam(0.001), # TODO lookup, learning speed?
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
    if ds_test is not None:
        score = model.evaluate(
            ds_test, # Run the testing examples through the model to find the accuracy
            verbose=0 # don't print anything
        )
        return model, score
    else:
        return model


# Main control flow
def main(batchsize=128, epochs=2):
    # Get data
    ds_train, ds_test = get_data()

    # Preprocessing
    ds_train = preprocess(ds_train, batchsize, shuffle=True)
    ds_test = preprocess(ds_test, batchsize)

    # Define models
    models = def_models()

    # Train and evaluate models
    best_model = None
    highest_acc = 0
    for model in models:
        model, score = train(ds_train, model, epochs, ds_test=ds_test) # Train and evaluate
        accuracy = score[1]
        if accuracy > highest_acc: # Save best
            best_model = model
            highest_acc = accuracy
        # TODO log score over time
    return best_model # Return best trained model

if __name__ == '__main__':
    main()

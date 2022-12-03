from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.data import Dataset
from tensorflow import cast, float32
from time import time

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
def preprocess(dataset, batchsize, shuffle=False):
    def normalise(image, label): 
        return cast(image, float32) / 255.0, label

    dataset = dataset.map(normalise) # Normalise pixel values
    if shuffle:
        dataset = dataset.shuffle(len(dataset)) # Shuffle to create random batches (instead of examples with the same label)
    dataset = dataset.batch(batchsize) # After each batch of some examples the error is calculated and the model trained; this improves runtime
    dataset = dataset.cache() # Cache (keep in memory) for improved runtime
    return dataset

# Define models and return them one by one
def def_models(layersize, filters):
    yield (
        "OneLayerNN", # Simple one layered model
        Sequential([ # Feed forward NN
            Flatten(input_shape=(28, 28)), # Flatten 2D image matrix into 1D matrix; input layer with nodes for each pixel
            Dense( # Hidden layer, densely connected with previous layer
                units=layersize, # Number of nodes in this layer
                activation='relu', # relu popular, avoid vanishing gradient problem; deactivates some neurons
                use_bias=True # Can shift activation function, making the network more flexible
            ),
            # Dropout can be used to reduce overfitting, however it was disadvantageous in this task
            Dense(10) # Output layer with a node for each number, densely connected with previous
        ])
    )
    yield (
        "OneConvLayerNN", # Convolutional model
        Sequential([
            Conv2D( # convolutional layer 
                filters=filters, # How many convolutional filters to apply
                kernel_size=3, # Size of convolution window
                use_bias=True,
                input_shape=(28, 28, 1), # Grayscale, 3rd dimension has one channel instead of 3
                activation="relu",
                padding="same" # padding to ensure that the shape of the data stays the same
            ),
            # Pooling can be used to reduce features and thus prevent overfitting, but this was not advantageous for this task
            Flatten(input_shape=(28, 28)),
            Dense(10)
        ])
    )
    yield (
        "ThreeLayerNN", # Three layered model
        Sequential([ 
            Flatten(input_shape=(28, 28)), 
            Dense( 
                units=layersize, 
                activation='relu',
                use_bias=True 
            ),
            Dense( 
                units=layersize, 
                activation='relu',
                use_bias=True 
            ),
            Dense( 
                units=layersize, 
                activation='relu',
                use_bias=True 
            ),
            Dense(10) 
        ])
    )


# Train and evaluate model
def train(ds_train, model, epochs, ds_test=None):
    # Set parameters
    model.compile( 
        optimizer=Adam( # Gradient descent optimisation algorithm
            0.01 # Learning rate, higher will increase convergence speed but make it more susceptible to local optima
        ), 
        loss=SparseCategoricalCrossentropy(from_logits=True), # Loss definition
        metrics=[SparseCategoricalAccuracy()] # Metric to optimise
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
def main(batchsize=128, epochs=2, layersize=250, filters=10):
    # Get data
    ds_train, ds_test = get_data()

    # Preprocessing
    ds_train = preprocess(ds_train, batchsize, shuffle=True)
    ds_test = preprocess(ds_test, batchsize)

    # Define models
    models = def_models(layersize, filters)

    # Train and evaluate models
    best_model = None
    highest_acc = 0
    time_prev = time()
    for name, model in models:
        model, score = train(ds_train, model, epochs, ds_test=ds_test) # Train and evaluate
        loss, accuracy = score
        if accuracy > highest_acc: # Save best
            best_model = model
            highest_acc = accuracy
        runtime, time_prev = time() - time_prev, time()
        
        print(f"Model {name} was evaluated with an average accuracy of {round(accuracy, 3)} and a loss of {round(loss, 3)}. Runtime was {runtime}s")
    return best_model # Return best trained model


if __name__ == '__main__':
    main()

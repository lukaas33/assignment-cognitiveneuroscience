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
    (x_train, y_train), (x_test, y_test) = mnist.load_data( 
         # Load dataset and store on disk for improved runtime
        path='mnist.npz'
    )
    # Dataset object allows for some useful methods
    ds_train = Dataset.from_tensor_slices((x_train, y_train)) 
    ds_test = Dataset.from_tensor_slices((x_test, y_test))
    return ds_train, ds_test

# Preprocess data
def preprocess(dataset, batchsize, shuffle=False):
    # Normalise pixel values by dividing by the max value
    # Required for the model
    def normalise(image, label): 
        return cast(image, float32) / 255.0, label 

    dataset = dataset.map(normalise) 
    if shuffle:
        # Shuffle to create random batches even if dataset is sorted
        dataset = dataset.shuffle(len(dataset))
    # After each 'batch' of x examples the error is calculated and the model trained; 
    # this improves runtime by doing less backpropagation steps
    dataset = dataset.batch(batchsize) 
    # Cache (keep in memory) for improved runtime
    dataset = dataset.cache() 
    return dataset

# Define models and return them one by one
def def_models(layersize, filters):
    # Simple one layered model
    yield (
        "OneLayerNN", 
        # Feed forward NN
        Sequential([ 
            # Flatten 2D image matrix into 1D matrix; 
            # Creates input layer with nodes for each pixel
            Flatten(input_shape=(28, 28)), 
            # Hidden layer, densely connected with previous layer
            Dense(
                # Number of nodes in this layer
                # Larger number increased runtime without significant performance gain
                # Lower number reduced performance
                units=layersize, 
                # relu popular and fast
                # avoids vanishing gradient problem; 
                # But deactivates some neurons
                activation='relu',
                # Can shift activation function, making the network more flexible
                # Useful when many zeroes are present in the previous layer 
                use_bias=True 
            ),
            # Dropout can be used to reduce overfitting, however it was disadvantageous in this task
            # Output layer with a node for each number, densely connected with previous
            Dense(10) 
        ])
    )
    # Convolutional model
    yield (
        "OneConvLayerNN", 
        Sequential([
            # convolutional layer
            # applies convolution to the 2D matrix
            Conv2D( 
                # How many kernels to apply
                # Can detect different kinds of basic features
                # Will significantly affect the performance
                filters=filters,
                # Size of kernel to slide over image 
                # Smaller is often more accurate because it can detect more detail 
                # Too small would be less useful in detecting small features
                # Larger can be faster 
                kernel_size=3, 
                use_bias=True, 
                # Grayscale, 3rd dimension has one channel instead of 3
                input_shape=(28, 28, 1), 
                activation="relu",
                 # padding to ensure that the shape of the data stays the same
                padding="same"
            ),
            # Pooling can be used to reduce features and thus prevent overfitting, but this was not advantageous for this task
            # Needed to connect it to the output layer
            Flatten(input_shape=(28, 28)), 
            Dense(10)
        ])
    )
    yield (
        # Three layered model
        "ThreeLayerNN", 
        Sequential([ 
            Flatten(input_shape=(28, 28)), 
            # Repeating the same hidden layer multiple times
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
        # Gradient descent optimisation algorithm
        optimizer=Adam( 
            # Learning rate, 
            # used to scale weight changes
            # higher will increase convergence speed 
            # but make it more susceptible to local optima
            0.01 
        ), 
        # Is a more compact measure of cat. cross entropy, 
        # which is a measure of similarity between two probability distributions:
        # the prediction and the true output
        loss=SparseCategoricalCrossentropy(from_logits=True), 
        # Measure accuracy
        # Useful for comparing and intepreting the evaluation
        metrics=[SparseCategoricalAccuracy()] 
    )
    # Train the model
    model.fit(
        # Run the training examples through the model which yields a prediction
        # The loss function finds the error of this prediction compared to the true result
        # Backpropagation is used to propagate the error down the layers
        # and update the weights to reduce the loss
        ds_train, 
        # Go over dataset x times
        # More repeats didn't significantly affect the results
        epochs=epochs, 
        # Improve runtime by multithreading
        use_multiprocessing=True, 
        # don't print progress of training
        verbose=0 
    )
    # Evaluate the model if there is evaluation data
    if ds_test is not None:
        score = model.evaluate(
            # Run the testing examples through the model to find predictions
            # The correct output is known 
            # The accuracy is the number of correct predictions divided by the total
            ds_test, 
            # don't print anything,
            verbose=0, 
            return_dict=True
        )
        return model, score
    else:
        return model


# Main control flow
# Most important hyperparameters can be optimised by calling main 
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
        model, score = train(ds_train, model, epochs, ds_test=ds_test) 
        accuracy = score['sparse_categorical_accuracy']
        # Save best model
        if accuracy > highest_acc: 
            best_model = model
            highest_acc = accuracy
        # Calculate runtime
        runtime, time_prev = time() - time_prev, time()
        # Output for the user
        print(f"Model {name} was evaluated with an average accuracy of {round(accuracy, 3)}. Runtime was {runtime}s")
    # Return best trained model
    return best_model 


if __name__ == '__main__':
    main()

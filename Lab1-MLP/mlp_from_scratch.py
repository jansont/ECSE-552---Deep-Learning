import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt

class Layer:
    """
    A layer class to represent a layer of a MLP
    Each layer is composed of num_units neurons, each with num_weights weights and a sigmoid activation
    """
    def __init__(self, num_weights, num_units):
        self.sigmoid = lambda x: 1 / (1 + np.exp(-x)) # Sigmoid activation function
        self.d_sigmoid = lambda x: self.sigmoid(x) * (1 - self.sigmoid(x))  # Derivative of sigmoid
        self.weights = np.random.uniform(-1, 1, size=(num_units, num_weights))  # Initialize weights
        self.biases = np.random.uniform(-1, 1, size=(num_units, 1)) * 0.1   # Initialize biases
        self.weights_grad = np.zeros_like(self.weights) # Initialize weight gradients
        self.biases_grad = np.zeros_like(self.biases)   # Initialize bias gradients
        

    def __call__(self, x):
        """
        Callable method to allow for layer to be called as a function
        """
        return self.forward(x)


    def forward(self, x):
        """
        Forward pass of the layer

        Args:
            x (np.ndarray): The input data for the layer

        Returns:
            Tuple[np.ndarray]: The output of the layer and the activation of the layer
        """
        x = np.dot(self.weights, x) + self.biases
        return x, self.sigmoid(x)


    def backward(self, a, z, prev_gradient):
        """
        Performs a backward pass through the layer

        Args:
            a (np.ndarray): The output of the network at the L-1 layer
            z (np.ndarray): The pre-activation output of the network at the L-th layer
            prev_gradient (np.ndarray): The gradient of the loss with respect to the L-th layer
        """
        # Move back through the sigmoid activation with our gradient
        activation_grad = self.d_sigmoid(z) * prev_gradient
        # Calculate the gradient of the loss with respect to the weights
        self.weights_grad = np.dot(activation_grad, a.T)
        # Calculate the gradient of the loss with respect to the biases
        self.biases_grad = np.sum(activation_grad, axis=1, keepdims=True) / a.shape[1]
        # Calculate our new gradient with respect to our inputs of this layer (outputs of layer L-1)
        new_gradient = np.dot(self.weights.T, activation_grad)
        return new_gradient


class Network:
    """
    A network class to represent a MLP
    Composed of fully connected layers with sigmoid activations
    """
    def __init__(self, input_shape, learning_rate=0.01):
        self.hidden_1 = Layer(input_shape, 2)
        self.hidden_2 = Layer(2, 2)
        self.out = Layer(2, 3)
        self.learning_rate = learning_rate
        self.loss_prime = lambda x, y: x - y # Set loss to 1/2*(x-y)^2 for easy computation of gradients
        
        self.zs = [] # Store the pre-activation outputs of each layer
        self.outputs = [] # Store the outputs of each layer
        self.build_layers()


    def __call__(self, x):
        """
        Callable method to allow for network to be called as a function
        """
        return self.forward(x)


    def build_layers(self):
        self.layers = [
            self.hidden_1,
            self.hidden_2,
            self.out
        ]


    def forward(self, x):
        """
        Performs a forward pass through the network

        Args:
            x (np.ndarray): The train or test data

        Returns:
            np.ndarray: The binary probabilities for each class
        """
        self.outputs = []
        self.zs = []
        
        for layer in self.layers:
            # Save the input as "output of layer -1"
            self.outputs.append(x)
            z, x = layer(x)
            self.zs.append(z)

        return x


    def backward(self, output, expected):
        """
        Performs a backward pass through the network

        Args:
            output (np.ndarray): An array of the network output
            expected (np.ndarray): An array of the expected output
        """
        # Initial gradient of the loss with respect to the output of the network
        gradient = self.loss_prime(output, expected)
        
        for idx, layer in reversed(list(enumerate(self.layers))):
            z = self.zs[idx]
            a = self.outputs[idx]
            gradient = layer.backward(a, z, gradient)
            

    def update(self):
        """
        Update the weights of the network and zero the gradients
        """
        for layer in self.layers:
            layer.weights -= self.learning_rate * layer.weights_grad
            layer.biases -= self.learning_rate * layer.biases_grad
            layer.weights_grad = np.zeros_like(layer.weights_grad)
            layer.biases_grad = np.zeros_like(layer.biases_grad)


def plot_curves(train_error, val_error):
    """
    Plot the training and validation error curves on two subplots
    Args:
        train_error (list): List of training error values
        val_error (list): List of validation error values
    """
    epochs = range(1, len(train_error) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(epochs, train_error, label='Training Error')
    ax1.set_title('Training Error')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Error')
    ax1.legend()

    ax2.plot(epochs[::10], val_error, label='Validation Error')
    ax2.set_title('Validation Error')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Error')
    ax2.legend()

    plt.show()


def main():
    start = time.time()

    # Training Data
    x_train = pd.read_csv('data/training_set.csv', header=None).values
    y_train = pd.read_csv('data/training_labels_bin.csv', header=None).values
    x_val = pd.read_csv('data/validation_set.csv', header=None).values
    y_val = pd.read_csv('data/validation_labels_bin.csv', header=None).values
    N = len(x_train)

    train_mses = []
    val_mses = []
    
    num_feats = x_train.shape[1]
    n_out = y_train.shape[1]

    # hyperparameters (you may change these)
    eta = 0.1 # intial learning rate
    gamma = 0.1 # multiplier for the learning rate
    stepsize = 200 # epochs before changing learning rate
    threshold = 0.105 # stopping criterion
    test_interval = 10 # number of epoch before validating
    max_epoch = 3000

    # Define Architecture of NN
    # [ ] Intialize your network weights and biases here

    network = Network(num_feats, learning_rate=eta)

    for epoch in range(0, max_epoch):
        
        order = np.random.permutation(N) # shuffle data
        
        sse = 0
        for n in range(0, N):
            idx = order[n]

            # get a sample (batch size=1)
            x_in = np.array(x_train[idx]).reshape((num_feats, 1))
            y = np.array(y_train[idx]).reshape((n_out, 1))

            # [ ] do the forward pass here
            # hint: you need to save the output of each layer to calculate the gradients later
            y_preds = network(x_in)
        
            # [ ] compute error and gradients here
            # hint: don't forget the chain rule
            error = (y_preds - y)**2 / 2
            network.backward(y_preds, y)

            squared_error = np.sum(error)
            # [ ] update weights and biases here
            # update weights and biases in output layer 
            network.update()
        
            sse += squared_error

        train_mse = sse/len(x_train)
        train_mses.append(train_mse)
        
        if epoch % test_interval == 0: 
            # [ ] test on validation set here
            val_sse = 0.0
            N_val = len(x_val)
            order_val = np.random.permutation(N_val)
            for n in range(0, N_val):
                idx = order_val[n]
                
                x_in = np.array(x_val[idx]).reshape((num_feats, 1))
                y = np.array(y_val[idx]).reshape((n_out, 1))

                val_pred = network(x_in)
        
                val_sse += np.sum((val_pred - y)**2 / 2)
            
            val_mse = val_sse/len(x_val)
            val_mses.append(val_mse)
            print(f'Epoch {epoch}: Train MSE: {train_mse}\tValidation MSE: {val_mse}')

            # if termination condition is satisfied, exit
            if val_mse < threshold:
                break

        if epoch % stepsize == 0 and epoch != 0:
            eta = eta*gamma
            network.learning_rate = eta
            print('Changed learning rate to lr=' + str(eta))

    plot_curves(train_mses, val_mses)
    
    
if __name__ == '__main__':
    main()
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from load_mnist import load_mnist

class NeuralNetwork:
    def __init__(self, hidden_size=10, eta=0.05, epochs=1000,
                 l2=0.5,
                 minibatches=100,
                 activation='sigmoid',
                 random_seed=None):

        self.hidden_size = hidden_size
        self.eta = eta
        self.epochs = epochs
        self.l2 = l2
        self.activation = activation
        self.minibatches = minibatches
        self.random_seed = random_seed
        self.cost_train_ = []
        self.cost_test_  = []
        self.acc_train_  = []
        self.acc_test_   = []

    
    def _init_params(self, weights_1_shape, bias_1_shape, 
                    weights_2_shape, bias_2_shape,
                    dtype='float64', scale=0.01, random_seed=None):
        """Initialize weight coefficients."""
        if random_seed:
            np.random.seed(random_seed)
        self.W1 = np.random.normal(loc=0.0, scale=scale, size=weights_1_shape)
        self.b1 = np.zeros(shape=bias_1_shape, dtype=dtype)
        self.W2 = np.random.normal(loc=0.0, scale=scale, size=weights_2_shape)
        self.b2 = np.zeros(shape=bias_2_shape, dtype=dtype)
        return self

    def sigmoid(self, x):
        x = np.clip(x, -500, 500 )
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)

    def softmax(self, x):
        exp = np.exp(x - np.max(x, axis=1, keepdims=True))
    
        # Calculating softmax for all examples.
        for i in range(len(x)):
            exp[i] /= np.sum(exp[i])
        
        return exp

    
    def forward(self, X):
        self.z1 = self.linear_forward(X, self.W1, self.b1)
        self.a1 = self.sigmoid(self.z1) if self.activation == 'sigmoid' else self.relu(self.z1)
        self.z2 = self.linear_forward(self.a1, self.W2, self.b2)
        self.a2 = self.softmax(self.z2)
        return self.a2

    def linear_forward(self, X, W, b):
        return X.dot(W) + b

    def compute_cost(self, X, y):
        output = self.predict(X)
        cross_ent = self._cross_entropy(output=output, y_target=y)
        cost = self._cost(cross_ent)
        return cost

    def _cross_entropy(self, output, y_target):
        return - np.sum(np.log(output + 1e-15) * (y_target), axis=1) #+ 1e-15

    def _cost(self, cross_entropy):
        L2_term = self.l2 * (np.sum(self.W1 ** 2) + np.sum(self.W2 ** 2))
        cross_entropy = cross_entropy + L2_term
        return 0.5 * np.mean(cross_entropy)

    def backward(self, X, Y, output):
        self.output_error = output - Y
        self.output_delta = self.output_error
        self.z1_error = self.output_delta.dot(self.W2.T)
        self.z1_delta = self.z1_error * (self.sigmoid_derivative(self.a1) if self.activation == 'sigmoid' else self.relu_derivative(self.a1))
        self.dw2 = self.a1.T.dot(self.output_delta) / X.shape[0]
        self.db2 = np.sum(self.output_delta, axis=0, keepdims=False) / X.shape[0]
        self.dw1 = X.T.dot(self.z1_delta) / X.shape[0]
        self.db1 = np.sum(self.z1_delta, axis=0, keepdims=False) / X.shape[0]
    
    def update_parameters(self):
        self.W2 += -self.eta * (self.dw2 + self.l2 * self.W2)
        self.b2 += -self.eta * self.db2
        self.W1 += -self.eta * (self.dw1 + self.l2 * self.W1)
        self.b1 += -self.eta * self.db1
    

    def train(self, X, Y, X_test, Y_test):
        features = X.shape[1]
        n_classes = Y.shape[1]
        self._init_params(
            weights_1_shape=(features, self.hidden_size),
            bias_1_shape=(self.hidden_size,),
            weights_2_shape=(self.hidden_size, n_classes),
            bias_2_shape=(n_classes,),
            random_seed=self.random_seed)
        # Compute accuracy
        acc_train = self.accuracy(X, Y)
        acc_test = self.accuracy(X_test, Y_test)
        self.acc_train_.append(acc_train)
        self.acc_test_.append(acc_test)

        # Compute cost
        cost_train = self.compute_cost(X, Y)
        cost_test = self.compute_cost(X_test, Y_test)
        self.cost_train_.append(cost_train)
        self.cost_test_.append(cost_test)
        print(f'Epoch: {0} | Lost: {cost_train:.4f} | Train Acc.: {acc_train:.3f}% | Test Acc.: {acc_test:.3f}%')
        
        for epoch in range(self.epochs):
            for idx in self._yield_minibatches_idx(
                    n_batches=self.minibatches,
                    data_ary=Y,
                    shuffle=True):
                # Forward propagation
                output = self.forward(X[idx])


                # Backpropagation
                self.backward(X[idx], Y[idx], output)
                # Update weights
                self.update_parameters()

            # Compute accuracy
            acc_train = self.accuracy(X, Y)
            acc_test = self.accuracy(X_test, Y_test)
            self.acc_train_.append(acc_train)
            self.acc_test_.append(acc_test)

            # Compute cost
            cost_train = self.compute_cost(X, Y)
            cost_test = self.compute_cost(X_test, Y_test)
            self.cost_train_.append(cost_train)
            self.cost_test_.append(cost_test)
            
            print(f'Epoch: {epoch+1} | Lost: {cost_train:.4f} | Train Acc.: {acc_train:.3f}% | Test Acc.: {acc_test:.3f}%')

    def _yield_minibatches_idx(self, n_batches, data_ary, shuffle=True):
        indices = np.arange(data_ary.shape[0])

        if shuffle:
            indices = np.random.permutation(indices)
        if n_batches > 1:
            remainder = data_ary.shape[0] % n_batches

            if remainder:
                minis = np.array_split(indices[:-remainder], n_batches)
                minis[-1] = np.concatenate((minis[-1],
                                            indices[-remainder:]),
                                            axis=0)
            else:
                minis = np.array_split(indices, n_batches)
        else:
            minis = (indices,)

        for idx_batch in minis:
            yield idx_batch

    def predict(self, X):
        z1 = self.linear_forward(X, self.W1, self.b1)
        a1 = self.sigmoid(z1) if self.activation == 'sigmoid' else self.relu(z1)
        z2 = self.linear_forward(a1, self.W2, self.b2)
        a2 = self.softmax(z2)
        return a2
        
    def accuracy(self, X, y_target):
        y_pred = self.predict(X)
        return 100.0 * np.sum(y_pred.argmax(axis=1)==y_target.argmax(axis=1))/len(X) #accuracy score in percentage

def main():
    # Load the data
    X_train, Y_train, X_test, Y_test = load_mnist()

    nn = NeuralNetwork(epochs=100, eta=0.1, hidden_size=50, minibatches=200, l2=0, random_seed=2, activation='relu')
        
    start_time = time.time()
    # train the model
    nn.train(X_train, Y_train, X_test, Y_test)
    end_time = time.time()
    print(f'Training Time: {end_time - start_time:.2f} sec')

    if not os.path.exists('./results'):
        os.makedirs('./results')

    fig = plt.figure()
    plt.plot(range(len(nn.cost_train_)), nn.cost_train_, label='Training set')
    plt.plot(range(len(nn.cost_test_)), nn.cost_test_, label='Testing set')
    plt.title('Losses on training and testing set')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.grid()
    fig.savefig('./results/losses_nn.png')

    fig = plt.figure()
    plt.plot(range(len(nn.acc_train_)), nn.acc_train_, label='Training set')
    plt.plot(range(len(nn.acc_test_)), nn.acc_test_, label='Testing set')
    plt.title('Accuracy on training and testing set')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy(%)')
    plt.legend(loc='lower right')
    plt.grid()
    fig.savefig('./results/accuracy_nn.png')

if __name__ == '__main__':
    main()

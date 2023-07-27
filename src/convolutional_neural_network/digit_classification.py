import numpy as np
from load_mnist import load_mnist
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(         
            nn.Conv2d(
                in_channels=1,              
                out_channels=8,            
                kernel_size=3,              
                stride=1,                   
                padding=1,                  
            ),                              
            nn.ReLU(),                      
            nn.MaxPool2d(kernel_size=2, stride=2),    
        )
        self.conv2 = nn.Sequential(         
            nn.Conv2d(8, 16, 3, 1, 1),     
            nn.ReLU(),                      
            nn.MaxPool2d(2),                
        )
        self.conv3 = nn.Sequential(         
            nn.Conv2d(16, 32, 3, 1, 1),     
            nn.ReLU(),                      
            #nn.MaxPool2d(2),                
        )
        # fully connected layer, output 10 classes
        self.out = nn.Linear(32 * 7 * 7, 10)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output   # return x for visualization

def acc_fn(y_pred, y_true):
    return (y_pred.argmax(1) == y_true.argmax(1)).float().mean().item()

def train(model, loaders, optimizer, loss_fn, n_epochs, X_train, Y_train, X_test, Y_test):
    losses_train, losses_test, acc_train, acc_test = [], [], [], []
    for epoch in range(n_epochs):
        model.train()
        for idx, (X, Y) in enumerate(loaders['train']):
            y_pred = model(X)
            loss = loss_fn(y_pred, Y)
            #losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        with torch.no_grad():
            y_pred_train = model(X_train)
            y_pred_test = model(X_test)
        losses_train.append(loss_fn(y_pred_train, Y_train).item())
        losses_test.append(loss_fn(y_pred_test, Y_test).item())
        acc_train.append(acc_fn(y_pred_train, Y_train))
        acc_test.append(acc_fn(y_pred_test, Y_test))

        print(f'Finished epoch {epoch}, train loss {losses_train[-1]:.4f}, test loss {losses_test[-1]:.4f}, \
            train acc {acc_train[-1]:.4f}, test acc {acc_test[-1]:.4f}')
    
    return losses_train, losses_test, acc_train, acc_test


def plot_fig(losses_train, losses_test, acc_train, acc_test):

    if not os.path.exists('./results'):
        os.makedirs('./results')

    fig = plt.figure()
    plt.plot(range(1, len(losses_train) + 1), losses_train, label='Training set')

    plt.plot(range(1, len(losses_test) + 1), losses_test, label='Testing set')
    plt.title('Losses on training and testing set')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.grid()
    plt.show()
    fig.savefig('./results/losses_softmax.png')


    fig = plt.figure()
    plt.plot(range(1, len(acc_train) + 1), acc_train, label='Training set')
    plt.plot(range(1, len(acc_test) + 1), acc_test, label='Testing set')
    plt.title('Accuracy on training and testing set')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy(%)')
    plt.ylim(0.8, 1.1)
    plt.legend(loc='lower right')
    plt.grid()
    plt.show()
    fig.savefig('./results/accuracy_softmax.png')

def main():

    # Load MNIST dataset
    X_train, Y_train, X_test, Y_test = load_mnist()

    device = ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")


    X_train = torch.from_numpy(X_train).float().to(device)
    Y_train = torch.from_numpy(Y_train).float().to(device)
    X_test = torch.from_numpy(X_test).float().to(device)
    Y_test = torch.from_numpy(Y_test).float().to(device)

    X_train = X_train.reshape(-1,1,28,28)
    X_test = X_test.reshape(-1,1,28,28)

    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
    torch.cuda.empty_cache()

    n_epochs = 100
    batch_size = 200
    learning_rate = 0.1
    n_input = 28*28
    n_output = 10

    train_data = TensorDataset(X_train,Y_train)
    test_data = TensorDataset(X_test,Y_test)
    loaders = {
        'train' : DataLoader(train_data, 
                            batch_size=batch_size, 
                            shuffle=True, 
                            num_workers=0),

        'test'  : DataLoader(test_data, 
                            batch_size=batch_size, 
                            shuffle=True, 
                            num_workers=0),
    }

    # initialize the model
    model = CNN().to(device)
    # train the model
    loss_fn   = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    print("Start training ...")

    losses_train, losses_test, acc_train, acc_test = train(model, loaders, optimizer, loss_fn, n_epochs, X_train, Y_train, X_test, Y_test)

    print("Training finished")
    
    plot_fig(losses_train, losses_test, acc_train, acc_test)

if __name__ == "__main__":
    main()
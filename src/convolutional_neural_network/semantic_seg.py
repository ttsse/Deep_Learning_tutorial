
import numpy as np
import os
import torch
from torch import nn, optim
import matplotlib.pyplot as plt
from torchmetrics.functional import dice
from load_warwick import load_image

# define the model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(         
            nn.Conv2d(
                in_channels=3,              
                out_channels=32,            
                kernel_size=3,              
                stride=1,                   
                padding=1,                  
            ),                              
            nn.ReLU(),                      
            nn.MaxPool2d(kernel_size=2, stride=2),  
            nn.ConvTranspose2d(32, 64, 3, 2, 1, 1),  
        )
        self.conv2 = nn.Sequential(         
            nn.Conv2d(64, 32, 3, 1, 1),     
            nn.ReLU(),                      
            nn.MaxPool2d(2),
            nn.ConvTranspose2d(32, 64, 3, 2, 1, 1),              
        )
        self.conv3 = nn.Sequential(         
            nn.Conv2d(64, 32, 3, 1, 1),     
            nn.ReLU(),                      
        )
        # 1*1 convolution kernel
        self.out = nn.Conv2d(32, out_channels=2, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        output = self.out(x)
        return output
    
# define the prediction function
def predict(model, X):
    model.eval()
    with torch.no_grad():
        y_pred = model(X)#['out']
        y_pred = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
    return y_pred

# define the accuracy function
def acc_fn(y_pred, y_true):
    dice_scores =[]
    y_pred = torch.softmax(y_pred, dim=1)
    y_pred = torch.argmax(y_pred, dim=1)
    y_true_label = torch.argmax(y_true, dim=1)
    for i in range(len(y_pred)):
        dice_score = dice(y_pred[i].int(), y_true_label[i].int(), num_classes=2)
        dice_scores.append(dice_score)

    return torch.mean(torch.stack(dice_scores))

# define the training function
def model_train(model, X_train, Y_train, X_test, Y_test, n_epochs,  loss_fn, optimizer, batch_size=5):
    losses_train, losses_test, acc_train, acc_test = [], [], [], []
    batch_start = torch.arange(0, len(X_train), batch_size)

    model.train()
    for epoch in range(n_epochs):
        for start in batch_start:
            X_batch = X_train[start:start+batch_size]
            Y_batch = Y_train[start:start+batch_size]
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, Y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        with torch.no_grad():
            y_pred_train = model(X_train)
            y_pred_test = model(X_test)
        losses_train.append(loss_fn(y_pred_train, Y_train).item())
        losses_test.append(loss_fn(y_pred_test, Y_test).item())
        acc_train.append(acc_fn(y_pred_train, Y_train).item())
        acc_test.append(acc_fn(y_pred_test, Y_test).item())
        print(f'Finished epoch {epoch}, train loss {losses_train[-1]:.4f}, test loss {losses_test[-1]:.4f}, \
           train acc {acc_train[-1]:.4f}, test acc {acc_test[-1]:.4f}')
    return losses_train, losses_test, acc_train, acc_test



def plot_fig(losses_train, losses_test, acc_train, acc_test):

    if not os.path.exists('./results'):
        os.makedirs('./results')
    # plot
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(losses_train, label='train')
    ax[0].plot(losses_test, label='test')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Loss')
    ax[0].set_title('Training Loss vs Test Loss')
    ax[0].legend()
    ax[1].plot(acc_train, label='train')
    ax[1].plot(acc_test, label='test')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Accuracy')
    ax[1].set_title('Training Accuracy vs Test Accuracy')
    ax[1].legend()
    plt.show()
    fig.savefig('./results/losses_acc.png')


def plot_prediction(model, X_test, Y_test):
    # plot a prediction
    y_pred = predict(model, X_test)

    idx = np.random.randint(0, len(y_pred))
    dice_score = dice(y_pred[idx].int(), torch.argmax(Y_test[idx], dim=0).int(), num_classes=2)
    print(f'Dice Score: {dice_score:.4f}')

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(y_pred[idx].cpu().squeeze().numpy(), cmap='gray')
    ax[0].set_title(f'Prediction (Dice Score: {dice_score:.4f})')
    ax[0].axis('off')
    ax[1].imshow(torch.argmax(Y_test[idx], dim=0).cpu().squeeze().numpy(), cmap='gray')
    ax[1].set_title('Ground Truth')
    ax[1].axis('off')
    #plt.show()
    plt.savefig('./results/prediction.png')

def main():
    # load image data
    X_train, Y_train, X_test, Y_test = load_image()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    # move data to the device
    X_train = torch.from_numpy(np.array(X_train)).float().to(device)
    Y_train = torch.from_numpy(np.array(Y_train)).float().to(device)
    X_test = torch.from_numpy(np.array(X_test)).float().to(device)
    Y_test = torch.from_numpy(np.array(Y_test)).float().to(device)


    #display an sample image and its label

    plt.imshow(X_train[19].cpu().squeeze().numpy()/255, cmap='gray')
    plt.axis('off')
    plt.show()

    plt.imshow(Y_train[19].cpu().squeeze().numpy(), cmap='gray')
    plt.axis('off')
    plt.show()


    # Reshape the data to 4D tensor - (sample_number, num_channels, x_img_size, y_img_size)
    X_train = X_train.permute(0, 3, 1, 2)
    X_test = X_test.permute(0, 3, 1, 2)

    Y_train.unsqueeze_(1)
    Y_test.unsqueeze_(1)

    # Normalize the data
    X_train = X_train/255.0
    X_test = X_test/255.0
    Y_train = Y_train/255.0
    Y_test = Y_test/255.0

    # reshape true labels to 2-channel tensor - (sample_number, num_classes, image_height, image_width)
    n, c, h, w = Y_train.shape

    tmp = torch.zeros(n, 2, h, w, device=device)
    tmp[:, 0, :, :][Y_train[:, 0, :, :]==0] = 1
    tmp[:, 1, :, :][Y_train[:, 0, :, :]==1] = 1

    Y_train = tmp
    n, c, h, w = Y_test.shape
    tmp = torch.zeros(n, 2, h, w, device=device)
    tmp[:, 0, :, :][Y_test[:, 0, :, :]==0] = 1
    tmp[:, 1, :, :][Y_test[:, 0, :, :]==1] = 1
    Y_test = tmp



    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
    torch.cuda.empty_cache()
    # initialize the model
    model = CNN()
    model = model.to(device)
    print(model)

    n_epochs = 100
    learning_rate = 0.001
    batch_size = 10

    loss_fn   = nn.CrossEntropyLoss(reduction='mean')
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print("Start training ...")

    losses_train, losses_test, acc_train, acc_test = model_train(model, X_train, Y_train, X_test, Y_test,
                                                                n_epochs=n_epochs,
                                                                loss_fn=loss_fn,
                                                                optimizer=optimizer,
                                                                batch_size=batch_size)
    
    print("Training finished")
    
    plot_fig(losses_train, losses_test, acc_train, acc_test)
    plot_prediction(model, X_test, Y_test)

if __name__ == "__main__":
    main()
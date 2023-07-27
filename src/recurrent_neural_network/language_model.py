
import numpy as np
import os
import torch
from torch import nn, optim
import matplotlib.pyplot as plt
from load_PTB import load_PTB, preprocess
import time

class ELMAN_RNN(nn.Module):
    '''ELMAN_RNN model
    Args:
        vocab_size: size of vocabulary
        embedding_dim: size of embedding
        hidden_dim: size of hidden layer
        num_layers: number of layers
        dropout: dropout rate'''
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout):
        super(ELMAN_RNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden):
        x = self.embeddings(x)
        out, hidden = self.rnn(x, hidden)
        out = self.linear(out)
        return out, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(self.num_layers, batch_size, self.hidden_dim)

def train(model, model_type, train_data, valid_data, vocab_size, seq_len, batch_size, num_epochs, learning_rate, device, clip=0):
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    train_losses = []
    valid_losses = []
    train_perplexities = []
    valid_perplexities = []

    num_batches = len(train_data[0]) // seq_len
    for epoch in range(num_epochs):
        start_time = time.time()
        train_loss = []
        hidden = model.init_hidden(batch_size)
        for i in range(0, len(train_data[0,:]) - seq_len, seq_len):
            inputs = torch.from_numpy(train_data[:,i:i+seq_len]).to(device)
            targets = torch.from_numpy(train_data[:,(i+1):(i+1)+seq_len]).to(device)

            if model_type == 'LSTM':
                hidden = [h.data.to(device) for h in hidden]
            else:
                hidden = hidden.data.to(device)

            outputs, hidden = model(inputs, hidden)
            loss = criterion(outputs.reshape(-1, vocab_size), targets.reshape(-1))

            model.zero_grad()
            loss.backward()
            if clip != 0:
                nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            
            train_loss.append(loss.item())  
            
            step = (i+1) // seq_len
            if step % 100 == 0:
                print ('Epoch [{}/{}], Step[{}/{}], Loss: {:.4f}, Perplexity: {:5.2f}'
                        .format(epoch+1, num_epochs, step, num_batches, loss.item(), np.exp(loss.item())))
                
        train_loss = np.mean(train_loss)
        train_losses.append(train_loss)
        train_perplexities.append(np.exp(train_loss))
        
        valid_loss = []
        hidden = model.init_hidden(batch_size)
        with torch.no_grad():
            for i in range(0, len(valid_data[0,:]) - seq_len, seq_len):
                inputs = torch.from_numpy(valid_data[:,i:i+seq_len]).to(device)
                targets = torch.from_numpy(valid_data[:,(i+1):(i+1)+seq_len]).to(device)

                if model_type == 'LSTM':
                    hidden = [h.data.to(device) for h in hidden]
                else:
                    hidden = hidden.data.to(device)
                    
                outputs, hidden = model(inputs, hidden)
                loss = criterion(outputs.reshape(-1, vocab_size), targets.reshape(-1))
                valid_loss.append(loss.item())
                
        valid_loss = np.mean(valid_loss)
        valid_losses.append(valid_loss)
        valid_perplexities.append(np.exp(valid_loss))

        end_time = time.time()
        print('Epoch:', '%04d' % (epoch + 1), 'train_loss =', '{:.4f}'.format(train_loss),
                'valid_loss =', '{:.4f}'.format(valid_loss),
                'time =', '{:.4f}'.format(end_time - start_time))
        
    return train_losses, valid_losses, train_perplexities, valid_perplexities

# evaluate
def evaluate(model, model_type, test_data, vocab_size, batch_size, seq_len, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    test_loss = []
    test_perplexity = 0
    hidden = model.init_hidden(batch_size)
    with torch.no_grad():
        for i in range(0, len(test_data[0]) - seq_len, seq_len):
            inputs = torch.from_numpy(train_data[:,i:i+seq_len]).to(device)
            targets = torch.from_numpy(train_data[:,(i+1):(i+1)+seq_len]).to(device)

            if model_type == 'LSTM':
                hidden = [h.data.to(device) for h in hidden]
            else:
                hidden = hidden.data.to(device)

            outputs, hidden = model(inputs, hidden)
            loss = criterion(outputs.reshape(-1, vocab_size), targets.reshape(-1))
            test_loss.append(loss.item())
            
    test_loss = np.mean(test_loss)
    test_perplexity = np.exp(test_loss)
    return test_loss, test_perplexity

# This function takes in the model and character as arguments and returns the next character prediction and hidden state
def predict(model, hidden, model_type, character, int2word, vocab, sample_method, device):

    character = np.array([[vocab[c] for c in character]])
    character = torch.from_numpy(character).to(device)
    
    out, hidden = model(character, hidden)

    if sample_method == 'greedy':
        prob = nn.functional.softmax(out[-1], dim=0).data
        # Taking the class with the highest probability score from the output
        char_ind = torch.max(prob, dim=1)[-1][-1].item()
    elif sample_method == 'sampling':
        word_weights = out[-1].squeeze().exp().cpu()
        char_ind = torch.multinomial(word_weights, num_samples=1)[0].item()
    else:
        raise ValueError('sample_method should be either greedy or sampling')

    return int2word[char_ind], hidden

# This function takes the desired output length and input characters as arguments, returning the produced sentence
def sample(model, model_type, int2word, out_len, vocab, sample_method, device, start='however director'):
    model.eval() # eval mode
    start = start.lower()
    # First off, run through the starting characters
    chars = [ch for ch in start.split()]
    size = out_len - len(chars)
    hidden = model.init_hidden(1)

    if model_type == 'LSTM':
        hidden = [h.data.to(device) for h in hidden]
    else:
        hidden = hidden.data.to(device)

    # Now pass in the previous characters and get a new one
    for ii in range(size):

        char, hidden = predict(model, hidden, model_type, chars, int2word, vocab, sample_method, device)
        chars.append(char)

    return ' '.join(chars)

def plot_fig(train_losses, valid_losses):

    if not os.path.exists('./results'):
        os.makedirs('./results')
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].plot(train_losses, label='train_loss')
    ax[0].set_xlabel('epoch')
    ax[0].set_ylabel('loss')
    ax[0].legend(loc='upper right')

    ax[1].plot(valid_losses, label='valid_loss')
    ax[1].set_xlabel('epoch')
    ax[1].set_ylabel('loss')
    ax[1].legend(loc='upper right')
    fig.savefig('./results/losses.png')

if __name__ == '__main__':

    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
    torch.cuda.empty_cache()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load data
    ptb_data = load_PTB()
    batch_size = 20
    seq_len = 35

    # Preprocess data
    train_data, valid_data, test_data, vocab, int2word = preprocess(ptb_data, batch_size, seq_len)

    # Hyperparameters
    vocab_size = len(vocab)
    embedding_dim = 128
    hidden_dim = 1024
    num_layers = 1
    dropout = 0.0
    num_epochs = 10
    learning_rate = 0.0001

    # Initialize model
    model = ELMAN_RNN(vocab_size, embedding_dim, hidden_dim, num_layers, dropout).to(device)

    # Train
    train_losses, valid_losses, train_perplexities, valid_perplexities = train(model=model,
                                                                                model_type='ELMANRNN',
                                                                                train_data=train_data,
                                                                                valid_data=valid_data,
                                                                                vocab_size=vocab_size, 
                                                                                batch_size=batch_size,
                                                                                seq_len=seq_len,
                                                                                num_epochs=num_epochs, 
                                                                                learning_rate=learning_rate, 
                                                                                device=device
                                                                                )

    # Test
    test_loss, test_perplexity = evaluate(model, "ELMANRNN", test_data, vocab_size, batch_size, seq_len, device)
    print('test_loss =', '{:.4f}'.format(test_loss), 'test_perplexity =', '{:.4f}'.format(test_perplexity))

    # Generate text
    start = 'however director thomas'
    output_length = 50
    sample_method = 'sampling'
    generated_text = sample(model, 'ELMANRNN', int2word, output_length, vocab, sample_method, device, start=start)
    print(generated_text)

    # Plot losses
    plot_fig(train_losses, valid_losses)



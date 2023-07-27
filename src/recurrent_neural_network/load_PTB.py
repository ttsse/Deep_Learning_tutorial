import numpy as np
# Load data
def load_data(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        data = []
        for line in lines:
            data.append(line.strip())
        return data
    

def load_PTB():
    paths = ['train.txt', 'valid.txt', 'test.txt']
    data = []
    for path in paths:
        data.append(load_data(path))
    return data

# Build vocabulary
def build_vocab(data):
    vocab = {}
    for word in data:
        if word not in vocab:
            vocab[word] = len(vocab)
    return vocab

# Convert data to indices
def convert_to_indices(data, vocab):
    return [vocab[word] for word in data]

# Create batches
def create_batches(data, batch_size, seq_len):
    num_batches = len(data) // batch_size
    data = data[:num_batches * batch_size]
    data = np.array(data)

    return data.reshape(batch_size, -1)

def add_eos(data):
    return [line + ' <eos>' for line in data]   

def preprocess(ptb_data, batch_size, seq_len):
        # Add <eos> to the end of each sentence
    ptb_data = [add_eos(data) for data in ptb_data]
    train_data, valid_data, test_data = [' '.join(data).split(' ') for data in ptb_data]


    # Load data
    data_all = train_data + valid_data + test_data
    vocab = build_vocab(data_all)
    int2word = dict((i, c) for i, c in enumerate(vocab))

    # Convert data to indices
    train_data = convert_to_indices(train_data, vocab)
    valid_data = convert_to_indices(valid_data, vocab)
    test_data = convert_to_indices(test_data, vocab)

    # Create batches
    train_data = create_batches(train_data, batch_size, seq_len)
    valid_data = create_batches(valid_data, batch_size, seq_len)
    test_data = create_batches(test_data, batch_size, seq_len)

    return train_data, valid_data, test_data, vocab, int2word
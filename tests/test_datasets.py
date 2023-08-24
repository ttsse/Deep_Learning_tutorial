import pytest
import logging

import sys
import os

sys.path.append(os.path.abspath(os.path.join('../src')))
import linear_regression.load_data as load_data

logging.basicConfig(level=logging.DEBUG)

def test_load_data():
    data = load_data.load_auto('../datasets/Auto.csv')
    assert data[0].shape == (392, 9)
    assert data[1].shape == (392, 1)

def test_load_data_bad_path():
    with pytest.raises(FileNotFoundError):
        data = load_data.load_auto('../datasets/Auto2.csv')

import neural_network.load_mnist as load_mnist

def test_mnist_exists():
    assert os.path.exists('../datasets/MNIST.zip')

def test_load_mnist():

    if not os.path.exists("MNIST"):
        os.system("unzip ../datasets/MNIST.zip -d .")

    data = load_mnist.load_mnist()
    os.system("rm -rf MNIST")

    assert data[0].shape == (60000, 784)
    assert data[1].shape == (60000, 10)
    assert data[2].shape == (10000, 784)
    assert data[3].shape == (10000, 10)

import convolutional_neural_network.load_warwick as load_warwick

def test_warwick_exists():
    assert os.path.exists('../datasets/WARWICK.zip')

def test_load_warwick():
    
        if not os.path.exists("WARWICK"):
            os.system("unzip ../datasets/WARWICK.zip -d .")
    
        data = load_warwick.load_image()
        os.system("rm -rf WARWICK")
    
        assert data[0][0].shape == (128, 128, 3)
        assert data[1][0].shape == (128, 128)

import recurrent_neural_network.load_PTB as load_PTB

def test_PTB_exists():
    assert os.path.exists('../datasets/PTB.zip')

def test_load_PTB():
        
    if not os.path.exists("train.txt"):
        os.system("unzip ../datasets/PTB.zip -d .")

    data = load_PTB.load_PTB()
    os.system("rm -rf *.txt")

    assert len(data[0]) == 42068
    assert len(data[1]) == 3370
    assert len(data[2]) == 3761
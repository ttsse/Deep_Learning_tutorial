# Deep Learning Tutorial
This repository contains tutorial codes for multiple machine learning task. The codes are developed for the assignments of Deep Learning course at UU in the begining. To demo the contents of TTSSE course, the codes and running environment are re-consolidated afterwards. It could be taken as a simple guide which the students in machine learning could run step by step to learn the fundamentional knowledge of AI and ML.

-------------------
## 0. Environment Setup
    python3
    pip3
    docker
    jupyter-notebook

## 1. Linear Regression Model
* Dataset: datasets/Auto.csv
* Packages needed: numpy matplotlib pandas
* Task Introduction: It contains the codes for linear regression model which are implemented manually instead of using the existing packages.
* How to run the code:
    * run in notebook
    * run on bare metal
    ```bash
        $ cd linear_regression
        $ cp ../../datasets/Auto.csv .
        $ pip install -r requirements.txt
        $ python main.py
    ```
    * run inside a container
    ```bash
        $ docker pull zlsundocker/deep-learning:linear-regression
        $ docker run -v $(pwd)/results:/app/linear_regression/results -t zlsundocker/deep-learning:linear-regression
    ```
* Files generated under results folder: 
    * cost1.png
    * cost2.png
    * prediction.png
## 2. Neural Network Model
* Dataset: datasets/MNIST.zip
* Packages needed: numpy matplotlib imageio
* Task Introduction: It contains the codes for neural network model which are implemented manually instead of using the existing packages.
* How to run the code for softmax regression model:
    * run in notebook
    * run on bare metal
    ```bash
        $ cd neural_network
        $ cp ../../datasets/MNIST.zip .
        $ unzip MNIST.zip
        $ pip install -r requirements.txt
        $ python softmax_regression.py
    ```
    * run inside a container
    ```bash
        $ docker pull zlsundocker/deep-learning:nn
        $ docker run -v $(pwd)/results:/app/neural_network/results -t zlsundocker/deep-learning:nn
    ```
* Files generated under results folder: 
    * cost1.png
    * cost2.png
    * prediction.png

## 3. Convolutional Neural Network Model
* Dataset: datasets/MNIST.zip
* Packages needed: numpy matplotlib imageio
* Task Introduction: It contains the codes for convolutional neural network model which are implemented manually instead of using the existing packages.
* How to run the code for softmax regression model:
    * classificaiton task
        * run in notebook 
        * run on bare metal
        ```bash
            $ cd convolutional_neural_network
            $ cp ../../datasets/MNIST.zip .
            $ unzip MNIST.zip
            $ pip install -r requirements.txt
            $ python digit_classification.py
        ```
        * run inside a container
        ```bash
            $ docker pull zlsundocker/deep-learning:cnn
            $ docker run --env task=classification -v $(pwd)/results:/app/convolutional_neural_network/results -t zlsundocker/deep-learning:cnn
        ```
        * Files generated under results folder: 
            * cost1.png
            * cost2.png
            * prediction.png
    * semantic segmentation
        * run in notebook
        * run on bare metal
        ```bash
            $ cd convolutional_neural_network
            $ cp ../../datasets/WARWICK.zip .
            $ unzip WARWICK.zip
            $ pip install -r requirements.txt
            $ python semantic_seg.py
        ```
        * run inside a container
        ```bash
            $ docker pull zlsundocker/deep-learning:cnn
            $ docker run --env task=segmentation -v $(pwd)/results:/app/convolutional_neural_network/results -t zlsundocker/deep-learning:cnn
        ```
## 4. Recurrent Neural Network Model
* Dataset: datasets/PTB.zip
* Packages needed: numpy matplotlib imageio
* Task Introduction: It contains the codes for recurrent neural network model which are implemented manually instead of using the existing packages.
* How to run the code for softmax regression model:
    * run in notebook
    * run on bare metal
    ```bash
        $ cd recurrent_neural_network
        $ cp ../../datasets/PTB.zip .
        $ unzip PTB.zip
        $ pip install -r requirements.txt
        $ python language_model.py
    ```
    * run inside a container
    ```bash
        $ docker pull zlsundocker/deep-learning:rnn
        $ docker run -v $(pwd)/results:/app/recurrent_neural_network/results -t zlsundocker/deep-learning:rnn
    ```
    * Files generated under results folder:
        * cost1.png
        * cost2.png
        * prediction.png
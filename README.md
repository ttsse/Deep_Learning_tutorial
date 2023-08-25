# Deep Learning Tutorial
This repository contains tutorial codes for multiple machine learning task, e.g., regression, classification, semantic segmentation, and generative prediction. part of the codes are originally developed for the assignments of Deep Learning course at UU. To reflect the contents of TTSSE course, the codes and running environment are re-consolidated afterwards. It could be taken as a tutorial by the machine learning beginners to study AI and ML from simple models to deep models.

-------------------
## 0. Environment Setup
* To reproduce the results, the following softwares are needed:
    * python3
    * pip3
    * docker
    * jupyter-notebook

## 1. Linear Regression
* Dataset
    * datasets/Auto.csv.
    * Description: The dataset was used in the 1983 American Statistical Association Exposition. It contains Gas mileage, horsepower, and other variables for 392 vehicles.
* Packages needed: numpy matplotlib pandas
* Task Introduction:  Implement linear regression model from scratch instead of using the high-level packages. Fit two linear regression models for two different choices of inputs.
* How to run the code:
    * run in notebook
    * run on bare metal
    ```bash
        $ cd linear_regression
        $ cp ../../datasets/Auto.csv .
        $ python3 -m venv .venv && source .venv/bin/activate
        $ pip install -r requirements.txt
        $ python main.py
    ```
    * run inside a container
    ```bash
        $ docker pull zlsundocker/deep-learning:linear-regression
        $ docker run -v $(pwd)/results:/app/linear_regression/results -t zlsundocker/deep-learning:linear-regression
    ```
* Files generated under results folder:
    * cost1.png. Loss curve for the first model.
    * cost2.png. Accuracy curve for the second model.
    * prediction.png. Prediction results for the second model.
## 2. Neural Network Model
* Dataset
    * datasets/MNIST.zip
    * Description: The MNIST database of hand-written digits, available from this page, has a training set of 60,000 examples, and a test set of 10,000 examples.
* Packages needed: numpy matplotlib imageio
* Task Introduction: It contains the codes for neural network model and softmax regression model. The models are for digit classification task and the dataset is MNIST.
* How to run the code:
    * softmax regression model
        * run in notebook
        * run on bare metal
        ```bash
            $ cd neural_network
            $ cp ../../datasets/MNIST.zip .
            $ unzip MNIST.zip
            $ python3 -m venv .venv && source .venv/bin/activate
            $ pip install -r requirements.txt
            $ python softmax_regression.py
        ```
        * run inside a container
        ```bash
            $ docker pull zlsundocker/deep-learning:nn
            $ docker run -e task=softmax -v $(pwd)/results:/app/neural_network/results -t zlsundocker/deep-learning:nn
        ```
        * Files generated under results folder: 
            * losses_softmax.png. Loss curve for the softmax regression model.
            * accuracy_softmax.png. Accuracy curve for the softmax regression model.
            * softmax_weights.png. The weights of the softmax regression model for each digit.
    * neural network model
        * run in notebook
        * run on bare metal
        ```bash
            $ cd neural_network
            $ cp ../../datasets/MNIST.zip .
            $ unzip MNIST.zip
            $ python3 -m venv .venv && source .venv/bin/activate
            $ pip install -r requirements.txt
            $ python nn.py
        ```
        * run inside a container
        ```bash
            $ docker pull zlsundocker/deep-learning:nn
            $ docker run -e task=nn -v $(pwd)/results:/app/neural_network/results -t zlsundocker/deep-learning:nn
        ```
        * Files generated under results folder: 
            * losses_nn.png. Loss curve for the neural network model.
            * accuracy_nn.png. Accuracy curve for the neural network model.
## 3. Convolutional Neural Network
* Dataset
    * datasets/MNIST.zip
    * datasets/WARWICK.zip
    * Description: For the MNIST database, see above. The WARWICK dataset is a dataset for semantic segmentation task.
* Packages needed: numpy matplotlib imageio
* Task Introduction: It contains the codes for convolutional neural network model using the pytorch software package. The model is for digit classification task and the dataset is MNIST. It also contains the codes for semantic segmentation task using the pytorch software package. The model is for semantic segmentation task and the dataset is WARWICK.
* How to run the code:
    * classificaiton task
        * run in notebook 
        * run on bare metal
        ```bash
            $ cd convolutional_neural_network
            $ cp ../../datasets/MNIST.zip .
            $ unzip MNIST.zip
            $ python3 -m venv .venv && source .venv/bin/activate
            $ pip install -r requirements.txt
            $ python digit_classification.py
        ```
        * run inside a container
        ```bash
            $ docker pull zlsundocker/deep-learning:cnn
            $ docker run --env task=classification -v $(pwd)/results:/app/convolutional_neural_network/results -t zlsundocker/deep-learning:cnn
        ```
        * Files generated under results folder: 
            * losses_nn.png. Loss curve for the cnn model.
            * accuracy_nn.png. Accuracy curve for the cnn model.
    * semantic segmentation
        * run in notebook
        * run on bare metal
        ```bash
            $ cd convolutional_neural_network
            $ cp ../../datasets/WARWICK.zip .
            $ unzip WARWICK.zip
            $ python3 -m venv .venv && source .venv/bin/activate
            $ pip install -r requirements.txt
            $ python semantic_seg.py
        ```
        * run inside a container
        ```bash
            $ docker pull zlsundocker/deep-learning:cnn
            $ docker run --env task=segmentation -v $(pwd)/results:/app/convolutional_neural_network/results -t zlsundocker/deep-learning:cnn
        ```
        * Files generated under results folder: 
            * losses_acc.png. Loss and accuracy curves for the cnn model.
            * prediction.png. Prediction results for the cnn model.
## 4. Recurrent Neural Network
* Dataset
    * datasets/PTB.zip
    * Description: The Penn Tree Bank (PTB) dataset is a dataset for language model task. It is a popular dataset in the field of natural language processing. With the dataset, the model is trained to predict the next word given the previous words.
* Packages needed: numpy matplotlib imageio
* Task Introduction: It contains the codes for recurrent neural network model.
* How to run the code:
    * run in notebook
    * run on bare metal
    ```bash
        $ cd recurrent_neural_network
        $ cp ../../datasets/PTB.zip .
        $ unzip PTB.zip
        $ python3 -m venv .venv && source .venv/bin/activate
        $ pip install -r requirements.txt
        $ python language_model.py
    ```
    * run inside a container
    ```bash
        $ docker pull zlsundocker/deep-learning:rnn
        $ docker run -v $(pwd)/results:/app/recurrent_neural_network/results -t zlsundocker/deep-learning:rnn
    ```
    * Files generated under results folder:
        * losses.png. Loss curve for the rnn model.

## Test
* To run the test, run the following commands:
    ```bash
        $ cd test
        $ pytest
    ```
## License
The repository is licensed under the Apache License, Version 2.0. (http://www.apache.org/licenses/LICENSE-2.0)

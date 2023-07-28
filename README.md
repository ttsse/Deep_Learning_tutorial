# Deep Learning Tutorial
This repository contains tutorial codes for multiple machine learning task, e.g., regression, classification, semantic segmentation, and generative prediction. The codes are developed for the assignments of Deep Learning course at UU in the begining. To demonstrate the contents of TTSSE course, the codes and running environment are re-consolidated afterwards. Moreover, to make it reproducible by others, docker container is leveraged. It could be taken as a tutorial by which the students in machine learning could learn the fundamentional knowledge of AI and ML.

-------------------
## 0. Environment Setup
* To reproduce the results, the following softwares are needed:
    * python3
    * pip3
    * docker
    * jupyter-notebook

## 1. Linear Regression Model
* Dataset
    * datasets/Auto.csv.
    * Description: The dataset was used in the 1983 American Statistical Association Exposition. Gas mileage, horsepower, and other information for 392 vehicles.
* Packages needed: numpy matplotlib pandas
* Task Introduction: In the codes, linear regression model is implemented from scratch instead of using the existing packages. The task is to fit two linear regression models for the two choices of inputs.
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
* Dataset
    * datasets/MNIST.zip
    * Description: The MNIST database of handwritten digits, available from this page, has a training set of 60,000 examples, and a test set of 10,000 examples.
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
            $ pip install -r requirements.txt
            $ python softmax_regression.py
        ```
        * run inside a container
        ```bash
            $ docker pull zlsundocker/deep-learning:nn
            $ docker run -e task=softmax -v $(pwd)/results:/app/neural_network/results -t zlsundocker/deep-learning:nn
        ```
        * Files generated under results folder: 
            * cost1.png
            * cost2.png
            * prediction.png
    * neural network model
        * run in notebook
        * run on bare metal
        ```bash
            $ cd neural_network
            $ cp ../../datasets/MNIST.zip .
            $ unzip MNIST.zip
            $ pip install -r requirements.txt
            $ python nn.py
        ```
        * run inside a container
        ```bash
            $ docker pull zlsundocker/deep-learning:nn
            $ docker run -e task=nn -v $(pwd)/results:/app/neural_network/results -t zlsundocker/deep-learning:nn
        ```
        * Files generated under results folder: 
            * cost1.png
            * cost2.png
            * prediction.png
## 3. Convolutional Neural Network Model
* Dataset
    * datasets/MNIST.zip
    * datasets/WARWICK.zip
    * Description: For the MNIST database, see above. The WARWICK dataset is a dataset for semantic segmentation task.
* Packages needed: numpy matplotlib imageio
* Task Introduction: It contains the codes for convolutional neural network model using the pytorch software package. The model is for digit classification task and the dataset is MNIST. The other model is also designed for semantic segmentation task and the dataset is WARWICK.
* How to run the code:
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
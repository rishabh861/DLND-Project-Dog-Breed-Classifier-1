# [DLND] Project: Dog-Breed Classifier

Project code for Udacity's DeepLearning Nanodegree program 2018.
In this project, I develop an algorithm for a Dog Identification Application, a Dog-Breed-Image-Classifier.

## Getting Started

In this notebook, I will make the first steps towards developing an algorithm that could be used as part of a mobile or web app. At the end of this project, I code will accept any user-supplied image as input. If a dog is detected in the image, it will provide an estimate of the dog's breed. If a human is detected, it will provide an estimate of the dog breed that is most resembling. The image below displays potential sample output of your finished project.

![Sample Dog Output](./assets/sample_dog_output.png)

In this real-world setting, I'll need to piece together a series of models to perform different tasks; for instance, the algorithm that detects humans in an image will be different from the CNN that infers dog breed.

### Prerequisites

Thinks you have to install or installed on your working machine:

* Python 3.7
* Numpy (win-64 v1.15.4)
* Pandas (win-64 v0.23.4)
* Matplotlib (win-64 v3.0.2)
* Jupyter Notebook
* Torchvision (win-64 v0.2.1)
* PyTorch (win-64 v0.4.1)

### Environment:
* [Miniconda](https://conda.io/miniconda.html) or [Anaconda](https://www.anaconda.com/download/)

### Installing

Use the package manager [pip](https://pip.pypa.io/en/stable/) or
[miniconda](https://conda.io/miniconda.html) or [Anaconda](https://www.anaconda.com/download/) to install your packages.  
A step by step guide to install the all necessary components in Anaconda for a Windows-64 System:
```bash
conda install -c conda-forge numpy
conda install -c conda-forge pandas
conda install -c conda-forge matplotlib
pip install torchvision
conda install -c pytorch pytorch
```

## Jupyter Notebook
* `dog_app.ipynb`

This jupyter notebook describe the whole project from udacity, from the beginning to the end.

## Download the Datasets

To train and test the model, you need to download these 2 datasets:

* Download the [dog dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip).
Unzip the folder and place it in this project's home directory, at the location `/dog_images`.
* Download the [human dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/lfw.zip).
Unzip the folder and place it in the home directory, at location `/lfw`.


## Running the project

The whole project is located in the file `dog_app.ipynb` and it's include the training and the prediction part.

### Model architecture

I choose a pre-trained network `vgg19` with a sequence of convolutional and max pooling layers, two fully connected hidden layer and one fully connected output layer with output size of 133 class integers.

```python
VGG(
  (features): Sequential(
    (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): ReLU(inplace)
    (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (3): ReLU(inplace)
    (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (6): ReLU(inplace)
    (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (8): ReLU(inplace)
    (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (11): ReLU(inplace)
    (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (13): ReLU(inplace)
    (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (15): ReLU(inplace)
    (16): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (17): ReLU(inplace)
    (18): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (19): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (20): ReLU(inplace)
    (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (22): ReLU(inplace)
    (23): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (24): ReLU(inplace)
    (25): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (26): ReLU(inplace)
    (27): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (29): ReLU(inplace)
    (30): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (31): ReLU(inplace)
    (32): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (33): ReLU(inplace)
    (34): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (35): ReLU(inplace)
    (36): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (classifier): Sequential(
    (0): Linear(in_features=25088, out_features=4096, bias=True)
    (1): ReLU(inplace)
    (2): Dropout(p=0.5)
    (3): Linear(in_features=4096, out_features=4096, bias=True)
    (4): ReLU(inplace)
    (5): Dropout(p=0.5)
    (6): Linear(in_features=4096, out_features=133, bias=True)
  )
)
```
### Loss function and optimizer

```Python
# loss function and optimizer for normal output
criterion_transfer = nn.CrossEntropyLoss()
# for VGG 19
optimizer_transfer = optim.SGD(filter(lambda p: p.requires_grad,model_transfer.parameters()), lr=param_learning_rate)
```
I use the `CrossEntropyLoss` function and the `SGD` optimizer to train the model.

### Train the model

To train the neural network (CNN), start the last part of file `dog_app.ipynb` marked as `train the model`.


### Parameters of training

To change to input folder, the output size and some other parameters for the neural network, you can adapt these global constants inside the python file.

```python
# ---- set parameters ---------------
param_data_directory = "dogImages"          # default: dogImages
param_learning_rate = 0.01                  # 0.001
param_output_size = 133                     # 133 - original
param_epochs = 5

param_transform_resize = 224
param_transform_crop = 224
# -----------------------------------
```

### Output of training

```bash
start training for 5 epochs ...
load previous saved model ...
Epoch: 1 	Training Loss: 1.387199 	Validation Loss: 0.574920  Saving model ...
Epoch: 2 	Training Loss: 1.320768 	Validation Loss: 0.547921  Saving model ...
Epoch: 3 	Training Loss: 1.285078 	Validation Loss: 0.525249  Saving model ...
Epoch: 4 	Training Loss: 1.281811 	Validation Loss: 0.501114  Saving model ...
Epoch: 5 	Training Loss: 1.202475 	Validation Loss: 0.496238  Saving model ...
done
```

I got the training results of:

```bash
Test Loss: 0.000668
Test Accuracy: 82% (691/836)
```
After 5 epochs, I got a test accuracy of `82%`. That's god and can be improved.

## Improvements

This is actually my best version of flower image classifier.
The next steps will be:
* do some experiments with the neural network to change the hyperparameters / hidden units / optimizer / structure of fully connected layers, ...
* convert the image classifier algorithm to keras
* convert the image classifier algorithm to fast.ai
* I will see what's comming next ...

## Authors

* Daniel Jaensch

## License
[MIT](https://choosealicense.com/licenses/mit/)

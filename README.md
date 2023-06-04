# AIN3002TermProject
the term project for the 2022/2023 second semester course AIN3002 (Deep Learning)

![Project Logo](https://cdn.discordapp.com/attachments/688277804680216605/1114577128869089330/bau.png) 

# Examining the Impact of Dropout on Overfitting in Deep Learning Architectures

this repository contains the codes and resources needed to implement the experiments described in the [project report](AIN3002ProjectReport.pdf).

we conducted a number of separate experiments to test the effectiveness of using dropout on machine learning models to prevent overfitting and improve the validation performance, some other experiments tested out how well dropout compares with other techniques that were previously used to conduct similar tasks.

the codes for the experiments are in the codes folder in this repository, the structure of the file is as follows:

- [codes]( /codes/)
   - CIFAR
         -[CIFAR-10](codes/CIFAR/CIFAR_10.ipynb)
         -[CIFAR-100](codes/CIFAR/CIFAR_100.ipynb)
   - DNN vs BNN
         -[DNNvsBNN]( codes/DNNvsBNN/DNNvsBNN.ipynb)
   - MINST
         -[MINSTdropouteffects]( /codes/MINST/MINSTdropouteffects.ipynb)
   - Plot DataPoints
         -[classification](/codes/PlotDataPoints/dropoutClassifcationOverfittingExample.ipynb)
         -[regression](/codes/PlotDataPoints/regression.ipynb)
   - Regualizers
         -[Regualizers Comparison](/codes/Regualizers/RegualizersComparison.ipynb)

We will go over the libraries needed to build our models, the datasets used, the model strecture and how to tune its parameters, and where you can easily train the models. then, we will go over the results that we got and what conclusions they lead us into.

## Table of Contents
- [Libraries and Packages](#LibrariesandPackages)
- [DataSets](#DataSets)
- [Models](#Models)
- [Training Platform](#TrainingPlatform)
- [Results Showcase](#ResultsShowcase)
- [Conclusion](#Conclusion)
- [Authors](#authors)
- [Links](#links)

## Libraries and Packages


- **TensorFlow**: TensorFlow is an open-source machine learning framework widely used for building and training deep learning models. It provides a comprehensive set of tools and libraries for numerical computation, dataflow programming, and building neural networks.
 To install TensorFlow using pip, you can run the following command in a code cell:
  ```python
  !pip install tensorflow
  ```
  
- **Keras**: Keras is a high-level neural networks API written in Python and built on top of TensorFlow. It provides a user-friendly and intuitive interface to define and train deep learning models.

To install Keras using pip, you can run the following command:
  ```python
  pip install keras
  ```
- **NumPy**: NumPy is a fundamental library for scientific computing in Python. It provides support for large, multi-dimensional arrays and matrices, along with a collection of mathematical functions to operate on these arrays.

To install NumPy using pip, you can run the following command:

  ```python
  pip install numpy
  ```
- **Matplotlib**: Matplotlib is a plotting library for Python that provides a flexible and comprehensive set of tools for creating static, animated, and interactive visualizations in Python.

To install Matplotlib using pip, you can run the following command:

  ```python
  pip install numpy
  ```
  
  ## DataSets

To install the necessary packages for accessing the datasets offered by tensorflow, you can run the following command in a code cell:
```python
!pip install tensorflow-datasets
```

### MNIST

The MNIST dataset is a widely-used benchmark dataset for image classification tasks. It consists of 60,000 training images and 10,000 test images of handwritten digits (0-9) with corresponding labels.
![image](https://github.com/Magmuma/AIN3002TermProject/assets/63364100/0f2450d6-80c2-41ee-be24-19fdb2840f81)


To import the MNIST dataset from TensorFlow, you can use the following code:
```python
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
```

### CIFAR-10/100

The CIFAR-10 and CIFAR-100 datasets are popular benchmark datasets for image classification tasks. CIFAR-10 consists of 60,000 32x32 color images in 10 classes, while CIFAR-100 consists of 60,000 32x32 color images in 100 classes.

![image](https://github.com/Magmuma/AIN3002TermProject/assets/63364100/3d43474b-7eee-4125-b1d6-2a059595b437)

To import the CIFAR-10 and CIFAR-100 datasets from TensorFlow, you can use the following code:

```python
from tensorflow.keras.datasets import cifar10, cifar100

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

(x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')

```

for preprocessing, we simply normalized the pixel values of the images by dividing them by 255.0. The original pixel values range from 0 to 255, where 0 represents black and 255 represents white. Dividing by 255.0 scales the pixel values to the range of 0.0 to 1.0, which is a common practice for neural networks that process image data. Normalization helps to ensure that the model can learn effectively by reducing the impact of different scales and improving convergence during training.

By performing this normalization, the pixel values of the images are transformed from the original integer range to a floating-point range between 0.0 and 1.0, making it easier for neural networks to process and learn patterns from the data.

```python

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()


train_images, test_images = train_images / 255.0, test_images / 255.0


```


### 2 Classes of Datapoints

this is a manually generated database that consists of 150 datapoints, the data points are separated into 2 different classes labeled 1 and 0, each class is a cluster of data points with some outlier points in the middle of the cluster of the opposite class, this is done to showcase the effects of overfitting clearly in the case of it happening.


![image](https://github.com/Magmuma/AIN3002TermProject/assets/63364100/c34a87ca-e203-4824-9dca-99eff2a55404)


## Models
Tensorflow keras allows us to build machine learning models in a simple and user-friendly way, the many libraries they offer cover much of what we need to build models of all sizes, it is very easy to add and remove layers, set input dimensions, set activation functions, change and fine-tune parameters, add regularizes and optimizers, and most importantly for our project, add dropout layers.

To build a Kares model, it’s very easy, and can be done in this format:

```python
model = Sequential()
model.add(Dense(64, activation='relu', kernel_constraint=MaxNorm(3), input_dim=100))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu', kernel_constraint=MaxNorm(3)))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

```
this model contains 2 hidden layers with 64 units, 2 dropout layers, uses re-Lu activation in its hidden layers and softmax in its output layer, as seen, it is very easy to fine-tune and change the parameters and components of this model, for example, the dropout rate is the only parameter that can be used for dropout layers, you can define the rate between 0 and 1, in this model, it is set as 0.5, as you try to fine-tune your model and find the optimal dropout rate for each layer, you will be able to change it very easily, this model also applies max-norm regularization which uses the parameter c, you can also tune this parameters to find the most optimal value for your model and data. 


After defining the model, it is very easy to Compile and train the model, keras offers built in functions for this, you can set and change the optimizer for your model, along with the learning rate and decay rate, or you can import optimizers like adam who have these values set already, you can also set the number of epochs and the batch size very easily, these are values you can change depending on your training environment, too many epochs may take a very long time to train, so make sure to decrease it if you don’t want a high training time, this will most likely effect your validation, so be careful and keep your results in mind, most of our training was done with 50-500 epochs:

```python
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=32)

```

To evaluate how well your models are performing, you can use the evaluate built-in function to see how your loss and validation are doing, to conduct experiments, you can use these values from all different models and compare them with eachother:

```python
loss, accuracy = model.evaluate(X_test, y_test)
```

## Training Platform

Make sure to run these codes on a powerful machine, training machine learning models usually take a lot of computational power, so it’s best to train on GPUs, thankfully, google collab provides free to use GPU training on their platform, and we conducted our experiments using their services, it is very easy to use google collab, all you need is your google account, and you can visit [their site](https://colab.research.google.com/) and create or upload a new notebook.
It is important to remember to turn on GPU training, otherwise, collab will default run your code on its CPU, this makes training a very long and slow process, to turn on GPU training, click on the ram and disk usage icon on the top right:

![image](https://github.com/Magmuma/AIN3002TermProject/assets/63364100/c133b6c0-0b11-42b0-ae79-eeb995034f12)

 after the resources window opens up on the left of the screen, look to the bottom left, and click on the “change runtime type” option, after doing so, a pop-up will appear, in the hardware accelerator section, make sure that GPU is chosen, Collab offers different type of GPUs to train on, however, most of them are only available for CollabPro users, a paid service by google Collab, you can still use the free-to-use T4 GPU that contains 15GB, for our tasks, this should be plenty enough:
 
 ![image](https://github.com/Magmuma/AIN3002TermProject/assets/63364100/fb9d7e87-ffb3-41dd-b949-7374e3b4a975)



## Results Showcase
<!-- Explain how users can contribute to your project -->

## Conclusion
<!-- Specify the license under which your project is distributed -->

## Authors
- Mohammed Ali Maghmoum - [Email](mailto:mohamedali.maghmoum@bahcesehir.edu.tr)
- Hashem Ali Alshami - [Email](mailto:ali.alshami1@bahcesehir.edu.tr)


## Links
- [GitHub Repository](https://github.com/Magmuma/AIN3002TermProject)

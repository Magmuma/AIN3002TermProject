# AIN3002TermProject
the term project for the 2022/2023 second semester course AIN3002 (Deep Learning)
![Project Logo](https://cdn.discordapp.com/attachments/688277804680216605/1114577128869089330/bau.png) 

# Examining the Impact of Dropout on Overfitting in Deep Learning Architectures

this repository contains the codes and resources needed to implement the experiments described in the [project report](AIN3002ProjectReport.pdf).

we conducted a number of separate experiments to test the effectiveness of using dropout on machine learning models to prevent overfitting and improve the validation performance, some other experiments tested out how well dropout compares with other techniques that were previously used to conduct similar tasks
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

We will go over the libraries needed to build our models, import datasets, preprocess and prepare data, and plot our results, as well as information about the training process, our method and platform of training, and the finetuning and hyperparameters used to achieve our results.

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

### 2 Classes of Datapoints

this is a manually generated database that consists of 150 datapoints, the data points are separated into 2 different classes labeled 1 and 0, each class is a cluster of data points with some outlier points in the middle of the cluster of the opposite class, this is done to showcase the effects of overfitting clearly in the case of it happening.
![image](https://github.com/Magmuma/AIN3002TermProject/assets/63364100/c34a87ca-e203-4824-9dca-99eff2a55404)


## Models
<!-- Include examples or screenshots -->

## Training Platform
<!-- Provide links to additional documentation or tutorials -->

## Results Showcase
<!-- Explain how users can contribute to your project -->

## Conclusion
<!-- Specify the license under which your project is distributed -->

## Authors
- Mohammed Ali Maghmoum - [Email](mailto:mohamedali.maghmoum@bahcesehir.edu.tr)
- Hashem Ali Alshami - [Email](mailto:ali.alshami1@bahcesehir.edu.tr)


## Links
- [GitHub Repository](https://github.com/Magmuma/AIN3002TermProject)

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

We will go over the libraries needed to build our models, the datasets used and how to preprocess them, the model strecture and how to tune its parameters, and where you can easily train the models. then, we will go over the results that we got and what conclusions they lead us into.

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

for preprocessing on MINST, we first converted the pixel values of the training and test images to floating-point numbers and then normalized them by dividing by 255. The original pixel values are integers ranging from 0 to 255, and dividing by 255 scales them to the range of 0.0 to 1.0.

after that, we reshape the training and test images from a 3D array (representing images with dimensions of 28x28 pixels) to a 2D array. Each image is flattened into a 1D array of size 784 (28*28) pixels. This transformation is necessary to feed the data into a neural network that expects a 2D input.

we then add a line that converts the training and test labels (y) into one-hot encoded vectors using the to_categorical() function from the tf.keras.utils module. The original labels are integers ranging from 0 to 9, representing the digits from 0 to 9. One-hot encoding transforms these integer labels into binary vectors, where each element of the vector represents a class. The index corresponding to the true class is set to 1, while all other indices are set to 0.


```python
# Normalize pixel values to range [0, 1]
X_train = X_train.astype('float32') / 255.
X_test = X_test.astype('float32') / 255.

# Reshape input data into 2D array (28x28 pixels)
X_train = X_train.reshape(len(X_train), 28*28)
X_test = X_test.reshape(len(X_test), 28*28)

# Convert target variable to one-hot encoding
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

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

For the CIFAR-100 dataset, we added an extra line of preprocessing, This line converts the training and test set labels into binary class matrices using the to_categorical() function. The original labels are represented as integers ranging from 0 to 99, representing the 100 classes in CIFAR-100. The to_categorical() function converts these integer labels into binary matrices, where each row corresponds to a sample, and each column represents a class. The column corresponding to the true class is set to 1, while all other columns are set to 0. This conversion is necessary for training certain types of neural networks, such as those using categorical cross-entropy loss.


```python

train_labels = to_categorical(train_labels, 100)
test_labels = to_categorical(test_labels, 100)

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

### [Data Points Classification](/codes/PlotDataPoints/dropoutClassifcationOverfittingExample.ipynb)

![image](https://github.com/Magmuma/AIN3002TermProject/assets/63364100/a432f14e-0262-4a05-937c-69e0de85b36d)

The model used for classification here contains 2 hidden layers with 128 units, the same model is used again with 2 dropout units added, the dropout here was chosen as an optimal 0.5, which means that 50% of the units in the previous layer would be dropped.

Optimizer adam was used with a defined learning rate of 0.01

The models were trained for 500 epochs, this is not a very large dataset so the high number of epochs didn’t badly affect the training time.

The results clearly show overfitting of the data in the first model without dropout, this is represented in both the training and validation accuracies plot as well as the classified data point plot, the model had a very high training validation, reaching almost 100% training accuracy near the ending of the training process,  but this is in contrast to the validation accuracy, which dropped much further below at around 80%, with it dropping as low as 65% during some points in the training.

The second model, with all parameters of the model still the same, showed that dropout did, in fact, prevent overfitting, the datapoint plot showcases that the data didn’t overfit the model, and the training and validation accuracies stayed close to each other throughout the training processes.


### [DataPoint Regression](/codes/PlotDataPoints/regression.ipynb)

![image](https://github.com/Magmuma/AIN3002TermProject/assets/63364100/d381795d-6e65-4a49-87e0-abe3066ac7b7)

Similar to the classification task, 2 identical models with the same parameters were used, with dropout deployed on one of them, here we can see how the first models overfits the data, while the model with dropout prevents it, dropout rate was set to 0.2 this time around, the models used had 2 hidden layers with 128 units each.

### [MINST dropout effects]( /codes/MINST/MINSTdropouteffects.ipynb)

![image](https://github.com/Magmuma/AIN3002TermProject/assets/63364100/db57257d-fe17-45f5-b39b-f7cb105c6127)

in general, incorporating dropout techniques enhances the performance of models by mitigating the impact of overfitting. However, it is worth noting that this improvement was not consistently observed across all models. Specifically, models 3, 6, and 7 experienced a decline in performance when dropout was applied. Model 3, which shares the same architecture as model 2 but utilizes the relu activation function, consistently exhibited underperformance when combined with dropout. On the other hand, models 6 and 7, both wide models with an increased number of units, demonstrated a substantial decrease in accuracy when the dropout rates remained unchanged. These results suggest that the interaction between dropout and specific model architectures or activation functions can lead to unexpected outcomes, underscoring the importance of careful analysis and parameter selection when implementing dropout.

### [CIFAR-10](codes/CIFAR/CIFAR_10.ipynb) and [CIFAR-100](codes/CIFAR/CIFAR_100.ipynb)

![image](https://github.com/Magmuma/AIN3002TermProject/assets/63364100/20a076b5-51e6-4681-a785-b68f895b6170)

The same model of was tried on the 2 datasets with and without adding fully connected layers with dropout, the dropout models showed a noticeable increase in the performance as the classification error is reduced.

The model architecture used for this experiment combines convolutional layers for extracting spatial features from the input images with fully connected layers for classification. For the dropout model, a dropout rate of 0.5 was used in the fully connected layers.


### [DNNvsBNN]( codes/DNNvsBNN/DNNvsBNN.ipynb)




Dropout and Bayesian neural networks are both techniques used for model averaging, but they have distinct approaches. Dropout involves averaging multiple models with shared weights, giving equal weight to each model. On the other hand, Bayesian neural networks assign weights to individual models based on the prior and how well they fit the data. Bayesian neural networks are particularly effective in domains with limited data, such as medical diagnosis and genetics. However, they are slower to train and may struggle to scale to large networks. In contrast, dropout networks are faster to train and more straightforward to use during testing.

To evaluate and compare the performance of Bayesian neural networks and dropout networks, an experiment was conducted using the MNIST dataset. The objective of the experiment was to assess the extent to which dropout networks fall short in comparison to Bayesian networks in terms of performance.


![image](https://github.com/Magmuma/AIN3002TermProject/assets/63364100/34454584-ef71-4ecf-a1d5-5a2f213ff25d)
         
As expected, dropout fell a bit short in comparison, however, considering the simplicity of dropout, it can be argued that it was the more efficient method for averaging out these models.

### [Regualizers Comparison](/codes/Regualizers/RegualizersComparison.ipynb)

In the context of regularization techniques, dropout is a method that is being evaluated in comparison to other standard regularization methods such as max-norm and L2 regularization. The experiment focuses on observing and analyzing the performance of dropout when applied to a model tasked with classification using the MNIST dataset. The specific architecture of the model consists of three layers with sizes 1024, 1024, and 2048 respectively. In this experiment, a dropout rate of 0.2 is applied to all layers of the model. The goal is to assess how dropout, as a regularization technique, compares to other established regularization methods in terms of improving model performance and generalization on the MNIST dataset.

![image](https://github.com/Magmuma/AIN3002TermProject/assets/63364100/62f90708-682c-42b0-b22c-60e703be6b82)

Based on these results, we can see that max-norm and max-norm with dropout achieved the best accuracy and the lowest error, with L2, L2 + KL sparsity, and L2+dropout performing the worst, this is again another example that showcases dropout does not always improve results across all models. Dropout was able to slightly Improve the performance of max-norm, but significantly lowered the performance L2.

Dropout rate and other parameters were unchanged throughout the models, dropout rate was 0.2, c for max-norm was 3, and regularization rate for L2 was 0.001.


## Conclusion


Dropout has emerged as an innovative regularization technique in the field of machine learning, offering a simple yet effective solution to tackle the problem of overfitting in deep neural networks. By randomly deactivating certain units during training, dropout prevents complex co-adaptations and reduces the model's reliance on specific features, thereby promoting better generalization. Extensive experiments using different architectures and datasets like MNIST and CIFAR have consistently demonstrated the efficacy of dropout, although its impact may vary depending on the specific models and configurations employed.

One of the notable advantages of dropout is its superiority over traditional regularizers such as max-norm and L2 in various scenarios. This further strengthens its utility in improving the generalization capability of models. However, it is worth mentioning that dropout does not always guarantee improved performance and there have been instances where its application resulted in a decrease in model accuracy. This raises the question of whether dropout is the optimal method for reducing overfitting across all machine learning models.

In comparison to Bayesian Neural Networks, dropout can be viewed as a technique that combines multiple pre-trained models. However, it generally underperforms when compared to Bayesian approaches. Nevertheless, dropout still provides an efficient and computationally cost-effective way to average the predictions of many trained models, which can be beneficial in certain applications.

As the field of machine learning continues to evolve, it is crucial to understand and consider both the strengths and limitations of dropout in specific applications and datasets. While dropout has proven its effectiveness in preventing overfitting and improving model generalization, researchers and practitioners should carefully evaluate its suitability for their particular use cases. By considering the constantly evolving landscape of machine learning techniques, we can make informed decisions about when and where to leverage dropout for optimal results.

## Authors
- Mohammed Ali Maghmoum - [Email](mailto:mohamedali.maghmoum@bahcesehir.edu.tr)
- Hashem Ali Alshami - [Email](mailto:ali.alshami1@bahcesehir.edu.tr)


## Links
- [GitHub Repository](https://github.com/Magmuma/AIN3002TermProject)
- [reference paper](https://paperswithcode.com/paper/dropout-a-simple-way-to-prevent-neural)
- [Keras website](https://keras.io/)
- [Google Collab](https://colab.research.google.com/)

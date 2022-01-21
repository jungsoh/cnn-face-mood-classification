# Convolutional neural network: Face mood classification
We have the task of recognizing the mood of people as happy (smiling) or not happy from their face images. This is a binary classification problem and we build a convolutional neural network classifier using TensorFlow Keras Sequential API. I did this project in the [Convolutional Neural Networks](https://www.coursera.org/learn/convolutional-neural-networks) course as part of the [Deep Learning Specialization](https://www.coursera.org/specializations/deep-learning).

## Datasets
We have 600 training examples and 150 test examples, where each example is of shape (64, 64, 3) with each of RGB channel image is of size 64x64. The examples are labeled as either 0 for 'not happy' or 1 for 'happy'.

## Convolutional neural network
We used TensorFlow Keras Sequential API to build a convolutional neural network model. The resulting model's `.summary()` method shows the following layers and parameters.
```
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
zero_padding2d (ZeroPadding2 (None, 70, 70, 3)         0         
_________________________________________________________________
conv2d (Conv2D)              (None, 64, 64, 32)        4736      
_________________________________________________________________
batch_normalization (BatchNo (None, 64, 64, 32)        128       
_________________________________________________________________
re_lu (ReLU)                 (None, 64, 64, 32)        0         
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 32, 32, 32)        0         
_________________________________________________________________
flatten (Flatten)            (None, 32768)             0         
_________________________________________________________________
dense (Dense)                (None, 1)                 32769     
=================================================================
Total params: 37,633
Trainable params: 37,569
Non-trainable params: 64
_________________________________________________________________
```
We trained the with the Keras model's `.fit()` and evaluated its performance on the test set with the `.evaluate()` method. The training accuracy is 0.92 and the test accuracy is 0.72.

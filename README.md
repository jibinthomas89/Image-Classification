# Image-Classification
Image Classification using Convoluted Neural Networks

## Summary
A convoluted neural network (CNN) algorithm was used to classify dog breeds from digital photographs. Previous approaches used support vector machines (SVMs) and scale invariant feature transforms (SIFTs) or CNNs to determine differences between breeds with image preprocessing that focused on facial features. Compared the accuracy of two CNN workflows on a dataset, consisting of 10,000 images with 120 dog breeds. In the second approach, data was mirrored to increase the training set, resulting in 21,000 images and further preprocessed to focus on facial features using CENSURE. Accuracy in breed recognition improved from 3% in the first workflow to 4.7% in the second.

## Implementation details

Step1: Image preprocessing - Converted the RGB images to 150x150 greyscale images (due to limited computing power) and then mirrored the images to increase the training dataset. Applied a Center Surround Extremas (CENSURE) algorithm to extract key facial features from each image.

Step2: Used the Conv2d Pytorch neural network package to build a model consisting of 2 CNN layers in addition to 2 ReLu layers and a maxpool layer. The model was trained on a sample batch size of 100 with a learning rate of 0.01 for 10 epochs. Used CrossEntropyLoss criterion to evaluate the performance of the model

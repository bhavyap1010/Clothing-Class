# README

## Table of Contents

- [Project Overview](#project-overview)
- [Files](#files)
- [Purpose](#purpose)
- [Inspiration](#inspiration)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Running the Examples](#running-the-examples)
- [Comparison of Key Elements](#comparison-of-key-elements)
- [Conclusion](#conclusion)

## Project Overview

This project aims to compare two popular deep learning frameworks, **PyTorch** and **TensorFlow**, using simple examples to illustrate their usage. The main focus is on building a basic image classifier for the Fashion MNIST dataset, which contains grayscale images of clothing items. By implementing similar models in both frameworks, we can highlight the differences in their APIs, usability, and overall approaches to model training.

## Files

1. **pytorch_clothing_classifier.py**: This file contains the implementation of a clothing classifier using PyTorch.
2. **tensorflow_clothing_classifier.py**: This file contains the implementation of a clothing classifier using TensorFlow.

## Purpose

The objective of this project is to provide insights into the following aspects of each framework:

- **Ease of Use**: How straightforward it is to set up a neural network and run training loops.
- **Syntax and Structure**: The differences in code structure and readability between PyTorch and TensorFlow.
- **Performance**: While the focus is on simplicity, we will also consider how each framework performs during training.

## Inspiration

This project was inspired by Fireship's videos on YouTube:
- [PyTorch in 100 Seconds](https://youtu.be/i8NETqtGHms)
- [TensorFlow in 100 Seconds](https://youtu.be/ORMx45xqWkA)

## Getting Started

To get started with this project, follow the instructions below:

### Prerequisites

- Python 3.12
- PyTorch (for the PyTorch implementation)
- TensorFlow (for the TensorFlow implementation)
- torchvision (for image transformations and datasets)

You can install the required libraries using the following commands:

```
pip install torch torchvision
pip install tensorflow
```

### Running the Examples

1. **PyTorch Implementation**: 
   - Open the `pytorch_clothing_classifier.py` file.
   - Run the script to train the clothing classifier using PyTorch.

2. **TensorFlow Implementation**: 
   - Open the `tensorflow_clothing_classifier.py` file.
   - Run the script to train the clothing classifier using TensorFlow.

### Comparison of Key Elements

- **Model Definition**: Both frameworks utilize classes to define the architecture of the neural network, although the syntax and structure may differ.
- **Device Management**: The device setup in PyTorch allows for easy integration with MPS (Metal Performance Shaders) on Mac, while TensorFlow offers its own methods for GPU utilization.
- **Training Loop**: The training loop is structured similarly, but with different API calls and conventions in each framework.

## Conclusion

This project serves as a hands-on comparison of PyTorch and TensorFlow, highlighting the strengths and weaknesses of each framework. By working through both implementations, users can gain valuable experience in deep learning and understand how to choose the right framework for their projects.

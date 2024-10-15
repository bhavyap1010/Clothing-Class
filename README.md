# Clothing Type Prediction with TensorFlow

This project demonstrates how to build a machine learning model using TensorFlow to predict clothing types from images. The model is trained on the Fashion MNIST dataset, which consists of grayscale images of clothing items. The goal is to create a neural network that can classify images into different clothing categories.

## Table of Contents

- [Technologies Used](#technologies-used)
- [Getting Started](#getting-started)
- [Installation](#installation)
- [Usage](#usage)
- [Training the Model](#training-the-model)
- [Model Evaluation](#model-evaluation)
- [Contributing](#contributing)
- [License](#license)

## Technologies Used

- Python 3.12
- TensorFlow
- NumPy
- Matplotlib (for visualization)
- VSCode

## Getting Started

To get started with this project, you’ll need to set up your development environment.

## Installation

1. **Clone the repository:**
```bash
git clone https://github.com/bhavyap1010/Clothing-Class.git
cd Clothing-Class
```
2.	**Create a virtual environment:**

```bash
python -m venv venv
```

3.	**Activate the virtual environment:**
	
- **Windows:**
```bash
.\venv\Scripts\activate
```

- **Mac/Linux:**
```bash
source venv/bin/activate
```

4.	**Install TensorFlow and other dependencies:**

  ```
  pip install tensorflow matplotlib numpy
  ```


## Usage

  1.	Open the project in VSCode.
	2.	Run the training script:

  ```
  python train.py
  ```


3.	The model will be trained on the Fashion MNIST dataset. You can modify the training parameters in the train.py file.

## Training the Model

The model is defined in the train.py file. It consists of:

- A flatten layer that converts each 28x28 pixel image into a one-dimensional array.
- A dense layer with 128 neurons using the ReLU activation function.
- An output layer with 10 neurons corresponding to the clothing categories.

The model is compiled using the sparse categorical cross-entropy loss function and the Adam optimizer.

## Model Evaluation

After training, the model’s accuracy can be evaluated using the test dataset. The evaluation results will be printed to the console.

## Contributing

Contributions are welcome! If you’d like to contribute to this project, please fork the repository and submit a pull request.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

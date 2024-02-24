# MNIST Classification with Convolutional Neural Networks

This repository contains the code for classifying MNIST digits using a Convolutional Neural Network (CNN) implemented in PyTorch. The project is structured into separate files for clarity and modularity.

## Project Structure

- `model.py`: Contains the definition of the `Net` class, which is our CNN model for MNIST digit classification. This file includes the network architecture, layer definitions, and the forward pass logic.

- `utils.py`: Includes utility functions and classes used in data loading, transformation, training, and testing of the model. This includes the functions `load_data`, `train`, and `test` which handle the MNIST dataset preparation and the training/testing loops respectively.

- `S5.ipynb`: This Jupyter notebook is the main file where the model and utility functions are utilized. It orchestrates the data loading, model initialization, training, testing, and visualization of results. Run this notebook to execute the whole MNIST classification workflow.


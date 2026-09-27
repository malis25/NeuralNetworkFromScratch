# Neural Network From Scratch

A neural network implementation written from scratch in **C++**, without relying on external machine learning frameworks.

The project focuses on implementing the fundamental building blocks of a neural network manually, from tensor operations to forward propagation, backpropagation, and training.

## Overview

**NeuralNetworkFromScratch** is a low-level neural network project built to explore how neural networks work internally.

Instead of using an existing machine learning framework, the project implements its own numerical and neural-network components. This provides direct control over the data structures, mathematical operations, and training process.

The current implementation uses a **Multilayer Perceptron (MLP)** and demonstrates it by training on the XOR problem.

## Features

* Custom `Tensor` implementation
* Matrix multiplication
* Tensor operations
* Element-wise operations
* Scalar operations
* Transpose operations
* Activation functions
* Multilayer Perceptron
* Forward propagation
* Backpropagation
* Parameter updates
* Training loop
* Benchmarks and tests

## Architecture

The project is built around a few core components:

```text
NeuralNetworkFromScratch/
│
├── math/
│   ├── Tensor.h
│   ├── Tensor.cpp
│   ├── Activations.h
│   └── Activations.cpp
│
├── test/
│   └── ...
│
├── MLP.h
├── MLP.cpp
├── main.cpp
├── graph.py
└── makefile
```

### Tensor

`Tensor` is the numerical foundation of the project.

It provides the data structure and operations used by the neural network, including:

* Shape management
* Element access
* Element-wise arithmetic
* Scalar operations
* Matrix multiplication
* Transposition
* Other low-level tensor operations

The neural network uses this component for its numerical computations.

### Activations

The activation module contains the activation functions used by the neural network along with their derivatives.

These functions are used during both forward propagation and backpropagation.

### MLP

`MLP` implements the neural network itself on top of the tensor system.

It handles:

* Network initialization
* Forward propagation
* Backpropagation
* Gradient calculation
* Parameter updates
* Training

## Example

The current example uses an MLP with the following architecture:

```text
Input
  │
  ▼
[ 2 neurons ]
  │
  ▼
[ 4 neurons ]
  │
  ▼
[ 1 neuron ]
  │
  ▼
Output
```

The network is trained to learn the XOR function:

```text
0 XOR 0 → 0
0 XOR 1 → 1
1 XOR 0 → 1
1 XOR 1 → 0
```

## Building

### Requirements

* C++17 compatible compiler
* GNU Make

### Build

```bash
make
```

### Run

On Linux:

```bash
./main
```

On Windows with a compatible MinGW environment:

```bash
main.exe
```

## Project Structure

| Component     | Description                                |
| ------------- | ------------------------------------------ |
| `Tensor`      | Core tensor and numerical operations       |
| `Activations` | Activation functions and their derivatives |
| `MLP`         | Multilayer Perceptron implementation       |
| `main.cpp`    | Example program and training entry point   |
| `test/`       | Tests and benchmarks                       |
| `graph.py`    | Graphing and visualization utility         |
| `makefile`    | Build configuration                        |

## Philosophy

The goal of this project is to understand neural networks by implementing their fundamental components directly.

Modern machine learning frameworks provide powerful abstractions, but they can hide many of the operations happening underneath.

This project takes a lower-level approach:

> Understand the fundamentals by building them yourself.

The neural network is built on top of the custom tensor implementation, keeping the relationship between the numerical operations and the higher-level model explicit.

## License

See the repository for license information.

#pragma once

#include <math/Tensor.h>

double Sigmoid(double x);
double SigmoidDerivative(double x);

Tensor Sigmoid(const Tensor& tensor);
Tensor SigmoidDerivative(const Tensor& tensor);
Tensor SigmoidDerivativeFromActivation(const Tensor& activation);

double LeakyReLU(double x);
double LeakyReLUDerivative(double x);

Tensor LeakyReLU(const Tensor& tensor);
Tensor LeakyReLUDerivative(const Tensor& tensor);
Tensor LeakyReLUDerivativeFromActivation(const Tensor& activation);

double ReLU(double x);
double ReLUDerivative(double x);

Tensor ReLU(const Tensor& tensor);
Tensor ReLUDerivative(const Tensor& tensor);
Tensor ReLUDerivativeFromActivation(const Tensor& activation);
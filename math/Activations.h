#pragma once

#include <math/Matrix.h>

double Sigmoid(double x);
double SigmoidDerivative(double x);

Matrix Sigmoid(const Matrix& matrix);
Matrix SigmoidDerivative(const Matrix& matrix);
Matrix SigmoidDerivativeFromActivation(const Matrix& activation);

double ReLU(double x);
double ReLUDerivative(double x);

Matrix ReLU(const Matrix& matrix);
Matrix ReLUDerivative(const Matrix& matrix);
Matrix ReLUDerivativeFromActivation(const Matrix& activation);
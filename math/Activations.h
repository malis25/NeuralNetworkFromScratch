#pragma once

#include <math/Matrix.h>

double Sigmoid(double x);
double SigmoidDerivative(double x);

Matrix Sigmoid(const Matrix& matrix);
Matrix SigmoidDerivative(const Matrix& matrix);
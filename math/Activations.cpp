#include <cmath>
#include <algorithm>
#include <math/Activations.h>

double Sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

double SigmoidDerivative(double x) {
    double sigmoidValue = Sigmoid(x);
    
    return sigmoidValue * (1.0 - sigmoidValue);
}

Matrix Sigmoid(const Matrix& matrix) {
    Matrix result = matrix;

    for (size_t i = 0; i < result.Rows() * result.Cols(); i++) {
        result[i] = Sigmoid(result[i]);
    }

    return result;
}

Matrix SigmoidDerivative(const Matrix& matrix) {
    Matrix result = matrix;

    for (size_t i = 0; i < result.Rows() * result.Cols(); i++) {
        result[i] = SigmoidDerivative(result[i]);
    }

    return result;
}

Matrix SigmoidDerivativeFromActivation(const Matrix& activation) {
    Matrix result = activation;

    for (size_t i = 0; i < result.Rows() * result.Cols(); i++) {
        result[i] *= 1.0 - result[i];
    }

    return result;
}

double ReLU(double x)
{
    return std::max(0.0, x);
}

double ReLUDerivative(double x)
{
    return x > 0 ? 1 : 0;
}

Matrix ReLU(const Matrix& matrix)
{
    Matrix result = matrix;

    for (size_t i = 0; i < result.Rows() * result.Cols(); i++) {
        result[i] = ReLU(result[i]);
    }

    return result;
}

Matrix ReLUDerivative(const Matrix& matrix)
{
    Matrix result = matrix;

    for (size_t i = 0; i < result.Rows() * result.Cols(); i++) {
        result[i] = ReLUDerivative(result[i]);
    }

    return result;
}

Matrix ReLUDerivativeFromActivation(const Matrix& matrix)
{
    return ReLUDerivative(matrix);
}
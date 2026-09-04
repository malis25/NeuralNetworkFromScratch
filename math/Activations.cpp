#include <cmath>
#include <math/Activations.h>

double Sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

double SigmoidDerivative(double x) {
    double sigmoidValue = Sigmoid(x);
    
    return sigmoidValue * (1.0 - sigmoidValue);
}

Matrix Sigmoid(const Matrix& matrix) {
    Matrix result(matrix.Transpose().Transpose());

    for (size_t i = 0; i < result.Rows() * result.Cols(); i++) {
        result[i] = Sigmoid(result[i]);
    }

    return result;
}

Matrix SigmoidDerivative(const Matrix& matrix) {
    Matrix result(matrix.Transpose().Transpose());

    for (size_t i = 0; i < result.Rows() * result.Cols(); i++) {
        result[i] = SigmoidDerivative(result[i]);
    }

    return result;
}
#include <cmath>
#include <math/Activations.h>

namespace {

constexpr double leakyReLUSlope = 0.01;

}

double Sigmoid(double x)
{
    return 1.0 / (1.0 + std::exp(-x));
}

double SigmoidDerivative(double x)
{
    double sigmoidValue = Sigmoid(x);
    
    return sigmoidValue * (1.0 - sigmoidValue);
}

Tensor Sigmoid(const Tensor& tensor)
{
    Tensor result = tensor;

    for (size_t i = 0; i < result.Size(); i++)
        result[i] = Sigmoid(result[i]);
    
    return result;
}

Tensor SigmoidDerivative(const Tensor& tensor)
{
    Tensor result = tensor;

    for (size_t i = 0; i < result.Size(); i++)
        result[i] = SigmoidDerivative(result[i]);
    
    return result;
}

Tensor SigmoidDerivativeFromActivation(const Tensor& activation)
{
    Tensor result = activation;

    for (size_t i = 0; i < result.Size(); i++)
        result[i] *= 1.0 - result[i];

    return result;
}

double LeakyReLU(double x)
{
    return x > 0.0 ? x : leakyReLUSlope * x;
}

double LeakyReLUDerivative(double x)
{
    return x > 0.0 ? 1.0 : leakyReLUSlope;
}

Tensor LeakyReLU(const Tensor& tensor)
{
    Tensor result = tensor;

    for (size_t i = 0; i < result.Size(); i++)
        result[i] = LeakyReLU(result[i]);

    return result;
}

Tensor LeakyReLUDerivative(const Tensor& tensor)
{
    Tensor result = tensor;

    for (size_t i = 0; i < result.Size(); i++)
        result[i] = LeakyReLUDerivative(result[i]);

    return result;
}

Tensor LeakyReLUDerivativeFromActivation(const Tensor& activation)
{
    return LeakyReLUDerivative(activation);
}

double ReLU(double x)
{
    return LeakyReLU(x);
}

double ReLUDerivative(double x)
{
    return LeakyReLUDerivative(x);
}

Tensor ReLU(const Tensor& tensor)
{
    return LeakyReLU(tensor);
}

Tensor ReLUDerivative(const Tensor& tensor)
{
    return LeakyReLUDerivative(tensor);
}

Tensor ReLUDerivativeFromActivation(const Tensor& activation)
{
    return LeakyReLUDerivativeFromActivation(activation);
}
#include <cmath>
#include <algorithm>
#include <math/Activations.h>

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

double ReLU(double x)
{
    return std::max(0.0, x);
}

double ReLUDerivative(double x)
{
    return x > 0 ? 1 : 0;
}

Tensor ReLU(const Tensor& tensor)
{
    Tensor result = tensor;

    for (size_t i = 0; i < result.Size(); i++)
        result[i] = ReLU(result[i]);

    return result;
}

Tensor ReLUDerivative(const Tensor& tensor)
{
    Tensor result = tensor;

    for (size_t i = 0; i < result.Size(); i++)
        result[i] = ReLUDerivative(result[i]);

    return result;
}

Tensor ReLUDerivativeFromActivation(const Tensor& tensor)
{
    return ReLUDerivative(tensor);
}
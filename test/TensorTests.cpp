#include <cmath>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <math/Tensor.h>

namespace {

void Check(bool condition, const std::string& message)
{
    if (!condition)
        throw std::runtime_error(message);
}

void CheckEqual(double actual, double expected, const std::string& message)
{
    Check(std::abs(actual - expected) < 1e-9, message);
}

void CheckTensorEqual(const Tensor& actual, const Tensor& expected, const std::string& message)
{
    Check(actual.Shape() == expected.Shape(), message + ": shape mismatch");
    for (size_t i = 0; i < actual.Size(); ++i)
        Check(std::abs(actual[i] - expected[i]) < 1e-9, message + ": value mismatch");
}

void CheckThrows(const std::function<void()>& action, const std::string& message)
{
    bool threw = false;
    try {
        action();
    } catch (const std::exception&) {
        threw = true;
    }
    Check(threw, message);
}

void TestConstructionAndIndexing()
{
    Tensor tensor({2, 3}, 1.5);
    Check(tensor.Shape() == std::vector<size_t>({2, 3}), "shape should be retained");
    Check(tensor.Strides() == std::vector<size_t>({3, 1}), "row-major strides should be calculated");
    Check(tensor.Size() == 6 && tensor.Dim() == 2, "size and dimensionality should be correct");
    CheckEqual(tensor({1, 2}), 1.5, "constructor should initialize each value");

    tensor({1, 2}) = 7.0;
    CheckEqual(tensor[5], 7.0, "multidimensional indexing should address row-major data");
    CheckThrows([&] { (void)tensor({1}); }, "incorrect index dimensionality should throw");
    CheckThrows([&] { (void)tensor({2, 0}); }, "out-of-range index should throw");
    CheckThrows([&] { tensor = {1, 2}; }, "initializer size mismatch should throw");
}

void TestArithmetic()
{
    Tensor left({2}, 0.0);
    left = {6.0, 8.0};
    Tensor right({2}, 0.0);
    right = {2.0, 4.0};
    Tensor expectedSum({2}, 0.0);
    expectedSum = {8.0, 12.0};
    Tensor expectedDifference({2}, 0.0);
    expectedDifference = {4.0, 4.0};
    Tensor expectedProduct({2}, 0.0);
    expectedProduct = {12.0, 32.0};
    Tensor expectedQuotient({2}, 0.0);
    expectedQuotient = {3.0, 2.0};
    Tensor expectedScaled({2}, 0.0);
    expectedScaled = {12.0, 16.0};
    Tensor expectedScalarQuotient({2}, 0.0);
    expectedScalarQuotient = {3.0, 4.0};

    CheckTensorEqual(left + right, expectedSum, "addition should be elementwise");
    CheckTensorEqual(left - right, expectedDifference, "subtraction should be elementwise");
    CheckTensorEqual(left * right, expectedProduct, "tensor multiplication should be elementwise");
    CheckTensorEqual(left / right, expectedQuotient, "tensor division should be elementwise");
    CheckTensorEqual(left * 2.0, expectedScaled, "right scalar multiplication should work");
    CheckTensorEqual(2.0 * left, expectedScaled, "left scalar multiplication should work");
    CheckTensorEqual(left / 2.0, expectedScalarQuotient, "scalar division should work");
    CheckTensorEqual((left + right) - right, left, "(A + B) - B should equal A");
    CheckTensorEqual((left * right) / right, left, "(A * B) / B should equal A when B has no zeros");

    Tensor expectedCompound({2}, 0.0);
    expectedCompound = {6.0, 8.0};
    left += right;
    left -= right;
    left *= right;
    left /= right;
    left *= 2.0;
    left /= 2.0;
    CheckTensorEqual(left, expectedCompound, "compound arithmetic should update values");

    Tensor differentShape({1}, 1.0);
    CheckThrows([&] { (void)(left + differentShape); }, "shape mismatch should throw");
}

void TestReshapeAndTranspose()
{
    Tensor tensor({2, 3}, 0.0);
    tensor = {1, 2, 3, 4, 5, 6};
    Tensor reshaped = tensor.Reshaped({3, 2});
    Check(reshaped.Shape() == std::vector<size_t>({3, 2}), "reshaped tensor should have requested shape");
    Tensor expectedReshaped({3, 2}, 0.0);
    expectedReshaped = {1, 2, 3, 4, 5, 6};
    CheckTensorEqual(reshaped, expectedReshaped, "reshape should preserve data order");
    Check(tensor.Shape() == std::vector<size_t>({2, 3}), "Reshaped should not change the original tensor");
    CheckThrows([&] { tensor.Reshape({4, 2}); }, "reshape with a different element count should throw");

    Tensor transposed = tensor.Transpose();
    Check(transposed.Shape() == std::vector<size_t>({3, 2}), "transpose should swap matrix dimensions");
    Tensor expectedTranspose({3, 2}, 0.0);
    expectedTranspose = {1, 4, 2, 5, 3, 6};
    CheckTensorEqual(transposed, expectedTranspose, "transpose should reorder values");
    CheckTensorEqual(tensor.Transpose().Transpose(), tensor, "double transpose should recover the original tensor");
    CheckTensorEqual(tensor.Transpose(0, 0), tensor, "transposing one dimension with itself should preserve data");
    CheckThrows([&] { (void)tensor.Transpose(0, 2); }, "invalid transpose dimension should throw");
}

void TestReductionsAndMatmul()
{
    Tensor tensor({2, 2}, 0.0);
    tensor = {1, 2, 3, 4};
    CheckEqual(tensor.Sum(), 10.0, "sum should include every element");
    CheckEqual(tensor.Mean(), 2.5, "mean should average every element");
    tensor.Fill(-2.0);
    Tensor expectedFill({2, 2}, -2.0);
    CheckTensorEqual(tensor, expectedFill, "Fill should replace every element");

    Tensor a({2, 3}, 0.0);
    a = {1, 2, 3, 4, 5, 6};
    Tensor b({3, 2}, 0.0);
    b = {7, 8, 9, 10, 11, 12};
    Tensor product = Matmul(a, b);
    Check(product.Shape() == std::vector<size_t>({2, 2}), "Matmul should return the output matrix shape");
    Tensor expectedProduct({2, 2}, 0.0);
    expectedProduct = {58, 64, 139, 154};
    CheckTensorEqual(product, expectedProduct, "Matmul should calculate matrix products");

    Tensor identity({2, 2}, 0.0);
    identity = {1, 0, 0, 1};
    CheckTensorEqual(Matmul(identity, a), a, "multiplying by the identity matrix should preserve a tensor");
    CheckThrows([&] { (void)Matmul(a, a); }, "incompatible Matmul dimensions should throw");
}

void TestRandomize()
{
    Tensor tensor = Tensor::Random({32}, -0.5, 0.5);
    Check(tensor.Size() == 32, "random tensor should have the requested number of elements");
    for (double value : tensor.Data())
        Check(value >= -0.5 && value <= 0.5, "random values should stay within the requested range");
}

}

int main()
{
    const std::vector<std::pair<std::string, std::function<void()>>> tests = {
        {"construction and indexing", TestConstructionAndIndexing},
        {"arithmetic", TestArithmetic},
        {"reshape and transpose", TestReshapeAndTranspose},
        {"reductions and Matmul", TestReductionsAndMatmul},
        {"randomize", TestRandomize},
    };

    for (const auto& test : tests) {
        try {
            test.second();
            std::cout << "[PASS] " << test.first << '\n';
        } catch (const std::exception& error) {
            std::cerr << "[FAIL] " << test.first << ": " << error.what() << '\n';
            return 1;
        }
    }

    std::cout << tests.size() << " tests passed\n";
    return 0;
}
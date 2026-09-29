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

template <typename Exception = std::exception>
void CheckThrows(const std::function<void()>& action, const std::string& message)
{
    bool caughtExpectedException = false;
    try {
        action();
    } catch (const Exception&) {
        caughtExpectedException = true;
    } catch (...) {
    }
    Check(caughtExpectedException, message);
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
    const Tensor& constTensor = tensor;
    CheckEqual(constTensor(std::vector<size_t>{1, 2}), 7.0, "vector indexing should support const tensor access");
    CheckThrows<std::invalid_argument>([&] { (void)tensor({1}); }, "incorrect index dimensionality should throw invalid_argument");
    CheckThrows<std::out_of_range>([&] { (void)tensor({2, 0}); }, "out-of-range index should throw out_of_range");
    CheckThrows<std::invalid_argument>([&] { (void)tensor(std::vector<size_t>{0}); },
                                       "vector index dimensionality mismatch should throw invalid_argument");
    CheckThrows<std::out_of_range>([&] { (void)tensor(std::vector<size_t>{0, 3}); },
                                   "vector index out of range should throw out_of_range");
    CheckThrows<std::invalid_argument>([&] { tensor = {1, 2}; }, "initializer size mismatch should throw invalid_argument");
    CheckEqual(tensor({1, 2}), 7.0, "invalid indexing should not affect valid tensor elements");

    Tensor zeroExtent({2, 0, 3}, 1.0);
    Check(zeroExtent.Shape() == std::vector<size_t>({2, 0, 3}), "zero-sized dimensions should be retained");
    Check(zeroExtent.Size() == 0, "a tensor with a zero-sized dimension should contain no elements");
}

void TestHandCalculatedNumerics()
{
    Tensor left({2, 3}, 0.0);
    left = {1.5, -2.0, 0.5, 4.0, -1.0, 3.0};
    Tensor right({2, 3}, 0.0);
    right = {0.5, 1.0, -1.5, 2.0, 3.0, -2.0};

    Tensor expectedAddition({2, 3}, 0.0);
    expectedAddition = {2.0, -1.0, -1.0, 6.0, 2.0, 1.0};
    CheckTensorEqual(left + right, expectedAddition, "signed fractional addition should be correct");

    Tensor expectedSubtraction({2, 3}, 0.0);
    expectedSubtraction = {1.0, -3.0, 2.0, 2.0, -4.0, 5.0};
    CheckTensorEqual(left - right, expectedSubtraction, "signed fractional subtraction should be correct");

    Tensor expectedMultiplication({2, 3}, 0.0);
    expectedMultiplication = {0.75, -2.0, -0.75, 8.0, -3.0, -6.0};
    CheckTensorEqual(left * right, expectedMultiplication, "signed fractional elementwise multiplication should be correct");

    Tensor expectedDivision({2, 3}, 0.0);
    expectedDivision = {3.0, -2.0, -1.0 / 3.0, 2.0, -1.0 / 3.0, -1.5};
    CheckTensorEqual(left / right, expectedDivision, "signed fractional elementwise division should be correct");

    Tensor expectedScalarMultiplication({2, 3}, 0.0);
    expectedScalarMultiplication = {3.75, -5.0, 1.25, 10.0, -2.5, 7.5};
    CheckTensorEqual(left * 2.5, expectedScalarMultiplication, "right scalar multiplication should be correct");
    CheckTensorEqual(2.5 * left, expectedScalarMultiplication, "left scalar multiplication should be correct");

    Tensor expectedScalarDivision({2, 3}, 0.0);
    expectedScalarDivision = {0.75, -1.0, 0.25, 2.0, -0.5, 1.5};
    CheckTensorEqual(left / 2.0, expectedScalarDivision, "scalar division should be correct");

    Tensor expectedTranspose({3, 2}, 0.0);
    expectedTranspose = {1.5, 4.0, -2.0, -1.0, 0.5, 3.0};
    CheckTensorEqual(left.Transpose(), expectedTranspose, "transpose should produce the expected shape and values");

    CheckEqual(left.Sum(), 6.0, "sum should add signed fractional values correctly");
    CheckEqual(left.Mean(), 1.0, "mean should average signed fractional values correctly");

    Tensor matrixLeft({2, 2}, 0.0);
    matrixLeft = {1.5, -2.0, 0.5, 4.0};
    Tensor matrixRight({2, 2}, 0.0);
    matrixRight = {2.0, 1.0, -1.0, 3.0};
    Tensor expectedMatmul({2, 2}, 0.0);
    expectedMatmul = {5.0, -4.5, -3.0, 12.5};
    CheckTensorEqual(Matmul(matrixLeft, matrixRight), expectedMatmul,
                     "small matrix multiplication should match its hand-calculated result");
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
    const Tensor originalLeft = left;
    const Tensor originalRight = right;
    CheckThrows<std::invalid_argument>([&] { (void)(left + differentShape); }, "addition shape mismatch should throw invalid_argument");
    CheckThrows<std::invalid_argument>([&] { (void)(left - differentShape); }, "subtraction shape mismatch should throw invalid_argument");
    CheckThrows<std::invalid_argument>([&] { (void)(left * differentShape); }, "multiplication shape mismatch should throw invalid_argument");
    CheckThrows<std::invalid_argument>([&] { (void)(left / differentShape); }, "division shape mismatch should throw invalid_argument");
    CheckThrows<std::invalid_argument>([&] { left += differentShape; }, "compound addition shape mismatch should throw invalid_argument");
    CheckThrows<std::invalid_argument>([&] { left -= differentShape; }, "compound subtraction shape mismatch should throw invalid_argument");
    CheckThrows<std::invalid_argument>([&] { left *= differentShape; }, "compound multiplication shape mismatch should throw invalid_argument");
    CheckThrows<std::invalid_argument>([&] { left /= differentShape; }, "compound division shape mismatch should throw invalid_argument");
    CheckTensorEqual(left, originalLeft, "failed elementwise operations should not modify the valid left operand");
    CheckTensorEqual(right, originalRight, "failed elementwise operations should not modify the valid right operand");
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
    const Tensor originalTensor = tensor;
    CheckThrows<std::invalid_argument>([&] { tensor.Reshape({4, 2}); },
                                      "reshape with a different element count should throw invalid_argument");
    CheckThrows<std::invalid_argument>([&] { (void)tensor.Reshaped({4, 2}); },
                                      "Reshaped with a different element count should throw invalid_argument");
    CheckTensorEqual(tensor, originalTensor, "failed reshape operations should not change the tensor");

    Tensor transposed = tensor.Transpose();
    Check(transposed.Shape() == std::vector<size_t>({3, 2}), "transpose should swap matrix dimensions");
    Tensor expectedTranspose({3, 2}, 0.0);
    expectedTranspose = {1, 4, 2, 5, 3, 6};
    CheckTensorEqual(transposed, expectedTranspose, "transpose should reorder values");
    CheckTensorEqual(tensor.Transpose().Transpose(), tensor, "double transpose should recover the original tensor");
    CheckTensorEqual(tensor.Transpose(0, 0), tensor, "transposing one dimension with itself should preserve data");
    CheckThrows<std::out_of_range>([&] { (void)tensor.Transpose(0, 2); },
                                   "invalid second transpose axis should throw out_of_range");
    CheckThrows<std::out_of_range>([&] { (void)tensor.Transpose(2, 0); },
                                   "invalid first transpose axis should throw out_of_range");
    CheckTensorEqual(tensor, originalTensor, "failed transpose operations should not change the tensor");

    tensor.Reshape(std::vector<size_t>{3, 2});
    Check(tensor.Shape() == std::vector<size_t>({3, 2}), "vector-based Reshape should update the shape");
    CheckTensorEqual(tensor, expectedReshaped, "Reshape should preserve values");

    Tensor threeDimensional({2, 2, 3}, 0.0);
    threeDimensional = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    CheckThrows<std::invalid_argument>([&] { (void)threeDimensional.Transpose(); },
                                       "default transpose should reject non-2D tensors with invalid_argument");
    Tensor expectedAxisTranspose({3, 2, 2}, 0.0);
    expectedAxisTranspose = {1, 7, 4, 10, 2, 8, 5, 11, 3, 9, 6, 12};
    CheckTensorEqual(threeDimensional.Transpose(0, 2), expectedAxisTranspose,
                     "N-D transpose should exchange the requested axes");

    Tensor emptyShape({}, 0.0);
    Check(emptyShape.Size() == 0, "empty shape should have zero elements");
    emptyShape.Reshape({});
    Check(emptyShape.Shape().empty() && emptyShape.Size() == 0,
          "reshaping an empty-shape tensor to its current shape should preserve its size");
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
    const Tensor originalA = a;
    const Tensor originalB = b;
    CheckThrows<std::invalid_argument>([&] { (void)Matmul(a, a); },
                                       "incompatible Matmul dimensions should throw invalid_argument");
    CheckThrows<std::invalid_argument>([&] { (void)Matmul(Tensor({2}, 1.0), b); },
                                       "Matmul should reject non-2D inputs with invalid_argument");
    CheckTensorEqual(a, originalA, "failed Matmul operations should not change the left input");
    CheckTensorEqual(b, originalB, "failed Matmul operations should not change the right input");

    Tensor zeroInnerLeft({2, 0}, 0.0);
    Tensor zeroInnerRight({0, 3}, 0.0);
    Tensor zeroInnerProduct = Matmul(zeroInnerLeft, zeroInnerRight);
    Check(zeroInnerProduct.Shape() == std::vector<size_t>({2, 3}),
          "Matmul with a zero inner dimension should retain the output shape");
    CheckTensorEqual(zeroInnerProduct, Tensor({2, 3}, 0.0),
                     "Matmul with a zero inner dimension should produce zeros");
}

void TestEmptyReductions()
{
    Tensor empty({0, 3}, 0.0);
    CheckEqual(empty.Sum(), 0.0, "sum of an empty tensor should be zero");
    CheckEqual(empty.Mean(), 0.0, "mean of an empty tensor should be zero");
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
        {"hand-calculated numerical results", TestHandCalculatedNumerics},
        {"arithmetic", TestArithmetic},
        {"reshape and transpose", TestReshapeAndTranspose},
        {"reductions and Matmul", TestReductionsAndMatmul},
        {"empty reductions", TestEmptyReductions},
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
#include <iostream>
#include <stdexcept>
#include <random>
#include <numeric>
#include <functional>
#include <algorithm>

#include <math/Tensor.h>

Tensor::Tensor(const std::vector<size_t>& shape, double initValue)
    : m_Shape(shape)
{
    CalculateStrides();
    m_Data.assign(m_Size, initValue);
}

Tensor::Tensor(std::initializer_list<size_t> shape, double initValue)
    : Tensor(std::vector<size_t>(shape), initValue)
{
}

size_t Tensor::GetOffset(const std::vector<size_t>& index) const
{
    if (index.size() != m_Shape.size())
        throw std::invalid_argument("Invalid dimension count");

    size_t offset = 0;

    for (size_t i = 0; i < m_Strides.size(); i++) {
        if (index[i] >= m_Shape[i])
            throw std::out_of_range("Index out of range");

        offset += index[i] * m_Strides[i];
    }

    return offset;
}

void Tensor::CalculateStrides()
{
    m_Strides.resize(m_Shape.size());

    if (m_Shape.empty()) {
        m_Size = 0;
        return;
    }

    m_Size = 1;
    for (size_t dimension : m_Shape)
        m_Size *= dimension;

    m_Strides.back() = 1;
    for (std::size_t i = m_Shape.size() - 1; i > 0; i--)
        m_Strides[i - 1] = m_Strides[i] * m_Shape[i];
}

double& Tensor::operator()(const std::vector<size_t>& index)
{
    return m_Data[GetOffset(index)];
}

const double& Tensor::operator()(const std::vector<size_t>& index) const
{
    return m_Data[GetOffset(index)];
}

double& Tensor::operator()(std::initializer_list<size_t> index)
{
    return (*this)(std::vector<size_t>(index));
}

const double& Tensor::operator()(std::initializer_list<size_t> index) const
{
    return (*this)(std::vector<size_t>(index));
}

Tensor& Tensor::operator=(std::initializer_list<double> values)
{
    if (values.size() != m_Size)
        throw std::invalid_argument("Initializer list size does not match tensor size");

    std::copy(values.begin(), values.end(), m_Data.begin());
    return *this;
}

Tensor Tensor::operator+(const Tensor& other) const
{
    if (m_Shape != other.m_Shape)
        throw std::invalid_argument("Tensor shape mismatch");

    Tensor result(m_Shape);
    for (size_t i = 0; i < m_Size; i++)
        result.m_Data[i] = m_Data[i] + other.m_Data[i];

    return result;
}

Tensor Tensor::operator-(const Tensor& other) const
{
    if (m_Shape != other.m_Shape)
        throw std::invalid_argument("Tensor shape mismatch");

    Tensor result(m_Shape);
    for (size_t i = 0; i < m_Size; i++)
        result.m_Data[i] = m_Data[i] - other.m_Data[i];

    return result;
}

Tensor Tensor::operator*(const Tensor& other) const
{
    if (m_Shape != other.m_Shape)
        throw std::invalid_argument("Tensor shape mismatch");

    Tensor result(m_Shape);
    for (size_t i = 0; i < m_Size; i++)
        result.m_Data[i] = m_Data[i] * other.m_Data[i];

    return result;
}

Tensor Tensor::operator/(const Tensor& other) const
{
    if (m_Shape != other.m_Shape)
        throw std::invalid_argument("Tensor shape mismatch");

    Tensor result(m_Shape);
    for (size_t i = 0; i < m_Size; i++)
        result.m_Data[i] = m_Data[i] / other.m_Data[i];

    return result;
}

Tensor& Tensor::operator+=(const Tensor& other)
{
    if (m_Shape != other.m_Shape)
        throw std::invalid_argument("Tensor shape mismatch");

    for (size_t i = 0; i < m_Size; i++)
        m_Data[i] += other.m_Data[i];

    return *this;
}

Tensor& Tensor::operator-=(const Tensor& other)
{
    if (m_Shape != other.m_Shape)
        throw std::invalid_argument("Tensor shape mismatch");

    for (size_t i = 0; i < m_Size; i++)
        m_Data[i] -= other.m_Data[i];

    return *this;
}

Tensor& Tensor::operator*=(const Tensor& other)
{
    if (m_Shape != other.m_Shape)
        throw std::invalid_argument("Tensor shape mismatch");

    for (size_t i = 0; i < m_Size; i++)
        m_Data[i] *= other.m_Data[i];

    return *this;
}

Tensor& Tensor::operator/=(const Tensor& other)
{
    if (m_Shape != other.m_Shape)
        throw std::invalid_argument("Tensor shape mismatch");

    for (size_t i = 0; i < m_Size; i++)
        m_Data[i] /= other.m_Data[i];

    return *this;
}

Tensor Tensor::operator*(double scalar) const
{
    Tensor result(m_Shape);
    for (size_t i = 0; i < m_Size; i++)
        result.m_Data[i] = m_Data[i] * scalar;
    return result;
}

Tensor Tensor::operator/(double scalar) const
{
    Tensor result(m_Shape);
    for (size_t i = 0; i < m_Size; i++)
        result.m_Data[i] = m_Data[i] / scalar;
    return result;
}

Tensor& Tensor::operator*=(double scalar)
{
    for (size_t i = 0; i < m_Size; i++)
        m_Data[i] *= scalar;
    return *this;
}

Tensor& Tensor::operator/=(double scalar)
{
    for (size_t i = 0; i < m_Size; i++)
        m_Data[i] /= scalar;
    return *this;
}

void Tensor::Reshape(const std::vector<size_t>& newShape)
{
    size_t newSize = newShape.empty() ? 0 : 1;
    for (size_t dim : newShape)
        newSize *= dim;

    if (newSize != m_Size)
        throw std::invalid_argument("Total size must remain identical for reshape");

    m_Shape = newShape;
    CalculateStrides();
}

void Tensor::Reshape(std::initializer_list<size_t> newShape)
{
    Reshape(std::vector<size_t>(newShape));
}

Tensor Tensor::Reshaped(const std::vector<size_t>& newShape) const
{
    Tensor copy = *this;
    copy.Reshape(newShape);
    return copy;
}

Tensor Tensor::Reshaped(std::initializer_list<size_t> newShape) const
{
    return Reshaped(std::vector<size_t>(newShape));
}

Tensor Tensor::Transpose() const
{
    if (m_Shape.size() != 2)
        throw std::invalid_argument("Default Transpose() only supports 2D tensors. Use Transpose(dim0, dim1) for N-D tensors.");
    
    return Transpose(0, 1);
}

Tensor Tensor::Transpose(size_t dim0, size_t dim1) const
{
    if (dim0 >= m_Shape.size() || dim1 >= m_Shape.size())
        throw std::out_of_range("Dimension index out of range for transpose");

    if (dim0 == dim1)
        return *this;

    std::vector<size_t> newShape = m_Shape;
    std::swap(newShape[dim0], newShape[dim1]);

    Tensor result(newShape);

    std::vector<size_t> currentIdx(m_Shape.size(), 0);
    for (size_t i = 0; i < m_Size; i++) {
        size_t temp = i;
        for (size_t d = 0; d < m_Shape.size(); d++) {
            currentIdx[d] = temp / m_Strides[d];
            temp %= m_Strides[d];
        }

        std::vector<size_t> targetIdx = currentIdx;
        std::swap(targetIdx[dim0], targetIdx[dim1]);

        result(targetIdx) = m_Data[i];
    }

    return result;
}

Tensor Matmul(const Tensor& a, const Tensor& b)
{
    if (a.m_Shape.size() != 2 || b.m_Shape.size() != 2)
        throw std::invalid_argument("Matmul requires 2D tensors");

    if (a.m_Shape[1] != b.m_Shape[0])
        throw std::invalid_argument("Tensor dimensions mismatch for Matmul");

    const size_t M = a.m_Shape[0];
    const size_t K = a.m_Shape[1];
    const size_t N = b.m_Shape[1];

    Tensor result({M, N}, 0.0);

    const double* A = a.RawData();
    const double* B = b.RawData();
    double* C = result.RawData();

    for (size_t i = 0; i < M; ++i) {
        for (size_t k = 0; k < K; ++k) {
            const double aik = A[i * K + k];

            for (size_t j = 0; j < N; ++j) {
                C[i * N + j] += aik * B[k * N + j];
            }
        }
    }

    return result;
}

void Tensor::Fill(double value)
{
    std::fill(m_Data.begin(), m_Data.end(), value);
}

void Tensor::Randomize(double min, double max)
{
    static thread_local std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<double> dist(min, max);

    for (double& val : m_Data)
        val = dist(gen);
}

double Tensor::Sum() const
{
    return std::accumulate(m_Data.begin(), m_Data.end(), 0.0);
}

double Tensor::Mean() const
{
    if (m_Size == 0) return 0.0;

    return Sum() / static_cast<double>(m_Size);
}

void Tensor::Print(std::ostream& os) const
{
    os << "Tensor(shape=[";

    for (size_t i = 0; i < m_Shape.size(); ++i) {
        os << m_Shape[i];

        if (i + 1 < m_Shape.size())
            os << ", ";
    }

    os << "], data=\n";

    if (m_Size == 0) {
        os << "[]\n)\n";
        return;
    }

    auto printRecursive = [&](auto&& self, size_t dim, size_t offset, size_t depth) -> void {
        if (dim == m_Shape.size() - 1) {
            os << std::string(depth * 2, ' ') << "[";

            for (size_t i = 0; i < m_Shape[dim]; ++i) {
                os << m_Data[offset + i * m_Strides[dim]];

                if (i + 1 < m_Shape[dim])
                    os << ", ";
            }

            os << "]";
            return;
        }

        os << std::string(depth * 2, ' ') << "[\n";

        for (size_t i = 0; i < m_Shape[dim]; ++i) {
            self(
                self,
                dim + 1,
                offset + i * m_Strides[dim],
                depth + 1
            );

            if (i + 1 < m_Shape[dim])
                os << ",\n";
        }

        os << "\n" << std::string(depth * 2, ' ') << "]";
    };

    printRecursive(printRecursive, 0, 0, 1);

    os << "\n)\n";
}

Tensor Tensor::Random(const std::vector<size_t>& shape, double min, double max)
{
    Tensor t(shape);
    t.Randomize(min, max);
    return t;
}

Tensor operator*(double scalar, const Tensor& tensor)
{
    return tensor * scalar;
}

std::ostream& operator<<(std::ostream& os, const Tensor& tensor)
{
    tensor.Print(os);
    return os;
}
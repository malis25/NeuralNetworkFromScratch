#pragma once

#include <vector>
#include <initializer_list>
#include <iostream>

class Tensor
{
public:
    explicit Tensor(const std::vector<size_t>& shape, double initValue = 0.0);
    Tensor(std::initializer_list<size_t> shape, double initValue = 0.0);

    const std::vector<size_t>& Shape() const { return m_Shape; }
    const std::vector<size_t>& Strides() const { return m_Strides; }
    size_t Size() const { return m_Size; }
    size_t Dim() const { return m_Shape.size(); }

    const std::vector<double>& Data() const { return m_Data; }
    std::vector<double>& Data() { return m_Data; }
    const double* RawData() const { return m_Data.data(); }
    double* RawData() { return m_Data.data(); }

    double& operator()(const std::vector<size_t>& index);
    const double& operator()(const std::vector<size_t>& index) const;
    double& operator()(std::initializer_list<size_t> index);
    const double& operator()(std::initializer_list<size_t> index) const;

    double& operator[](size_t index) { return m_Data[index]; }
    const double& operator[](size_t index) const { return m_Data[index]; }

    Tensor& operator=(std::initializer_list<double> values);

    Tensor operator+(const Tensor& other) const;
    Tensor operator-(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const;
    Tensor operator/(const Tensor& other) const;

    Tensor& operator+=(const Tensor& other);
    Tensor& operator-=(const Tensor& other);
    Tensor& operator*=(const Tensor& other);
    Tensor& operator/=(const Tensor& other);

    Tensor operator*(double scalar) const;
    Tensor operator/(double scalar) const;

    Tensor& operator*=(double scalar);
    Tensor& operator/=(double scalar);

    void Reshape(const std::vector<size_t>& newShape);
    void Reshape(std::initializer_list<size_t> newShape);
    Tensor Reshaped(const std::vector<size_t>& newShape) const;
    Tensor Reshaped(std::initializer_list<size_t> newShape) const;
    Tensor Transpose() const;
    Tensor Transpose(size_t dim0, size_t dim1) const;


    void Fill(double value);
    void Randomize(double min = -1.0, double max = 1.0);
    double Sum() const;
    double Mean() const;
    void Print(std::ostream& os = std::cout) const;

    friend Tensor Matmul(const Tensor& a, const Tensor& b);

    static Tensor Random(const std::vector<size_t>& shape, double min = -1.0, double max = 1.0);

private:
    std::vector<double> m_Data;
    std::vector<size_t> m_Shape;
    std::vector<size_t> m_Strides;
    size_t m_Size = 0;

    void CalculateStrides();
    size_t GetOffset(const std::vector<size_t>& index) const;
};

Tensor operator*(double scalar, const Tensor& tensor);
Tensor operator+(double scalar, const Tensor& tensor);
Tensor Matmul(const Tensor& a, const Tensor& b);
std::ostream& operator<<(std::ostream& os, const Tensor& tensor);
#pragma once

#include <iostream>
#include <vector>

class Matrix {
public:
    Matrix(size_t rows, size_t cols);

    size_t Rows() const { return m_Rows; }
    size_t Cols() const { return m_Cols; }

    double& operator()(size_t row, size_t col);
    const double& operator()(size_t row, size_t col) const;

    double & operator[](size_t index) { return m_Data[index]; }

    Matrix operator=(const std::initializer_list<double> values);

    Matrix operator+(const Matrix& other) const;
    Matrix operator-(const Matrix& other) const;
    Matrix operator*(const Matrix& other) const;

    Matrix operator*(double scalar) const;

    void Randomize();
    Matrix Transpose() const;

private:
    size_t m_Rows;
    size_t m_Cols;

    std::vector<double> m_Data;
};


Matrix operator*(double scalar, const Matrix& matrix);
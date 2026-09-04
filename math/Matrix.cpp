#include <math/Matrix.h>
#include <random>

Matrix::Matrix(size_t rows, size_t cols)
    : m_Rows(rows), m_Cols(cols), m_Data(rows * cols, 0.0)
{
}

double& Matrix::operator()(size_t row, size_t col)
{
    if (row >= m_Rows || col >= m_Cols) {
        throw std::out_of_range("Index out of bounds");
    }

    return m_Data[row * m_Cols + col];
}

const double& Matrix::operator()(size_t row, size_t col) const
{
    if (row >= m_Rows || col >= m_Cols) {
        throw std::out_of_range("Index out of bounds");
    }

    return m_Data[row * m_Cols + col];
}

Matrix Matrix::operator=(const std::initializer_list<double> values)
{
    if (values.size() != m_Rows * m_Cols) {
        throw std::invalid_argument("Initializer list size does not match matrix dimensions");
    }

    std::copy(values.begin(), values.end(), m_Data.begin());

    return *this;
}

Matrix Matrix::operator+(const Matrix& other) const
{
    if (m_Rows != other.m_Rows || m_Cols != other.m_Cols) {
        throw std::invalid_argument("Matrix dimensions must match for addition");
    }

    Matrix result(m_Rows, m_Cols);

    for (size_t i = 0; i < m_Data.size(); i++) {
        result.m_Data[i] = m_Data[i] + other.m_Data[i];
    }

    return result;
}

Matrix Matrix::operator-(const Matrix& other) const
{
    if (m_Rows != other.m_Rows || m_Cols != other.m_Cols) {
        throw std::invalid_argument("Matrix dimensions must match for subtraction");
    }

    Matrix result(m_Rows, m_Cols);
    for (size_t i = 0; i < m_Data.size(); i++) {
        result.m_Data[i] = m_Data[i] - other.m_Data[i];
    }

    return result;
}

Matrix Matrix::operator*(const Matrix& other) const
{
    if (m_Cols != other.m_Rows) {
        throw std::invalid_argument("Matrix dimensions must match for multiplication");
    }

    Matrix result(m_Rows, other.m_Cols);
    
    for (size_t i = 0; i < m_Rows; i++) {
        for (size_t k = 0; k < m_Cols; k++) {
            const double left = m_Data[i * m_Cols + k];
            for (size_t j = 0; j < other.m_Cols; j++) {
                result.m_Data[i * other.m_Cols + j] +=
                    left * other.m_Data[k * other.m_Cols + j];
            }
        }
    }

    return result;
}

Matrix Matrix::operator*(double scalar) const
{
    Matrix result(m_Rows, m_Cols);

    for (size_t i = 0; i < m_Data.size(); i++) {
        result.m_Data[i] = m_Data[i] * scalar;
    }

    return result;
}

Matrix Matrix::Transpose() const
{
    Matrix transposed(m_Cols, m_Rows);

    for (size_t i = 0; i < m_Rows; i++) {
        for (size_t j = 0; j < m_Cols; j++) {
            transposed(j, i) = m_Data[i * m_Cols + j];
        }
    }

    return transposed;
}

void Matrix::Randomize()
{
    static thread_local std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    for (double& value : m_Data) {
        value = dist(gen);
    }
}

Matrix operator*(double scalar, const Matrix& matrix)
{
    return matrix * scalar;
}
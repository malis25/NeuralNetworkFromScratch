#pragma once

#include <initializer_list>
#include <vector>

#include <math/Matrix.h>

class MLP {
public:
    explicit MLP(const std::vector<size_t>& layerSizes);
    MLP(std::initializer_list<size_t> layerSizes);

    Matrix Predict(const Matrix& input) const;
    void Train(const std::vector<Matrix>& inputs,
               const std::vector<Matrix>& targets,
               size_t epochs,
               double learningRate);

private:
    std::vector<Matrix> m_Weights;
    std::vector<Matrix> m_Biases;

    static Matrix HadamardProduct(const Matrix& left, const Matrix& right);
    void ValidateInput(const Matrix& input) const;
};
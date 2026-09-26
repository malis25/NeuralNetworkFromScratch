#pragma once

#include <initializer_list>
#include <vector>

#include <math/Tensor.h>

class MLP {
public:
    explicit MLP(const std::vector<size_t>& layerSizes);
    MLP(std::initializer_list<size_t> layerSizes);

    Tensor Predict(const Tensor& input) const;
    void Train(const std::vector<Tensor>& inputs,
               const std::vector<Tensor>& targets,
               size_t epochs,
               double learningRate);

private:
    std::vector<Tensor> m_Weights;
    std::vector<Tensor> m_Biases;

    void ValidateInput(const Tensor& input) const;
};
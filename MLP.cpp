#include <cmath>
#include <stdexcept>
#include <fstream>

#include <math/Activations.h>
#include <MLP.h>

MLP::MLP(const std::vector<size_t>& layerSizes)
{
    if (layerSizes.size() < 2)
        throw std::invalid_argument("An MLP needs at least an input and output layer");

    m_Weights.reserve(layerSizes.size() - 1);
    m_Biases.reserve(layerSizes.size() - 1);

    for (size_t layer = 0; layer + 1 < layerSizes.size(); layer++) {
        if (layerSizes[layer] == 0 || layerSizes[layer + 1] == 0)
            throw std::invalid_argument("MLP layer sizes must be greater than zero");

        m_Weights.emplace_back(Tensor{layerSizes[layer + 1], layerSizes[layer]});
        m_Weights.back().Randomize();

        m_Biases.emplace_back(Tensor{layerSizes[layer + 1], 1});
        m_Biases.back().Randomize();
    }
}

MLP::MLP(std::initializer_list<size_t> layerSizes)
    : MLP(std::vector<size_t>(layerSizes))
{
}

Tensor MLP::Predict(const Tensor& input) const
{
    ValidateInput(input);

    Tensor activation = input;

    for (size_t layer = 0; layer < m_Weights.size(); layer++) {
        activation = Matmul(m_Weights[layer], activation) + m_Biases[layer];
        activation = layer + 1 == m_Weights.size() ? Sigmoid(activation) : ReLU(activation);
    }

    return activation;
}

void MLP::Train(const std::vector<Tensor>& inputs,
                const std::vector<Tensor>& targets,
                size_t epochs,
                double learningRate)
{
    std::vector<double> lossHistory;
    std::vector<Tensor> activations;
    std::vector<Tensor> deltas;
    std::vector<Tensor> weightVelocity;
    std::vector<Tensor> biasVelocity;

    activations.reserve(m_Weights.size() + 1);
    deltas.reserve(m_Weights.size());
    weightVelocity.reserve(m_Weights.size());
    biasVelocity.reserve(m_Weights.size());
    
    for (const Tensor& weights : m_Weights) {
        deltas.emplace_back(Tensor{weights.Shape()[0], 1});
        biasVelocity.emplace_back(Tensor{weights.Shape()[0], 1});
        weightVelocity.emplace_back(Tensor{weights.Shape()[0], weights.Shape()[1]});
    }


    if (inputs.empty() || inputs.size() != targets.size())
        throw std::invalid_argument("Inputs and targets must contain the same non-zero number of samples");

    if (learningRate <= 0.0)
        throw std::invalid_argument("Learning rate must be greater than zero");

    for (size_t sample = 0; sample < inputs.size(); sample++) {
        ValidateInput(inputs[sample]);

        if (targets[sample].Shape()[0] != m_Biases.back().Shape()[0] || targets[sample].Shape()[1] != 1)
            throw std::invalid_argument("Target dimensions do not match the output layer");
    }

    for (size_t epoch = 0; epoch < epochs; epoch++) {
        for (size_t sample = 0; sample < inputs.size(); sample++) {
            activations.push_back(inputs[sample]);

            for (size_t layer = 0; layer < m_Weights.size(); layer++) {
                Tensor activation = Matmul(m_Weights[layer], activations.back()) + m_Biases[layer];
                activations.push_back(
                    layer + 1 == m_Weights.size() ? Sigmoid(activation) : ReLU(activation));
            }

            size_t outputLayer = m_Weights.size() - 1;

            deltas[outputLayer] = (activations.back() - targets[sample]) * SigmoidDerivativeFromActivation(activations.back());
            
            for (size_t layer = outputLayer; layer > 0; layer--) {
                deltas[layer - 1] = Matmul(
                    m_Weights[layer].Transpose(),
                    deltas[layer]) * ReLUDerivativeFromActivation(activations[layer]
                    );
            }

            for (size_t layer = 0; layer < m_Weights.size(); layer++) {
                Tensor weightGradient = Matmul(deltas[layer], activations[layer].Transpose());

                weightVelocity[layer] *= 0.99;
                weightVelocity[layer] -= learningRate * weightGradient;

                biasVelocity[layer] *= 0.99;
                biasVelocity[layer] -= learningRate * deltas[layer];

                m_Weights[layer] += weightVelocity[layer];
                m_Biases[layer] += biasVelocity[layer];
            }

            activations.clear();
        }

        double loss = 0;

        for (size_t sample = 0; sample < inputs.size(); sample++)
            loss += ((Predict(inputs[sample])[0] - targets[sample][0]) * (Predict(inputs[sample])[0] - targets[sample][0])) / 2.0;

        lossHistory.push_back(loss);
    }

    std::ofstream file("loss.csv");

    for (size_t i = 0; i < lossHistory.size(); ++i)
        file << i << "," << lossHistory[i] << "\n";
}

void MLP::ValidateInput(const Tensor& input) const
{
    if (input.Shape()[0] != m_Weights.front().Shape()[1] || input.Shape()[1] != 1)
        throw std::invalid_argument("Input must be a column vector matching the input layer");
}
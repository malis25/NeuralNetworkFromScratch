#include <iostream>
#include <vector>

#include <MLP.h>

int main() {
    MLP network({2, 4, 1});

    size_t epochs;
    double learningRate;

    std::vector<Matrix> inputs(4, Matrix(2, 1));
    inputs[0] = {0.0, 0.0};
    inputs[1] = {0.0, 1.0};
    inputs[2] = {1.0, 0.0};
    inputs[3] = {1.0, 1.0};

    std::vector<Matrix> targets(4, Matrix(1, 1));
    targets[0] = {0.0};
    targets[1] = {1.0};
    targets[2] = {1.0};
    targets[3] = {0.0};

    std::cout << "Enter the number of epochs: ";
    std::cin >> epochs;

    std::cout << "Enter the learning rate: ";
    std::cin >> learningRate;

    network.Train(inputs, targets, epochs, learningRate);

    for (size_t sample = 0; sample < inputs.size(); sample++) {
        Matrix prediction = network.Predict(inputs[sample]);
        std::cout << inputs[sample](0, 0) << " XOR " << inputs[sample](1, 0)
                  << " = " << prediction(0, 0);

        std::cout << " Loss: " << std::abs(targets[sample](0, 0) - prediction(0, 0)) << std::endl;
    }

    return 0;
}
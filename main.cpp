#include <iostream>
#include <vector>

#include "math/MLP.h"

using namespace std;

int main() {
    MLP network({2, 4, 1});

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

    network.Train(inputs, targets, 10000, 0.1);

    for (size_t sample = 0; sample < inputs.size(); sample++) {
        Matrix prediction = network.Predict(inputs[sample]);
        std::cout << inputs[sample](0, 0) << " XOR " << inputs[sample](1, 0)
                  << " = " << prediction(0, 0) << std::endl;

        std::cout << "Loss: " << targets[sample](0, 0) - prediction(0, 0) << std::endl;
    }

    return 0;
}
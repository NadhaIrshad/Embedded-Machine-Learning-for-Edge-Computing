#include <math.h>
#include "model.h"
#include "model_data.h"

// Function for ReLU activation
inline float relu(float x) {
    // TODO: Implement the ReLU activation function.
    return fmax(0.0f, x); //
}

// Function to perform forward pass of the model
float predict(float x) {
    // The activations of the first layer are small enough to store on the stack (16 floats = 64 bytes).
    float h1[DENSE1_SIZE];

    // First dense layer. Since there is only one input neuron, we don't need to perform a full-blown matrix multiply.
    for (int i = 0; i < DENSE1_SIZE; ++i) {
        // TODO: Compute the activation of the first layer (16 x 1 vector)
        h1[i] = relu(W1(i) * x + b1(i)); //
    }

    // Second dense layer.
    float h2[DENSE2_SIZE] = {0.0f};
    for (int i = 0; i < DENSE2_SIZE; ++i) {
        for (int j = 0; j < DENSE1_SIZE; ++j) {
            // TODO: Compute the activation of the second layer. (1 x 1 Scalar)
            h2[i] += W2(i, j) * h1[j];
        }
        h2[i] = relu(h2[i] + b2(i));
    }

    // Output layer.
    float y(0.0f);
    for (int i = 0; i < DENSE2_SIZE; ++i) {
        // TODO: Compute the output activation. (1 x 1 Scalar)
        y += W3(i) * h2[i];
    }

    // Final dense layer.
    return y + b3();
}

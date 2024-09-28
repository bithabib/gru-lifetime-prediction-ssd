#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Define the size of input and hidden layers
#define INPUT_SIZE 4   // Number of input features
#define HIDDEN_SIZE 10 // Size of hidden state

// Activation functions
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

double tanh_act(double x) {
    return tanh(x);
}

// Load weights (these should match the dimensions of the GRU model in TensorFlow)
void load_weights(double Wz[HIDDEN_SIZE][INPUT_SIZE], double Uz[HIDDEN_SIZE][HIDDEN_SIZE], double bz[HIDDEN_SIZE]) {
    FILE *file = fopen("gru_weights.txt", "r");
    if (!file) {
        printf("Error loading weights file\n");
        exit(1);
    }

    // Load Wz
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        for (int j = 0; j < INPUT_SIZE; j++) {
            fscanf(file, "%lf", &Wz[i][j]);
        }
    }

    // Load Uz
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            fscanf(file, "%lf", &Uz[i][j]);
        }
    }

    // Load bz
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        fscanf(file, "%lf", &bz[i]);
    }

    fclose(file);
}

// GRU forward pass
void gru_forward(double x[INPUT_SIZE], double h_prev[HIDDEN_SIZE], double Wz[HIDDEN_SIZE][INPUT_SIZE], 
                 double Uz[HIDDEN_SIZE][HIDDEN_SIZE], double bz[HIDDEN_SIZE], double h_new[HIDDEN_SIZE]) {
    double z[HIDDEN_SIZE], h_tilde[HIDDEN_SIZE], r[HIDDEN_SIZE];

    // Update gate z
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        z[i] = bz[i];
        for (int j = 0; j < INPUT_SIZE; j++) {
            z[i] += Wz[i][j] * x[j];
        }
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            z[i] += Uz[i][j] * h_prev[j];
        }
        z[i] = sigmoid(z[i]);
    }

    // Candidate hidden state h_tilde
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        h_tilde[i] = 0;
        for (int j = 0; j < INPUT_SIZE; j++) {
            h_tilde[i] += Wz[i][j] * x[j];  // Use the same weights as z for simplicity
        }
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            h_tilde[i] += Uz[i][j] * (z[i] * h_prev[j]);  // Reset gate effect
        }
        h_tilde[i] = tanh_act(h_tilde[i]);
    }

    // Final hidden state
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        h_new[i] = (1 - z[i]) * h_prev[i] + z[i] * h_tilde[i];
    }
}

int main() {
    double x[INPUT_SIZE] = {100, 32, 1, 1}; // Example input
    double h_prev[HIDDEN_SIZE] = {0};  // Initial hidden state
    double h_new[HIDDEN_SIZE];         // New hidden state

    // Define weights (load them from saved files)
    double Wz[HIDDEN_SIZE][INPUT_SIZE], Uz[HIDDEN_SIZE][HIDDEN_SIZE], bz[HIDDEN_SIZE];
    
    // Load weights from file
    load_weights(Wz, Uz, bz);

    // Perform forward pass
    gru_forward(x, h_prev, Wz, Uz, bz, h_new);

    // Print the new hidden state
    printf("New hidden state:\n");
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        printf("%lf ", h_new[i]);
    }

    return 0;
}

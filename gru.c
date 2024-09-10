#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

#define INPUT_SIZE 4   // Number of features: rw_rat, is_seq, chunk_w, chunk_r, etc.
#define HIDDEN_SIZE 10 // Number of units in the GRU's hidden state
#define OUTPUT_SIZE 1  // We are predicting life_t, which is a single value
#define LEARNING_RATE 0.001
#define EPOCHS 1000

// Activation functions
double sigmoid(double x)
{
    return 1.0 / (1.0 + exp(-x));
}

double dsigmoid(double x)
{
    return x * (1.0 - x);
}

double tanh_activation(double x)
{
    return tanh(x);
}

double dtanh_activation(double x)
{
    return 1.0 - x * x;
}

// Random initialization
double random_weight()
{
    return (double)rand() / RAND_MAX * 2.0 - 1.0; // Random values between -1 and 1
}

// Struct to represent the GRU model
typedef struct
{
    double Wz[HIDDEN_SIZE][INPUT_SIZE];  // Update gate weights
    double Uz[HIDDEN_SIZE][HIDDEN_SIZE]; // Recurrent update gate weights
    double Wr[HIDDEN_SIZE][INPUT_SIZE];  // Reset gate weights
    double Ur[HIDDEN_SIZE][HIDDEN_SIZE]; // Recurrent reset gate weights
    double Wh[HIDDEN_SIZE][INPUT_SIZE];  // Candidate hidden state weights
    double Uh[HIDDEN_SIZE][HIDDEN_SIZE]; // Recurrent candidate hidden state weights

    double bz[HIDDEN_SIZE]; // Update gate bias
    double br[HIDDEN_SIZE]; // Reset gate bias
    double bh[HIDDEN_SIZE]; // Candidate hidden state bias

    double hidden_state[HIDDEN_SIZE]; // Hidden state of the GRU
} GRU;

// Initialize the GRU weights randomly
void initialize_gru(GRU *gru)
{
    for (int i = 0; i < HIDDEN_SIZE; i++)
    {
        for (int j = 0; j < INPUT_SIZE; j++)
        {
            gru->Wz[i][j] = random_weight();
            gru->Wr[i][j] = random_weight();
            gru->Wh[i][j] = random_weight();
        }
        for (int j = 0; j < HIDDEN_SIZE; j++)
        {
            gru->Uz[i][j] = random_weight();
            gru->Ur[i][j] = random_weight();
            gru->Uh[i][j] = random_weight();
        }
        gru->bz[i] = random_weight();
        gru->br[i] = random_weight();
        gru->bh[i] = random_weight();
        gru->hidden_state[i] = 0.0; // Initialize hidden state to zero
    }
}

// Forward pass through the GRU cell
void gru_forward(GRU *gru, double *input, double *output)
{
    double z[HIDDEN_SIZE], r[HIDDEN_SIZE], h_tilde[HIDDEN_SIZE], h_next[HIDDEN_SIZE];

    // Compute update and reset gates
    for (int i = 0; i < HIDDEN_SIZE; i++)
    {
        double z_sum = gru->bz[i];
        double r_sum = gru->br[i];
        for (int j = 0; j < INPUT_SIZE; j++)
        {
            z_sum += gru->Wz[i][j] * input[j];
            r_sum += gru->Wr[i][j] * input[j];
        }
        for (int j = 0; j < HIDDEN_SIZE; j++)
        {
            z_sum += gru->Uz[i][j] * gru->hidden_state[j];
            r_sum += gru->Ur[i][j] * gru->hidden_state[j];
        }
        z[i] = sigmoid(z_sum);
        r[i] = sigmoid(r_sum);
    }

    // Compute candidate hidden state
    for (int i = 0; i < HIDDEN_SIZE; i++)
    {
        double h_sum = gru->bh[i];
        for (int j = 0; j < INPUT_SIZE; j++)
        {
            h_sum += gru->Wh[i][j] * input[j];
        }
        for (int j = 0; j < HIDDEN_SIZE; j++)
        {
            h_sum += gru->Uh[i][j] * gru->hidden_state[j] * r[i];
        }
        h_tilde[i] = tanh_activation(h_sum);
    }

    // Compute next hidden state
    for (int i = 0; i < HIDDEN_SIZE; i++)
    {
        h_next[i] = (1 - z[i]) * gru->hidden_state[i] + z[i] * h_tilde[i];
    }

    // Update hidden state
    for (int i = 0; i < HIDDEN_SIZE; i++)
    {
        gru->hidden_state[i] = h_next[i];
    }

    // Output layer (linear)
    output[0] = 0.0;
    for (int i = 0; i < HIDDEN_SIZE; i++)
    {
        output[0] += h_next[i]; // Simple linear combination
    }
}

void gru_backward(GRU *gru, double *input, double target, double output, double learning_rate)
{
    // Gradient of the loss with respect to the output (for Mean Squared Error)
    double d_loss = 2 * (output - target); // dL/dy

    double dz[HIDDEN_SIZE], dr[HIDDEN_SIZE], dh_tilde[HIDDEN_SIZE];
    double dWh[HIDDEN_SIZE][INPUT_SIZE], dUh[HIDDEN_SIZE][HIDDEN_SIZE];
    double dWz[HIDDEN_SIZE][INPUT_SIZE], dUz[HIDDEN_SIZE][HIDDEN_SIZE];
    double dWr[HIDDEN_SIZE][INPUT_SIZE], dUr[HIDDEN_SIZE][HIDDEN_SIZE];

    double dz_sum[HIDDEN_SIZE] = {0.0}, dr_sum[HIDDEN_SIZE] = {0.0};

    // Backpropagate through the output layer (since it’s a linear combination of the hidden state)
    for (int i = 0; i < HIDDEN_SIZE; i++)
    {
        // Gradients for the candidate hidden state
        dz[i] = d_loss * (gru->hidden_state[i] - tanh_activation(gru->hidden_state[i])); // dz/dh_tilde
        dh_tilde[i] = dz[i] * dtanh_activation(gru->hidden_state[i]);                    // dL/dh_tilde
        dr[i] = d_loss * dtanh_activation(gru->hidden_state[i]);

        // Compute the gradients for Wz, Uz, Wh, Uh, Wr, and Ur
        for (int j = 0; j < INPUT_SIZE; j++)
        {
            dWz[i][j] = dz[i] * input[j];
            dWh[i][j] = dh_tilde[i] * input[j];
            dWr[i][j] = dr[i] * input[j];
        }

        for (int j = 0; j < HIDDEN_SIZE; j++)
        {
            dUz[i][j] = dz[i] * gru->hidden_state[j];
            dUh[i][j] = dh_tilde[i] * gru->hidden_state[j];
            dUr[i][j] = dr[i] * gru->hidden_state[j];
        }
    }

    // Update the weights using gradient descent
    for (int i = 0; i < HIDDEN_SIZE; i++)
    {
        for (int j = 0; j < INPUT_SIZE; j++)
        {
            gru->Wz[i][j] -= learning_rate * dWz[i][j];
            gru->Wh[i][j] -= learning_rate * dWh[i][j];
            gru->Wr[i][j] -= learning_rate * dWr[i][j];
        }
        for (int j = 0; j < HIDDEN_SIZE; j++)
        {
            gru->Uz[i][j] -= learning_rate * dUz[i][j];
            gru->Uh[i][j] -= learning_rate * dUh[i][j];
            gru->Ur[i][j] -= learning_rate * dUr[i][j];
        }
    }
}

// Training function
void train_gru(GRU *gru, double input[][INPUT_SIZE], double target[], int dataset_size)
{
    for (int epoch = 0; epoch < EPOCHS; epoch++)
    {
        double total_loss = 0.0;

        for (int i = 0; i < dataset_size; i++)
        {
            double output[OUTPUT_SIZE];

            // Forward pass: Get the prediction
            gru_forward(gru, input[i], output);

            // Compute the loss (MSE)
            double error = target[i] - output[0];
            total_loss += error * error;

            // Call backpropagation to adjust weights
            gru_backward(gru, input[i], target[i], output[0], LEARNING_RATE);

            printf("Epoch %d, Data %d, Loss: %f\n", epoch, i, error * error);
        }

        printf("Epoch %d, Total Loss: %f\n", epoch, total_loss / dataset_size);
    }
}

// Evaluation function
void evaluate_model(GRU *gru, double input[][INPUT_SIZE], double target[], int dataset_size)
{
    double total_mae = 0.0;
    double total_mse = 0.0;
    double total_r_squared = 0.0;
    double mean_target = 0.0;

    // Calculate mean of target values
    for (int i = 0; i < dataset_size; i++)
    {
        mean_target += target[i];
    }
    mean_target /= dataset_size;

    for (int i = 0; i < dataset_size; i++)
    {
        double output[OUTPUT_SIZE];

        // Forward pass: Get the prediction
        gru_forward(gru, input[i], output);

        // Compute the errors
        double error = target[i] - output[0];
        total_mae += fabs(error);
        total_mse += error * error;

        // R-squared calculation
        total_r_squared += (target[i] - mean_target) * (target[i] - mean_target);
    }

    double mse = total_mse / dataset_size;
    double rmse = sqrt(mse);
    double r_squared = 1.0 - (total_mse / total_r_squared);

    printf("Mean Absolute Error (MAE): %f\n", total_mae / dataset_size);
    printf("Mean Squared Error (MSE): %f\n", mse);
    printf("Root Mean Squared Error (RMSE): %f\n", rmse);
    printf("R-squared (R²): %f\n", r_squared);
}

int count_lines(FILE *file)
{
    int lines = 0;
    char ch;
    while (!feof(file))
    {
        ch = fgetc(file);
        if (ch == '\n')
        {
            lines++;
        }
    }
    rewind(file);
    return lines;
}

int main()
{

    // Read dataset from .log file
    FILE *file = fopen("Data/NewData/FIO_test.log", "r");
    if (file == NULL)
    {
        printf("Error: Could not open file\n");
        return 1;
    }

    srand(time(NULL));

    // Define a GRU model
    GRU gru;
    initialize_gru(&gru);

    // count the number of lines in the file
    int dataset_size = count_lines(file);
    printf("Number of lines: %d\n", dataset_size);

    // Create a 2D array to store the input data
    double(*input_data)[INPUT_SIZE] = malloc(dataset_size * sizeof(*input_data));
    double *target_data = (double *)malloc(dataset_size * sizeof(double));

    if (input_data == NULL)
    {
        printf("Error: Memory allocation failed for input_data\n");
        fclose(file);
        return 1;
    }
    // for (int i = 0; i < dataset_size; i++)
    // {
    //     input_data[i] = (double *)malloc(INPUT_SIZE * sizeof(double));
    //     if (input_data[i] == NULL)
    //     {
    //         printf("Error: Memory allocation failed for input_data[%d]\n", i);
    //         fclose(file);
    //         return 1;
    //     }
    // }
    int read_counter = 1;
    int write_counter = 1;
    for (int i = 0; i < dataset_size; i++)
    {

        target_data[i] = 0;
        char readWrite;
        double a, b;
        fscanf(file, "%lf %lf %c %lf %lf", &a, &b, &readWrite, &input_data[i][0], &input_data[i][1]);
        input_data[i][3] = 0;
        if (i != 0)
        {
            if (input_data[i][0] == input_data[i - 1][0] + input_data[i - 1][1])
            {
                input_data[i][3] = 1;
            }
        }
        if (readWrite == 'r')
        {
            read_counter++;
        }
        else
        {
            write_counter++;
        }
        input_data[i][2] = (double)read_counter / (double)write_counter;
        for (int target_lb_counter = i - 1; target_lb_counter >= 0; target_lb_counter--)
        {
            if (input_data[i][0] == input_data[target_lb_counter][0])
            {
                double target_difference = i - target_lb_counter;
                if (target_difference < 1000)
                {
                    target_data[i] = 1;
                }
                else if (target_difference < 10000)
                {
                    target_data[i] = 2;
                }
                else if (target_difference < 100000)
                {
                    target_data[i] = 3;
                }
                else
                {
                    target_data[i] = 4;
                }
                target_data[i] = i - target_lb_counter;
                break;
            }
        }
        printf("%d\n", i);
    }

    fclose(file);

    // // Sample input data (random example, replace with your actual data)
    // double input_data[4][INPUT_SIZE] = {
    //     {3.0, 1.0, 0.0, 0.0, 1.0, 1.0}, // rw_rat, is_seq, chunk_w, chunk_r, etc.
    //     {5.0, 1.0, 2.0, 1.0, 1.0, 0.0},
    //     {7.0, 1.0, 1.0, 1.0, 0.0, 1.0},
    //     {9.0, 0.0, 0.0, 0.0, 0.0, 1.0}};

    // Sample target data (the life_t we want to predict)
    // double target_data[4] = {14.0, 13.0, 12.0, 11.0};

    // Train the GRU model
    train_gru(&gru, input_data, target_data, dataset_size);

    evaluate_model(&gru, input_data, target_data, dataset_size);

    // Free dynamically allocated memory
    // for (int i = 0; i < dataset_size; i++)
    // {
    //     free(input_data[i]);
    // }
    free(input_data);
    free(target_data);

    return 0;
}

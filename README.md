# GRU Model for Adaptive Lifetime Prediction of Pages
This project implements a Gated Recurrent Unit (GRU) neural network to predict the lifetime of pages written by a user. The goal is to perform data separation based on adaptive lifetime prediction. The GRU model processes time-series data to estimate how long specific pages will be "alive" based on features such as read/write ratio, sequence patterns, and chunk information.

## Table of Contents
  - Introduction
  - Requirements
  - Dataset Structure
  - Usage
  - How the Model Works
  - Training Process
  - References
## Introduction
This project trains a GRU neural network to predict the lifetime of pages written by users to optimize data management and storage. The model uses features such as read-write ratios and sequence information to adaptively predict the lifetime and separate data based on its behavior.

The GRU model operates over a time-series input where each data point is characterized by:

    1. Read-write ratio
    2. Sequential page access
    3. Chunk write/read information
The goal is to predict the lifetime of the pages using these features.

## Requirements
Before running the project, ensure you have the following software installed:

    1. GCC Compiler (or any C compiler)
    2. C Standard Library
    3. Make (optional)
To compile the code, you can use:
    
```bash
gcc -o gru_model main.c -lm
```
To run the compiled program:
```bash
./gru_model
```

## Dataset Structure
The dataset is read from a .log file (FIO_test.log) containing the page access records. The file must have the following structure:

```php
<time_stamp> <block_address> <r/w> <chunk_w> <chunk_r>
```
  - time_stamp: Time of the access event.
  - block_address: The block/page being accessed.
  - r/w: A character ('r' or 'w') representing a read or write operation.
  - chunk_w: The amount of data written in chunks.
  - chunk_r: The amount of data read in chunks.

Each line represents a page access event. This data will be fed into the GRU model for training and prediction.

## Usage
Prepare the Dataset: Place the dataset log file (FIO_test.log) in the Data/NewData/ directory (or adjust the path in the code as needed).

Compile and Run:

Compile the C code using a C compiler, such as GCC:

```bash 
gcc -o gru_model main.c -lm
```
Run the model:
    
```bash
./gru_model
```
## Data Separation:

  - The GRU model will process the dataset, and the program will predict the "lifetime" for each page based on the access patterns.
  - The output will include information about the loss during training and predictions for each page.

## How the Model Works
The GRU model operates in the following steps:

### 1. Initialization:

  - The model initializes random weights for the GRU gates (update, reset, and candidate hidden state).
  - The hidden state is initialized to zero.
### 2. Forward Pass:

  - For each data point, the GRU processes the input features to predict the page's lifetime.
  - The input features include:
    - rw_rat: Read/write ratio.
    - is_seq: Whether the page access is sequential.
    - chunk_w: Number of write chunks.
    - chunk_r: Number of read chunks.

### 3. Backpropagation:

  - The model computes the loss (Mean Squared Error) between the predicted and actual page lifetimes.
  - Gradients are computed, and the weights are updated using gradient descent to minimize the loss.
### 4. Output:

  - The model outputs the predicted lifetime of pages for each data point.

## Training Process
The training process is controlled by two main hyperparameters:

  - EPOCHS: Number of training epochs (set to 1000 by default).
  - LEARNING_RATE: The learning rate for gradient descent (set to 0.001 by default).

During training:

  - The model reads input data and target values from the dataset file.
  - It performs forward passes, computes the loss, and applies backpropagation to update the model weights.
  - The total loss for each epoch is printed to the console.

### Example Output
```yaml
Epoch 0, Data 0, Loss: 0.1523
Epoch 0, Data 1, Loss: 0.0489
Epoch 0, Total Loss: 0.0721
...
Epoch 999, Data 99, Loss: 0.0007
Epoch 999, Total Loss: 0.0021
```
### Predicted Lifetime:
The predicted lifetimes are based on the difference between consecutive accesses to the same block/page and are categorized into:

   1. Less than 1000 accesses.
   2. Less than 10,000 accesses.
   3. Less than 100,000 accesses.
   4. Greater than 100,000 accesses.

## References
  - GRU (Gated Recurrent Units): A type of recurrent neural network (RNN) used for time-series data that adapts through gates to predict the next state and output.
  - Mean Squared Error (MSE): The loss function used to measure the difference between the predicted and actual lifetime of pages.
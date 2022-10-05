#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

#include "simpleserial.h"
#include "hal.h"

#define DEBUG_MODE 0 // when 1, prints debug info to stdout

// Set Global vars
#define EPOCHS 5
#define BATCH_SIZE 10
#define NUM_FEATURES 6
#define NUM_SAMPLES 713
#define NUM_HIDDEN_LAYERS 2    
#define NUM_HIDDEN_NODES  8
#define NUM_OUTPUT_NODES  1    
#define LEARNING_RATE 0.025
#define BIAS 0.75

#define NUM_TRAIN 570
#define NUM_TEST 143

#define RELU 0
#define SIGMOID 1
#define ACTIVATION_0 RELU    // First hidden layer   
#define ACTIVATION_1 SIGMOID // Second hidden layer 
#define ACTIVATION_2 RELU    // Output layer 

// Callbacks
// Function to read data from csv into 2D array over serial
uint8_t read_data(uint8_t* data, uint8_t len);

// Initialize edge weights
uint8_t init_weights(uint8_t* data, uint8_t len);

// Function to run one training epoch
uint8_t epoch(uint8_t* data, uint8_t len);

// Function to test one sample
uint8_t test_sample(uint8_t* data, uint8_t len);

// Activation functions and derivative of activation functions
double sigmoid(double x);
double sigmoid_prime(double x);
double relu(double x);
double relu_prime(double x);

// NeuralNet tools
double net(int node_index, int num_inputs, uint8_t inputs[num_inputs], double weights[][num_inputs], double bias_weight);
void feed_forward(uint8_t input_sample[NUM_FEATURES]);
void back_prop(double expected_output[NUM_OUTPUT_NODES], uint8_t sample[NUM_FEATURES]);
double calculate_loss(double expected_output[NUM_OUTPUT_NODES]);

// Matrix tools
void rand_to_1_1d(int size, double arr[size]);
void rand_to_1_2d(int rows, int cols, double arr[rows][cols]);
void permute_array(int array_size, int arr[array_size]);


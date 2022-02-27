#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h> 

#include "simpleserial.h"
#include "hal.h"

#define DEBUG_MODE 0 // when 1, prints debug info to stdout

//Set global vars
#define MAX_EPOCHS 100  
#define BATCH_SIZE 1
#define NUM_FEATURES 6
#define NUM_SAMPLES 713
#define NUM_HIDDEN_LAYERS 2    
#define NUM_HIDDEN_NODES 8
#define NUM_OUTPUT_NODES  1    
#define LEARNING_RATE 0.025
#define BIAS 0.75
#define MAX_NUM_MODEL 2

#define num_first_hidden1 6                  
#define num_second_hidden1 3  // base model 1

#define num_first_hidden2 2
#define num_second_hidden2 5 // base model 2

//#define num_output 1

#define MAX_THRESHOLD 1 // For weight initialization
#define MIN_THRESHOLD -1

#define NUM_TRAIN 535
#define NUM_TEST 178

// Function to read data from csv into 2D array over serial
uint8_t read_data(uint8_t* data, uint8_t len);

// Function to train model
uint8_t epoch(uint8_t* data, uint8_t len);

// Initialize weights
uint8_t init_weights1(uint8_t* data, uint8_t len);
uint8_t init_weights2(uint8_t* data, uint8_t len);

// Activation function and derivative of activation functions
double sigmoid(double x);
double sigmoid_prime(double x);
double relu(double x);
double relu_prime(double x);

// Neural Network tools
void generate_nodes(int arr1[NUM_HIDDEN_LAYERS],int arr2[NUM_HIDDEN_LAYERS]);  
double net(int node_index, int num_inputs, uint8_t inputs[num_inputs], double weights[][num_inputs], double bias_w);

void feed_forward(int num_first_hidden, int num_second_hidden,int num_output,
                  uint8_t input_sample[NUM_FEATURES],
                  double input_weights[num_first_hidden][NUM_FEATURES],
                  double hidden_weights[num_second_hidden][num_first_hidden],
                  double output_weights[num_output][num_second_hidden],
                  double bias_weights1[num_first_hidden],
                  double bias_weights2[num_second_hidden],
                  double hidden_layers_first[num_first_hidden][NUM_HIDDEN_LAYERS],
                  double hidden_layers_second[num_second_hidden][NUM_HIDDEN_LAYERS],
                  double hidden_nets_first[num_first_hidden][NUM_HIDDEN_LAYERS],
                  double hidden_nets_second[num_second_hidden][NUM_HIDDEN_LAYERS],
                  double output_layer[num_output],
                  double output_nets[num_output]);

void back_prop(int num_first_hidden, int num_second_hidden, int num_output,
               double expected_output[num_output], 
               uint8_t sample[NUM_FEATURES],
               double input_weights[num_first_hidden][NUM_FEATURES],
               double hidden_weights[num_second_hidden][num_first_hidden],
               double output_weights[num_output][num_second_hidden],
               double bias_weights1[num_first_hidden],
               double bias_weights2[num_second_hidden],
               double hidden_layers_first[num_first_hidden][NUM_HIDDEN_LAYERS],
               double hidden_layers_second[num_second_hidden][NUM_HIDDEN_LAYERS],
               double hidden_nets_first[num_first_hidden][NUM_HIDDEN_LAYERS],
               double hidden_nets_second[num_second_hidden][NUM_HIDDEN_LAYERS],
               double output_layer[num_output],
               double output_nets[num_output]);

// Matrix tools
void permute_array(int array_size, int arr[array_size], int index);
void rand_to_1_1d(int size, double arr[size]);
void rand_to_1_2d(int rows, int cols, double min, double max, double arr[rows][cols]);
void print_2d_doubles(int num_rows, int num_cols, double a[num_rows][num_cols]);
void print_one_doubles(int size, double a[size]);

//Function to generate a random integer between min and max
int random_number(int min, int max);
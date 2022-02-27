#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>
#include <assert.h>
#include <time.h>

#define DEBUG_MODE 0 // when 1, prints debug info to stdout

// Set Global vars
#define MAX_EPOCHS 100 //hypaperameter
#define BATCH_SIZE 10  //hypaperameter
#define NUM_FEATURES 6
#define NUM_SAMPLES 713
#define NUM_HIDDEN_LAYERS 2    
#define NUM_HIDDEN_NODES 10 //10 or 8  //hypaperameter
#define NUM_OUTPUT_NODES  1   
#define NUM_CLASSES  2
#define LEARNING_RATE 0.025  //hypaperameter
#define BIAS 0.75  //hypaperameter
#define MAX_NUM_MODEL 2

#define MAX_THRESHOLD 5 // For weight initialization  //hypaperameter
#define MIN_THRESHOLD -5  //hypaperameter

#define NUM_TRAIN 535 //75%
#define NUM_TEST 178  //25%


// Function to read data from csv into 2D array over serial
void read_data();

// Function to split data into training/testing data/labels
void split_data(double training_data[NUM_TRAIN][NUM_FEATURES],
                double testing_data[NUM_TEST][NUM_FEATURES],
                double training_labels[NUM_TRAIN],
                double testing_labels[NUM_TEST]);

//Function to print data
void print_data(double training_data[NUM_TRAIN][NUM_FEATURES],
                double testing_data[NUM_TEST][NUM_FEATURES],
                double training_labels[NUM_TRAIN],
                double testing_labels[NUM_TEST]);

// Function to initialize weights
double init_weights(int num_first_hidden, int num_second_hidden, int num_output,
                    double input_weights[num_first_hidden][NUM_FEATURES],
                    double hidden_weights[num_second_hidden][num_first_hidden],
                    double output_weights[num_output][num_second_hidden],
                    double bias_weights1[num_first_hidden],
                    double bias_weights2[num_second_hidden]);

// Train model one epoch
double epoch(int num_first_hidden, int num_second_hidden,int num_output,
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

// Activation function and derivative of activation functions
double sigmoid(double x);
double sigmoid_prime(double x);
double relu(double x);
double relu_prime(double x);

// Neural Network tools
int generate_nodes(int arr1[NUM_HIDDEN_LAYERS],int arr2[NUM_HIDDEN_LAYERS]);  
double net(int node_index, int num_inputs, double inputs[num_inputs], double weights[][num_inputs], double bias_w);

void feed_forward(int num_first_hidden, int num_second_hidden,int num_output,
                  double input_sample[NUM_FEATURES],
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
               double sample[NUM_FEATURES],
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
void permute_array(int array_size, int arr[array_size]);
void rand_to_1_1d(int size, double arr[size]);
void rand_to_1_2d(int rows, int cols, double min, double max, double arr[rows][cols]);
void print_2d_doubles(int num_rows, int num_cols, double a[num_rows][num_cols]);
void print_one_doubles(int size, double a[size]);

//Function to generate a random integer between min and max
int random_number(int min, int max);

//Function to calculate prediction accuracy
double accuracy(int size, double data[size][NUM_FEATURES], double labels[size],
                int num_first_hidden, int num_second_hidden,int num_output,
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
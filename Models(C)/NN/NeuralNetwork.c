#include "NeuralNetwork.h"

// Create Array Structures
double hidden_layers[NUM_HIDDEN_NODES][NUM_HIDDEN_LAYERS];
double hidden_layer_bias[NUM_HIDDEN_NODES][NUM_HIDDEN_LAYERS];
double hidden_nets[NUM_HIDDEN_NODES][NUM_HIDDEN_LAYERS];

double output_layer[NUM_OUTPUT_NODES];
double output_layer_bias[NUM_OUTPUT_NODES];
double output_nets[NUM_OUTPUT_NODES];

double input_weights[NUM_HIDDEN_NODES][NUM_FEATURES];      // weights[to][from]
double hidden_weights[NUM_HIDDEN_NODES][NUM_HIDDEN_NODES];
double output_weights[NUM_OUTPUT_NODES][NUM_HIDDEN_NODES];
double bias_weights[NUM_HIDDEN_NODES];


int read_count;
int correct;
int incorrect;
int partial;
uint8_t training_data[NUM_TRAIN][NUM_FEATURES];
uint8_t testing_data[NUM_TEST][NUM_FEATURES];
uint8_t training_labels[NUM_TRAIN];
uint8_t testing_labels[NUM_TEST];

int main(void) {
    // Set seed for repeatable results
    srand(999);
    
    // Initialize counters
    read_count = 0;
    correct = 0;
    incorrect = 0;
    partial = 0;
    
    // Initialize CW stuff and things
    platform_init();
    init_uart();
    trigger_setup();
    simpleserial_init();
    
    // Add commands
    simpleserial_addcmd('d', NUM_FEATURES+1, read_data);  
    simpleserial_addcmd('i', 1, init_weights);
    simpleserial_addcmd('e', 1, epoch);
    simpleserial_addcmd('t', 1, test_sample);
    
    while(1) {
        simpleserial_get();
    }
}

uint8_t read_data(uint8_t* data, uint8_t len) {
    // Test to see if you are filling training or testing data
    if(read_count < NUM_TRAIN) {
        // Store features
        for(uint8_t f = 0; f < NUM_FEATURES; f++) {
            uint8_t feature = 0.0;
            memcpy(&feature, &data[f], sizeof(uint8_t));
            training_data[read_count][f] = feature;
        }
        
        // Store label
        uint8_t feature = 0.0;
        memcpy(&feature, &data[NUM_FEATURES], sizeof(uint8_t));
        training_labels[read_count] = feature;
        
        // Return confirmation back to notebook
        simpleserial_put('r', NUM_FEATURES, (uint8_t*) &training_data[read_count]);
    } else if (read_count < NUM_SAMPLES) {
        // Store features
        for(uint8_t f = 0; f < NUM_FEATURES; f++) {
            uint8_t feature = 0.0;
            memcpy(&feature, &data[f], sizeof(uint8_t));
            testing_data[read_count-NUM_TRAIN][f] = feature;
        }
        
        // Store label
        uint8_t feature = 0.0;
        memcpy(&feature, &data[NUM_FEATURES], sizeof(uint8_t));
        testing_labels[read_count-NUM_TRAIN] = feature;
        
        //Return confirmation back to notebook
        simpleserial_put('r', NUM_FEATURES, (uint8_t*) &testing_data[read_count-NUM_TRAIN]);
    }
    read_count++;
    return 0x00;
}

uint8_t init_weights(uint8_t* data, uint8_t len) {
    // Initialize Weights
    rand_to_1_2d(NUM_HIDDEN_NODES, NUM_FEATURES, input_weights);
    rand_to_1_2d(NUM_HIDDEN_NODES, NUM_HIDDEN_NODES, hidden_weights);
    rand_to_1_2d(NUM_OUTPUT_NODES, NUM_HIDDEN_NODES, output_weights);
    rand_to_1_1d(NUM_HIDDEN_NODES, bias_weights); 
    read_count++;
    simpleserial_put('r', 2, (uint8_t*) &read_count);
    return 0x00;
}

uint8_t epoch(uint8_t* data, uint8_t len) {

    // Randomize training data order
    int indices[NUM_TRAIN];
    for(int i = 0; i<NUM_TRAIN; i++) {
        indices[i] = i;
    }
    permute_array(NUM_TRAIN, indices);

    trigger_high();
    // For each training sample
    for (int s = 0; s < BATCH_SIZE; s++) {
        // Get Sample using random index
        uint8_t sample[NUM_FEATURES];
        for(int f = 0; f < NUM_FEATURES; f++) {
            sample[f] = training_data[indices[s]][f];
        }
        feed_forward(sample);
        // Cacluclate loss (to track learning)
        double expected_output[NUM_OUTPUT_NODES];
        for(int o = 0; o < NUM_OUTPUT_NODES; o++) {
            expected_output[o] = training_labels[indices[s]]; // Assumes one output, otherwise = training_labels[indices[s]][o];
        }
        back_prop(expected_output, sample);
    }  
    trigger_low();
    //Return confirmation back to notebook
    read_count++;
    simpleserial_put('r', 2, (uint8_t*) &read_count);
    return 0x00; 
}

uint8_t test_sample(uint8_t* index, uint8_t len) {
    trigger_high();
    int s = (int) index;
    
    // Get Sample and expected result
    uint8_t sample[NUM_FEATURES];
    for(int f = 0; f < NUM_FEATURES; f++) {
        sample[f] = testing_data[s][f];
    }
    double expected_output[NUM_OUTPUT_NODES];
    for(int o = 0; o < NUM_OUTPUT_NODES; o++) {
        expected_output[o] = testing_labels[s]; // Assumes one output, otherwise = testing_labels[s][o];
    }

    // Feed forward function
    feed_forward(sample);

    // Record accuracy
    int out_correct = 0;
    int out_incorrect = 0;
    double cutoff = 0;
    if(ACTIVATION_2 == RELU) { // Activation function on output nodes == RELU
        cutoff = 0.5;
    }
    for(int o = 0; o < NUM_OUTPUT_NODES; o++) {
        if((output_layer[o] > cutoff && expected_output[o] == 1) || (output_layer[o] <= cutoff && expected_output[o] == 0) ) {
            out_correct++;
        } else {
            out_incorrect++;
        }
    }
    if(out_correct == NUM_OUTPUT_NODES) {
        correct++;
    } else if (out_correct > out_incorrect) {
        partial++;
    } else {
        incorrect++;
    }  
    trigger_low();
    read_count++;
    simpleserial_put('r', 2, (uint8_t*) &read_count);
    return 0x00;  
}


// Activation functions
double sigmoid(double x) { 
    return 1 / (1 + exp(-x)); 
}
double relu(double x) {
    if (x > 0) {
        return x;
    } else {
        return 0;
    }
}

// Derivative of activation functions
double sigmoid_prime(double x) { 
    return sigmoid(x) * (1 - sigmoid(x)); 
}
double relu_prime(double x) {
    if (x > 0) {
        return 1;
    } else {
        return 0;
    }
}

// Neural Network Tools
// Weighted sum from input layer to current layer  (num inputs does not include bias)
double net(int node_index, int num_inputs, uint8_t inputs[num_inputs], double weights[][num_inputs], double bias_weight){
    double sum = 0.0;
    for(int i = 0; i < num_inputs; i++) {
        sum += inputs[i] * weights[node_index][i];
    }
    sum += BIAS*bias_weight;
    return sum;
}

// Feed sample through NN (saves results in output_layer and hidden_layers)
void feed_forward(uint8_t input_sample[NUM_FEATURES]) {
    // Get nets (weighted avgs) into first hidden layer
    double nets[NUM_HIDDEN_NODES];
    for(int i = 0; i < NUM_HIDDEN_NODES; i++) {
        nets[i] = net(i, NUM_FEATURES+1, input_sample, input_weights, bias_weights[i]);
        hidden_nets[i][0] = nets[i]; // Save the nets for back prop later on
    }

    // Apply sigmoid function in each hidden node in first layer
    uint8_t layer_output[NUM_HIDDEN_NODES];
    for(int n = 0; n < NUM_HIDDEN_NODES; n++) {
        if(ACTIVATION_0 == RELU) {
            layer_output[n] = relu(nets[n]);
        } else {
            layer_output[n] = sigmoid(nets[n]);
        }
        hidden_layers[n][0] = layer_output[n];
    }

    // Propogate through each hidden layer
    for (int l = 1; l < NUM_HIDDEN_LAYERS; l++) {
        // Get values for next hidden layer
        for(int i = 0; i < NUM_HIDDEN_NODES; i++) {
            nets[i] = net(i, NUM_HIDDEN_NODES, layer_output, hidden_weights, 0);
            hidden_nets[i][l] = nets[i]; // Save the nets for back prop later on
        }

        // Apply sigmoid function in each hidden node in this layer
        for(int n = 0; n < NUM_HIDDEN_NODES; n++) {
            if(ACTIVATION_1 == RELU) {
                layer_output[n] = relu(nets[n]);
            } else {
                layer_output[n] = sigmoid(nets[n]);
            }
            hidden_layers[n][l] = layer_output[n]; // Save layer outputs for back prop later on
        }
    }

    // Get values going to output layer
    for(int i = 0; i < NUM_OUTPUT_NODES; i++) {
        output_nets[i] = net(i, NUM_HIDDEN_NODES, layer_output, output_weights, 0);
    }

    // Apply sigmoid function in output nodes
    for (int i = 0; i < NUM_OUTPUT_NODES; i++) {
        if(ACTIVATION_2 == RELU) {
            output_layer[i] = relu(output_nets[i]);
        } else {
            output_layer[i] = sigmoid(output_nets[i]);
        }
    }
}

// Update Weights based on expected and given output
void back_prop(double expected_output[NUM_OUTPUT_NODES], uint8_t sample[NUM_FEATURES]) {
    // Get sensitivity for each output node
    double sensitivity[NUM_OUTPUT_NODES];
    for(int o = 0; o < NUM_OUTPUT_NODES; o++) {
        double diff = expected_output[o] - output_layer[o];
        double out_prime;
        if(ACTIVATION_2 == RELU) {
            out_prime = relu_prime(output_nets[o]);
        } else {
           out_prime = sigmoid_prime(output_nets[o]);
        }
        sensitivity[o] = diff*out_prime;
    }
    // Find change in weights from hidden layer to output layer
    double delta_output_weights[NUM_OUTPUT_NODES][NUM_HIDDEN_NODES];
    for(int o = 0; o < NUM_OUTPUT_NODES; o++) {
        for(int h = 0; h < NUM_HIDDEN_NODES; h++) {
            delta_output_weights[o][h] = LEARNING_RATE * sensitivity[o] * hidden_layers[h][NUM_HIDDEN_LAYERS-1];
        }
    }

    // Get sensitivity for each hidden node
    double sensitivity_h[NUM_HIDDEN_NODES];
    for(int h = 0; h < NUM_HIDDEN_NODES; h++) {
        double summed_sens = 0.0;
        for(int o = 0; o < NUM_OUTPUT_NODES; o++) {
            summed_sens += output_weights[o][h] * sensitivity[o];
        }
        double hidden_prime;
        if(ACTIVATION_1 == RELU) {
            hidden_prime = relu_prime(hidden_nets[h][1]);
        } else {
            hidden_prime = sigmoid_prime(hidden_nets[h][1]);
        }
        sensitivity_h[h] = summed_sens * hidden_prime;
    }
    // Find change in weights from first to second hidden layer 
    double delta_hidden_weights[NUM_HIDDEN_NODES][NUM_HIDDEN_NODES];
    for(int h_from = 0; h_from < NUM_HIDDEN_NODES; h_from++) {
        for(int h_to; h_to < NUM_HIDDEN_NODES; h_to++) {
            delta_hidden_weights[h_to][h_from] = LEARNING_RATE * sensitivity_h[h_to] * hidden_layers[h_from][0];
        }
    }
    
    // Get sensitivity for each input node
    double sensitivity_i[NUM_HIDDEN_NODES];
    for(int h_from = 0; h_from < NUM_HIDDEN_NODES; h_from++) {
        double summed_sens = 0.0;
        for(int h_to = 0; h_to < NUM_HIDDEN_NODES; h_to++) {
            summed_sens += hidden_weights[h_to][h_from] * sensitivity_h[h_to];
        }
        double input_prime;
        if(ACTIVATION_0 == RELU) {
            input_prime = relu_prime(hidden_nets[h_from][0]);
        } else {
            input_prime = sigmoid_prime(hidden_nets[h_from][0]);
        }
        sensitivity_i[h_from] = summed_sens * input_prime;
    }
    // Find change in weights from first hidden layer to input layer
    double delta_input_weights[NUM_HIDDEN_NODES][NUM_FEATURES];
    double delta_bias_weights[NUM_HIDDEN_NODES];
    for(int h = 0; h < NUM_HIDDEN_NODES; h++) {
        for(int i = 0; i < NUM_FEATURES; i++) {
            delta_input_weights[h][i] = LEARNING_RATE * sensitivity_i[h] * sample[i]; 
        }
        delta_bias_weights[h] = LEARNING_RATE * sensitivity_i[h] * BIAS;
    }

    // Update output weights
    for(int o = 0; o < NUM_OUTPUT_NODES; o++) {
        for(int h = 0; h < NUM_HIDDEN_NODES; h++) {
            output_weights[o][h] += delta_output_weights[o][h];
        }
    }
    // Update hidden weights
    for(int h_to = 0; h_to < NUM_HIDDEN_NODES; h_to++) {
        for(int h_from; h_from < NUM_HIDDEN_NODES; h_from++) {
            hidden_weights[h_to][h_from] += delta_hidden_weights[h_to][h_from];
        }
    }
    // Update input weights
    for(int h = 0; h < NUM_HIDDEN_NODES; h++) {
        for(int i = 0; i < NUM_FEATURES; i++) {
            input_weights[h][i] += delta_input_weights[h][i];
        }
    }
    // Update bias weights
    for(int h = 0; h < NUM_HIDDEN_NODES; h++) {
        bias_weights[h] += delta_bias_weights[h];
    }      

}

// Calculates training error 
double calculate_loss(double expected_output[NUM_OUTPUT_NODES]) {
    double loss = 0.0;
    for(int i = 0; i < NUM_OUTPUT_NODES; i++) {
        loss += pow(expected_output[i] - output_layer[i], 2);
    }
    return loss / 2;
}

// Ceate a 1D array with random values summing to 1
void rand_to_1_1d(int size, double arr[size]) {
    //srand(time(NULL));
    int currentSum = 0;
    for(int i = 0; i < size; i++) {
        arr[i] = rand() % 100;
        currentSum += arr[i];
    }
    for(int i = 0; i < size; i++) {
        arr[i] = arr[i] / currentSum;
    }
}

// Create a 2D array with random values summing to 1
void rand_to_1_2d(int rows, int cols, double arr[rows][cols]) {
    //srand(time(NULL));
    int currentSum = 0;
    for(int r = 0; r < rows; r++){
        for(int c = 0; c < cols; c++) {
            arr[r][c] = rand() % 100;
            currentSum += arr[r][c];
        }
    }
    for(int r = 0; r < rows; r++){
        for(int c = 0; c < cols; c++) {
            arr[r][c] = arr[r][c] / currentSum;
        }
    }
}

// Permute Array
void permute_array(int array_size, int arr[array_size]) {
    //srand(time(NULL));
    for(int i = array_size-1; i > 0; i--) {
        int swap_val = rand() % (i+1);
        double temp = arr[i];
        arr[i] = arr[swap_val];
        arr[swap_val] = temp;
    }
}

#include "NeuralNetwork.h"

// Split data
int read_count;
int seed;
uint8_t training_data[NUM_TRAIN][NUM_FEATURES];
uint8_t testing_data[NUM_TEST][NUM_FEATURES];
uint8_t training_labels[NUM_TRAIN];
uint8_t testing_labels[NUM_TEST];
int num_output = 1;
// Define base model structures
//int model_one_nodes[NUM_HIDDEN_LAYERS];  
//int model_second_nodes[NUM_HIDDEN_LAYERS];  

// Generate two base models
//generate_nodes(model_one_nodes,model_second_nodes); 
/*num_first_hidden1 = model_one_nodes[0];
num_second_hidden1 = model_one_nodes[1];
num_first_hidden2 = model_second_nodes[0];
num_second_hidden2 = model_second_nodes[1];*/
    
// Create array structures for base model 1
double hidden_layers_first1[num_first_hidden1][NUM_HIDDEN_LAYERS];
double hidden_layers_second1[num_second_hidden1][NUM_HIDDEN_LAYERS];
double output_layer1[NUM_OUTPUT_NODES];

double hidden_nets_first1[num_first_hidden1][NUM_HIDDEN_LAYERS];
double hidden_nets_second1[num_second_hidden1][NUM_HIDDEN_LAYERS];
double output_nets1[NUM_OUTPUT_NODES]; 
    
double input_weights1[num_first_hidden1][NUM_FEATURES];
double hidden_weights1[num_second_hidden1][num_first_hidden1];
double output_weights1[NUM_OUTPUT_NODES][num_second_hidden1];
double bias_weights11[num_first_hidden1];
double bias_weights21[num_second_hidden1];  

// Create array structures for base model 2
double hidden_layers_first2[num_first_hidden2][NUM_HIDDEN_LAYERS];
double hidden_layers_second2[num_second_hidden2][NUM_HIDDEN_LAYERS];
double output_layer2[NUM_OUTPUT_NODES];

double hidden_nets_first2[num_first_hidden2][NUM_HIDDEN_LAYERS];
double hidden_nets_second2[num_second_hidden2][NUM_HIDDEN_LAYERS];
double output_nets2[NUM_OUTPUT_NODES]; 
    
double input_weights2[num_first_hidden2][NUM_FEATURES];
double hidden_weights2[num_second_hidden2][num_first_hidden2];
double output_weights2[NUM_OUTPUT_NODES][num_second_hidden2];
double bias_weights12[num_first_hidden2];
double bias_weights22[num_second_hidden2]; 


int main(void) {
    srand(999); // Set seed for repeatable results
    read_count = 0; // Initialize counters
    seed = 1; 
    // Initialize CW stuff and things
    platform_init();
    init_uart();
    trigger_setup();
    simpleserial_init();
    
    // Add commands
    simpleserial_addcmd('d', NUM_FEATURES+1, read_data);  
    simpleserial_addcmd('i', 1, init_weights1);
    simpleserial_addcmd('k', 1, init_weights2);
    simpleserial_addcmd('e', 1, epoch); 
    
    while(1) {
        simpleserial_get();
    }
}

//Read data
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
  
uint8_t epoch(uint8_t* data, uint8_t len) {
    // Randomize training data order
    int indices[NUM_TRAIN];
    for(int i = 0; i<NUM_TRAIN; i++) {
        indices[i] = i;
    }
    seed++;
    permute_array(NUM_TRAIN, indices, seed);
    //Training
      for(int s = 0; s < BATCH_SIZE; s++){
        uint8_t sample[NUM_FEATURES];
        for(int f = 0; f < NUM_FEATURES; f++) {
           sample[f] = training_data[indices[s]][f];
         }
        double expected_output[NUM_OUTPUT_NODES];
        for(int o = 0; o < NUM_OUTPUT_NODES; o++) {
            expected_output[o] = training_labels[indices[s]];  
        }
        trigger_high(); // start power trace collection for each sample
        feed_forward(num_first_hidden1, num_second_hidden1, num_output, 
                     sample,input_weights1,hidden_weights1,output_weights1,
                     bias_weights11,bias_weights21,
                     hidden_layers_first1,hidden_layers_second1,hidden_nets_first1,hidden_nets_second1,
                     output_layer1,output_nets1); // Feed forward in base model 1
        back_prop(num_first_hidden1, num_second_hidden1, num_output, expected_output,
                  sample,input_weights1,hidden_weights1,output_weights1,
                  bias_weights11,bias_weights21,
                  hidden_layers_first1,hidden_layers_second1,hidden_nets_first1,hidden_nets_second1,
                  output_layer1,output_nets1);  // Back propogation in base model 1
        feed_forward(num_first_hidden2, num_second_hidden2, num_output,
                     sample,input_weights2,hidden_weights2,output_weights2,
                     bias_weights12,bias_weights22,
                     hidden_layers_first2,hidden_layers_second2,hidden_nets_first2,hidden_nets_second2,
                     output_layer2,output_nets2); // Feed forward in base model 2
        back_prop(num_first_hidden2, num_second_hidden2, num_output, expected_output, 
                  sample,input_weights2,hidden_weights2,output_weights2,
                  bias_weights12,bias_weights22,
                  hidden_layers_first2,hidden_layers_second2,hidden_nets_first2,hidden_nets_second2,
                  output_layer2,output_nets2);  // Back propogation in base model 2
        trigger_low();// stop power trace collection 
       }
    //Return confirmation back to notebook
    read_count++;
    simpleserial_put('r', 2, (uint8_t*) &read_count);
    return 0x00; 
    }

//Initialize weights for base model1
uint8_t init_weights1(uint8_t* data, uint8_t len) {
    rand_to_1_2d(num_first_hidden1, NUM_FEATURES, MIN_THRESHOLD, MAX_THRESHOLD, input_weights1);
    rand_to_1_2d(num_second_hidden1, num_first_hidden1, MIN_THRESHOLD, MAX_THRESHOLD, hidden_weights1);
    rand_to_1_2d(num_output, num_second_hidden1, MIN_THRESHOLD, MAX_THRESHOLD, output_weights1); 
    rand_to_1_1d(num_first_hidden1, bias_weights11); 
    rand_to_1_1d(num_second_hidden1, bias_weights21);  
    read_count++;
    simpleserial_put('r', 2, (uint8_t*) &read_count);
    return 0x00;
}

//Initialize weights for base model2
uint8_t init_weights2(uint8_t* data, uint8_t len) {
    rand_to_1_2d(num_first_hidden2, NUM_FEATURES, MIN_THRESHOLD, MAX_THRESHOLD, input_weights2);
    rand_to_1_2d(num_second_hidden2, num_first_hidden2, MIN_THRESHOLD, MAX_THRESHOLD, hidden_weights2);
    rand_to_1_2d(num_output, num_second_hidden2, MIN_THRESHOLD, MAX_THRESHOLD, output_weights2); 
    rand_to_1_1d(num_first_hidden2, bias_weights12); 
    rand_to_1_1d(num_second_hidden2, bias_weights22);  
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
// Weighted sum from input layer to current node 
double net(int node_index, int num_inputs, uint8_t inputs[num_inputs], double weights[][num_inputs], double bias_w){
    double sum = 0.0;
    for(int i = 0; i < num_inputs; i++) {
        sum += inputs[i] * weights[node_index][i];
    }
    sum += BIAS*bias_w;
    return sum;
}

// Feed-forward function
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
                  double output_nets[num_output]) {
    double nets1[num_first_hidden];
    for(int i = 0; i < num_first_hidden; i++) {
        nets1[i] = net(i, NUM_FEATURES, input_sample, input_weights, bias_weights1[i]);
        hidden_nets_first[i][0] = nets1[i]; // Save the nets for back prop
    }
    // Apply sigmoid function in each hidden node in first layer
    uint8_t layer_output[num_first_hidden];
    uint8_t layer_output2[num_second_hidden];
    for(int i = 0; i < num_first_hidden; i++) {
        layer_output[i] = sigmoid(nets1[i]);
        hidden_layers_first[i][0] = layer_output[i];
    }
    double nets2[num_second_hidden];
    // Propogate through each hidden layer
    for (int l = 1; l < NUM_HIDDEN_LAYERS; l++) {
        for(int i = 0; i < num_second_hidden; i++) {
            nets2[i] = net(i, num_first_hidden, layer_output, hidden_weights, bias_weights2[i]);
            hidden_nets_second[i][l] = nets2[i]; 
        }
    // Apply sigmoid function in each hidden node in second layer
        for(int i = 0; i < num_second_hidden; i++) {
            layer_output2[i] = sigmoid(nets2[i]);
            hidden_layers_second[i][l] = layer_output2[i]; 
        }
    }
    // Get values going to output layer
    for(int i = 0; i < num_output; i++) {
        output_nets[i] = net(i, num_second_hidden, layer_output2, output_weights, 0);
    }
    // Apply sigmoid function in output nodes
    for (int i = 0; i < num_output; i++) {
        output_layer[i] = sigmoid(output_nets[i]);
     }
}
    
// Back propogation function
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
               double output_nets[num_output]) {
    // Get sensitivity for each output node
    double sensitivity[num_output];
    for(int o = 0; o < num_output; o++) {
        double diff = expected_output[o] - output_layer[o];
        double out_prime;
        out_prime = sigmoid_prime(output_nets[o]);
        sensitivity[o] = diff*out_prime;
    }
    // Find change in weights from hidden layer to output layer
    double delta_output_weights[num_output][num_second_hidden];
    for(int o = 0; o < num_output; o++) {
        for(int h = 0; h < num_second_hidden; h++) {
            delta_output_weights[o][h] = LEARNING_RATE * sensitivity[o] * hidden_layers_second[h][NUM_HIDDEN_LAYERS-1];
        }
    }
    // Get sensitivity for each hidden node
    double sensitivity_h[num_second_hidden];
    for(int h = 0; h < num_second_hidden; h++) {
        double summed_sens = 0.0;
        for(int o = 0; o < num_output; o++) {
            summed_sens += output_weights[o][h] * sensitivity[o];
        }
        double hidden_prime;
        hidden_prime = sigmoid_prime(hidden_nets_second[h][1]);
        sensitivity_h[h] = summed_sens * hidden_prime;
    }
    // Find change in weights from first to second hidden layer 
    double delta_hidden_weights[num_second_hidden][num_first_hidden];
    double delta_bias_weights2[num_second_hidden];
    for(int h_to = 0; h_to < num_second_hidden; h_to++) {
        for(int h_from=0; h_from < num_first_hidden; h_from++) {
        delta_hidden_weights[h_to][h_from] = LEARNING_RATE * sensitivity_h[h_to] * hidden_layers_first[h_from][0];
        }
        delta_bias_weights2[h_to] = LEARNING_RATE * sensitivity_h[h_to] * BIAS;
    }
    // Get sensitivity for each input node
    double sensitivity_i[num_first_hidden];
    for(int h_from = 0; h_from < num_first_hidden; h_from++) {
        double summed_sens = 0.0;
        for(int h_to = 0; h_to < num_second_hidden; h_to++) {
            summed_sens += hidden_weights[h_to][h_from] * sensitivity_h[h_to];
        }
        double input_prime;
        input_prime = sigmoid_prime(hidden_nets_first[h_from][0]);
        sensitivity_i[h_from] = summed_sens * input_prime;
        
    }
    // Find change in weights from first hidden layer to input layer
    double delta_input_weights[num_first_hidden][NUM_FEATURES];
    double delta_bias_weights1[num_first_hidden];
    for(int h = 0; h < num_first_hidden; h++) {
        for(int i = 0; i < NUM_FEATURES; i++) {
        delta_input_weights[h][i] = LEARNING_RATE * sensitivity_i[h] * sample[i]; 
        }
        delta_bias_weights1[h] = LEARNING_RATE * sensitivity_i[h] * BIAS;
    }
    // Update output weights
    for(int o = 0; o < num_output; o++) {
        for(int h = 0; h < num_second_hidden; h++) {
            output_weights[o][h] += delta_output_weights[o][h];
        }
    }
    // Update hidden weights
    for(int h_to = 0; h_to < num_second_hidden; h_to++) {
        for(int h_from = 0; h_from < num_first_hidden; h_from++) {
            hidden_weights[h_to][h_from] += delta_hidden_weights[h_to][h_from];
        }
    }
    // Update input weights
    for(int h = 0; h < num_first_hidden; h++) {
        for(int i = 0; i < NUM_FEATURES; i++) {
            input_weights[h][i] += delta_input_weights[h][i];
        }
    }
    // Update bias weights
    for(int h = 0; h < num_first_hidden; h++) {
        bias_weights1[h] += delta_bias_weights1[h];
    } 
    for(int h = 0; h < num_second_hidden; h++) {
        bias_weights2[h] += delta_bias_weights2[h];
    }      
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

// Create a 2D array with random values from min to max
void rand_to_1_2d(int rows, int cols, double min, double max, double arr[rows][cols]) {
    double range = (max - min); 
    double div = RAND_MAX / range;
    //srand(time(NULL));
    for(int r = 0; r < rows; r++){
        for(int c = 0; c < cols; c++) {
            arr[r][c] = min + (rand() / div);
        }
    }
}
// Permute Array
void permute_array(int array_size, int arr[array_size], int index) {
    srand(index);
    for(int i = array_size-1; i > 0; i--) {
        int swap_val = rand() % (i+1);
        double temp = arr[i];
        arr[i] = arr[swap_val];
        arr[swap_val] = temp;
    }
}
// Randomly select nodes at each hidden layer for each base model
void generate_nodes(int arr1[NUM_HIDDEN_LAYERS],int arr2[NUM_HIDDEN_LAYERS]){
    //srand(time(NULL));
    for(int i=0; i<NUM_HIDDEN_LAYERS; i++){
        arr1[i] = random_number(2, NUM_HIDDEN_NODES-2);
        arr2[i] = NUM_HIDDEN_NODES - arr1[i];
    }
}

//Generate a random number in a certain range  
int random_number(int min, int max){
    int cur_num = 0;
    cur_num = min + rand( ) % (max-min)+1;
    return cur_num;
   }
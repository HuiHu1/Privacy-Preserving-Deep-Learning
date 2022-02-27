#include "NeuralNetwork.h"

char* FILE_NAME = "pima-indians-diabetes.csv"; // COMPAS

//Split training and testing
double training_data[NUM_TRAIN][NUM_FEATURES];
double testing_data[NUM_TEST][NUM_FEATURES];
double training_labels[NUM_TRAIN];
double testing_labels[NUM_TEST];

double full_data[NUM_SAMPLES][NUM_FEATURES+1];

// Define base model structures
int model_one_nodes[NUM_HIDDEN_LAYERS];  
int model_second_nodes[NUM_HIDDEN_LAYERS];  
int num_first_hidden1;  
int num_second_hidden1;  
int num_first_hidden2;  
int num_second_hidden2;  
int num_output = NUM_OUTPUT_NODES;  

/*-------------------------main start----------------------*/
int main(void) {
    read_data();  
    split_data(training_data,testing_data,training_labels,testing_labels);
     
    num_first_hidden1 = 5; 
    num_second_hidden1 = 10; //sub-network1

    num_first_hidden2 = 3;
    num_second_hidden2 = 7; //sub-network2
   
    // Create array structures for base model 1
    double hidden_layers_first1[num_first_hidden1][NUM_HIDDEN_LAYERS];
    double hidden_layers_second1[num_second_hidden1][NUM_HIDDEN_LAYERS];
    double output_layer1[num_output];

    double hidden_nets_first1[num_first_hidden1][NUM_HIDDEN_LAYERS];
    double hidden_nets_second1[num_second_hidden1][NUM_HIDDEN_LAYERS];
    double output_nets1[num_output]; 
    
    double input_weights1[num_first_hidden1][NUM_FEATURES];
    double hidden_weights1[num_second_hidden1][num_first_hidden1];
    double output_weights1[num_output][num_second_hidden1];
    double bias_weights11[num_first_hidden1];
    double bias_weights21[num_second_hidden1];  

    // Create array structures for base model 2
    double hidden_layers_first2[num_first_hidden2][NUM_HIDDEN_LAYERS];
    double hidden_layers_second2[num_second_hidden2][NUM_HIDDEN_LAYERS];
    double output_layer2[num_output];

    double hidden_nets_first2[num_first_hidden2][NUM_HIDDEN_LAYERS];
    double hidden_nets_second2[num_second_hidden2][NUM_HIDDEN_LAYERS];
    double output_nets2[num_output]; 
    
    double input_weights2[num_first_hidden2][NUM_FEATURES];
    double hidden_weights2[num_second_hidden2][num_first_hidden2];
    double output_weights2[num_output][num_second_hidden2];
    double bias_weights12[num_first_hidden2];
    double bias_weights22[num_second_hidden2];  

    // Initialize base model 1
    init_weights(num_first_hidden1,num_second_hidden1,num_output,
                    input_weights1,hidden_weights1,output_weights1,
                    bias_weights11,bias_weights21);

    // Initialize base model 2
    init_weights(num_first_hidden2,num_second_hidden2,num_output,
                    input_weights2,hidden_weights2,output_weights2,
                    bias_weights12,bias_weights22);        
    //Training
    for (int i=0; i < MAX_EPOCHS; i++){
        epoch(num_first_hidden1, num_second_hidden1,num_output,
                 input_weights1,hidden_weights1,output_weights1,
                 bias_weights11,bias_weights21,
                 hidden_layers_first1,hidden_layers_second1,hidden_nets_first1,hidden_nets_second1,
                 output_layer1,output_nets1);
    }
    
    //Training
    for (int i=0; i < MAX_EPOCHS; i++){
        epoch(num_first_hidden2, num_second_hidden2,num_output,
                 input_weights2,hidden_weights2,output_weights2,
                 bias_weights12,bias_weights22,
                 hidden_layers_first2,hidden_layers_second2,hidden_nets_first2,hidden_nets_second2,
                 output_layer2,output_nets2);
    }

    //Output training accuracy of each base model
    double train_accuracy1 = 0;
    double train_accuracy2 = 0;
    double test_accuracy = 0;
    train_accuracy1 = accuracy(NUM_TRAIN, training_data,training_labels,
                            num_first_hidden1,num_second_hidden1,num_output,
                            input_weights1,hidden_weights1,output_weights1,
                            bias_weights11,bias_weights21,
                            hidden_layers_first1,hidden_layers_second1,hidden_nets_first1,hidden_nets_second1,
                            output_layer1,output_nets1);
    train_accuracy2 = accuracy(NUM_TRAIN, training_data,training_labels,
                            num_first_hidden2,num_second_hidden2,num_output,
                            input_weights2,hidden_weights2,output_weights2,
                            bias_weights12,bias_weights22,
                            hidden_layers_first2,hidden_layers_second2,hidden_nets_first2,hidden_nets_second2,
                            output_layer2,output_nets2);
        
    //Select the best base model to predict on testing data
    if(train_accuracy1>=train_accuracy2) {
        test_accuracy = accuracy(NUM_TEST, testing_data,testing_labels,
                                num_first_hidden1,num_second_hidden1,num_output,
                                input_weights1,hidden_weights1,output_weights1,
                                bias_weights11,bias_weights21,
                                hidden_layers_first1,hidden_layers_second1,hidden_nets_first1,hidden_nets_second1,
                                output_layer1,output_nets1); 
        }
    else{
        test_accuracy = accuracy(NUM_TEST, testing_data,testing_labels,
                                num_first_hidden2,num_second_hidden2,num_output,
                                input_weights2,hidden_weights2,output_weights2,
                                bias_weights12,bias_weights22,
                                hidden_layers_first2,hidden_layers_second2,hidden_nets_first2,hidden_nets_second2,
                                output_layer2,output_nets2); 

    }
    
    printf("The best test accuracy is:\n");
    printf(" %f ", test_accuracy);
    fprintf(stdout, "\n");
}/*-------------------------main end----------------------*/

//Calculate prediction accuracy
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
                double output_nets[num_output]){
    double correct = 0.0;
    double incorrect = 0.0;
    double threshold = 0.5;               
    for(int i = 0; i<size; i++){
        double sample[NUM_FEATURES];
        for(int j =0; j<NUM_FEATURES; j++){
            sample[j] = data[i][j];
        }
        feed_forward(num_first_hidden, num_second_hidden, num_output,
                     sample,input_weights,hidden_weights,output_weights,
                     bias_weights1,bias_weights2,
                     hidden_layers_first,hidden_layers_second,hidden_nets_first,hidden_nets_second,
                     output_layer,output_nets); 
        for (int r = 0; r < num_output; r++){
            if ((output_layer[r] > threshold && labels[i] == 1) || (output_layer[r] <= threshold && labels[i] == 0)){
                correct++;
            }
            else{
                incorrect++;
            }
        }
    }
    return correct/ (double)size;
}
                  
// Randomly select nodes at each hidden layer for each base model
int generate_nodes(int arr1[NUM_HIDDEN_LAYERS],int arr2[NUM_HIDDEN_LAYERS]){
    srand(time(NULL));
    for(int i=0; i<NUM_HIDDEN_LAYERS; i++){
        arr1[i] = random_number(2, NUM_HIDDEN_NODES-2);
        arr2[i] = NUM_HIDDEN_NODES - arr1[i];
    }
}

//Initialize weights for each base model
double init_weights(int num_first_hidden, int num_second_hidden, int num_output,
                    double input_weights[num_first_hidden][NUM_FEATURES],
                    double hidden_weights[num_second_hidden][num_first_hidden],
                    double output_weights[num_output][num_second_hidden],
                    double bias_weights1[num_first_hidden],
                    double bias_weights2[num_second_hidden]) {
    rand_to_1_2d(num_first_hidden, NUM_FEATURES, MIN_THRESHOLD, MAX_THRESHOLD, input_weights);
    rand_to_1_2d(num_second_hidden, num_first_hidden, MIN_THRESHOLD, MAX_THRESHOLD, hidden_weights);
    rand_to_1_2d(num_output, num_second_hidden, MIN_THRESHOLD, MAX_THRESHOLD, output_weights); 
    rand_to_1_1d(num_first_hidden, bias_weights1); 
    rand_to_1_1d(num_second_hidden, bias_weights2); 
}

// Train one epoch for each base model
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
            double output_nets[num_output]) {
    // Randomize training data order
    int indices[NUM_TRAIN];
    for(int i = 0; i<NUM_TRAIN; i++) {
        indices[i] = i;
    }
    permute_array(NUM_TRAIN, indices); 
    double sample[NUM_FEATURES];
    for (int s = 0; s < BATCH_SIZE; s++) {
        for(int f = 0; f < NUM_FEATURES; f++) {
            sample[f] = training_data[indices[s]][f];
        } 
        feed_forward(num_first_hidden, num_second_hidden, num_output, 
                     sample,input_weights,hidden_weights,output_weights,
                     bias_weights1,bias_weights2,
                     hidden_layers_first,hidden_layers_second,hidden_nets_first,hidden_nets_second,
                     output_layer,output_nets); 
        double expected_output[num_output];
        for(int o = 0; o < num_output; o++) {
            expected_output[o] = training_labels[indices[s]]; 
        }
        back_prop(num_first_hidden, num_second_hidden, num_output, expected_output, 
                  sample,input_weights,hidden_weights,output_weights,
                  bias_weights1,bias_weights2,
                  hidden_layers_first,hidden_layers_second,hidden_nets_first,hidden_nets_second,
                  output_layer,output_nets); 
    }  
}

// Feed-forward function
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
                  double output_nets[num_output]) {
    double nets1[num_first_hidden];
    for(int i = 0; i < num_first_hidden; i++) {
        nets1[i] = net(i, NUM_FEATURES, input_sample, input_weights, bias_weights1[i]);
        hidden_nets_first[i][0] = nets1[i]; // Save the nets for back prop
    }
    // Apply sigmoid function in each hidden node in first layer
    double layer_output[num_first_hidden];
    double layer_output2[num_second_hidden];
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

// Weighted sum from input layer to current node 
double net(int node_index, int num_inputs, double inputs[num_inputs], double weights[][num_inputs], double bias_w){
    double sum = 0.0;
    for(int i = 0; i < num_inputs; i++) {
        sum += inputs[i] * weights[node_index][i];
    }
    sum += BIAS*bias_w;
    return sum;
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

// Permute Array
void permute_array(int array_size, int arr[array_size]) {
    srand(time(NULL));
    for(int i = array_size-1; i > 0; i--) {
        int swap_val = rand() % (i+1);
        double temp = arr[i];
        arr[i] = arr[swap_val];
        arr[swap_val] = temp;
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

// Create a 2D array with random values summing to 1
/*void rand_to_1_2d(int rows, int cols, double arr[rows][cols]) {
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
}*/

//Generate a random number in a certain range
int random_number(int min, int max){
    int cur_num = 0;
    cur_num = min + rand( ) % (max-min)+1;
    return cur_num;
   }

//Functions to print a two dimentional array
void print_2d_doubles(int num_rows, int num_cols, double a[num_rows][num_cols]) {
    for(int i = 0; i < num_rows; i++) {
        fprintf(stdout, "%i:  ", i);
        for(int j = 0; j < num_cols; j++) {
            fprintf(stdout, "%f  ", a[i][j]);
        }
        fprintf(stdout, "\n");
    }
    fprintf(stdout, "\n");
} 

//Functions to print a one dimentional array
void print_one_doubles(int size, double a[size]){
    for(int i=0; i < size; i++){
        fprintf(stdout,"%f  ", a[i]);
    }
    fprintf(stdout, "\n");
}

/*Below are three functions for data read, split and print*/
// Function to read data from csv into 2D array
void read_data() {
    FILE *file = fopen(FILE_NAME, "r");
    char currentline[256];
    int row_count = 0;
    assert(file != NULL);

    while (fgets(currentline, sizeof(currentline), file)) {
       //fprintf(stdout, "got line: %s\n", currentline);
        char* s_ptr;
        int n = 0;

        for (s_ptr = strtok(currentline, ",\n"); s_ptr && n < NUM_FEATURES+1; s_ptr = strtok(NULL, ",\n"), n++) {
            full_data[row_count][n] = strtod(s_ptr, NULL);
            //fprintf(stdout, "full_data[%i][%i] = %f\n", row_count, n, full_data[row_count][n]);
        }
        //fprintf(stdout, "Done with row %i\n\n", row_count);
        row_count++;
    }
    fclose(file);
}

// Function to split data into training/testing data/labels 
void split_data(double training_data[NUM_TRAIN][NUM_FEATURES],
                double testing_data[NUM_TEST][NUM_FEATURES],
                double training_labels[NUM_TRAIN],
                double testing_labels[NUM_TEST]) {
    for(int row = 0; row < NUM_TRAIN; row++) {
        int col = 0;
        for( ; col < NUM_FEATURES; col++) {
            training_data[row][col] = full_data[row][col];
        }
        training_labels[row] = full_data[row][col];
    }
    for(int row = 0; row < NUM_TEST; row++) {
        int col = 0;
        for( ; col < NUM_FEATURES; col++) {
            testing_data[row][col] = full_data[row + NUM_TRAIN][col];
        }
        testing_labels[row] = full_data[row + NUM_TRAIN][col];
    }
}

//Function to print data
void print_data(double training_data[NUM_TRAIN][NUM_FEATURES],
                double testing_data[NUM_TEST][NUM_FEATURES],
                double training_labels[NUM_TRAIN],
                double testing_labels[NUM_TEST]) {
    fprintf(stdout, "TRAINING DATA: \n");
    print_2d_doubles(NUM_TRAIN, NUM_FEATURES, training_data);

    fprintf(stdout, "TESTING DATA: \n");
    print_2d_doubles(NUM_TEST, NUM_FEATURES, testing_data);

    fprintf(stdout, "TRAINING LABELS: \n");
    for(int i = 0; i < NUM_TRAIN; i++) {
        fprintf(stdout, "%f  ", training_labels[i]);
    }

    fprintf(stdout, "TESTING LABELS: \n");
    for(int i = 0; i < NUM_TEST; i++) {
        fprintf(stdout, "%f  ", testing_labels[i]);
    }
    fprintf(stdout, "\n");
}
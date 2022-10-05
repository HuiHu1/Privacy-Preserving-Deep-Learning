#include "Matrix_uint.h"
#include "simpleserial.h"
#include "hal.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

#define NUM_SAMPLES  713
#define NUM_TRAIN 570
#define NUM_TEST 143

#define NUM_CLASSES  2
#define NUM_FEATURES 6
#define N 6

uint8_t eye[NUM_FEATURES][NUM_FEATURES];
uint8_t parameter[NUM_FEATURES];
uint8_t lamda=8.0;
int read_count;

uint8_t training_data[NUM_TRAIN][NUM_FEATURES];
uint8_t testing_data[NUM_TEST][NUM_FEATURES];
uint8_t training_labels[NUM_TRAIN];
uint8_t testing_labels[NUM_TEST];

uint8_t read_data(uint8_t* data, uint8_t len);
uint8_t init_eye(uint8_t* data, uint8_t len);
uint8_t epoch(uint8_t* data, uint8_t len);
uint8_t test_sample(uint8_t* data, uint8_t len);

int main(void){
    // Set seed for repeatable results
    srand(999);
    
    // Initialize counter
    read_count = 0;
    
    // Initialize CW stuff
    platform_init();
    init_uart();
    trigger_setup();
    simpleserial_init();
    
    // Add command to read data from serial
    simpleserial_addcmd('d', NUM_FEATURES+1, read_data);  
    simpleserial_addcmd('i', 1, init_eye);
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
        uint8_t feature = 0.0;
        memcpy(&feature, &data[NUM_FEATURES], sizeof(uint8_t));
        testing_labels[read_count-NUM_TRAIN] = feature;
        
        //Return confirmation back to notebook
        simpleserial_put('r', NUM_FEATURES, (uint8_t*) &testing_data[read_count-NUM_TRAIN]);
    }
    read_count++;
    return 0x00;
}

uint8_t init_eye(uint8_t* data, uint8_t len) {
    for (int i=0; i<N; i++)
        for (int j=0; j<N; j++)
        { if(i==j)
            eye[i][j]=1.0*lamda;
          else
            eye[i][j]=0.0;
        }
    
    read_count++;
    simpleserial_put('r', 2, (uint8_t*) &read_count);
    return 0x00;
}


uint8_t epoch(uint8_t* data, uint8_t len) {
    uint8_t transpose[NUM_FEATURES][NUM_TRAIN];
    uint8_t mult[NUM_FEATURES][NUM_FEATURES];
    uint8_t add[NUM_FEATURES][NUM_FEATURES];
    uint8_t inverse_matrix[N][N];
    uint8_t mult2[NUM_FEATURES][NUM_TRAIN];
    
    trigger_high();
    //Model parameter: (X^TX+lamda*I)^{-1}X^TY. lamda is hyperparameter.
    matrix_transpose(NUM_TRAIN,NUM_FEATURES,training_data,transpose);
    multiplyMatrices(NUM_FEATURES,NUM_TRAIN, NUM_TRAIN,NUM_FEATURES,transpose,training_data,mult);
    addMatrices(NUM_FEATURES,NUM_FEATURES,mult,eye,add);
    inverse(N,add,inverse_matrix);
    multiplyMatrices(NUM_FEATURES,NUM_FEATURES,NUM_FEATURES,NUM_TRAIN,inverse_matrix,transpose,mult2);
    multiply_matrice_vector(NUM_FEATURES,NUM_TRAIN,mult2,training_labels,parameter);
    //Return confirmation back to notebook
    trigger_low();
    read_count++;
    simpleserial_put('r', 2, (uint8_t*) &read_count);
    return 0x00; 
}

uint8_t test_sample(uint8_t* data, uint8_t len) {
    //Prediction on testing data
    uint8_t prediction[NUM_TEST];
    trigger_high();
    multiply_matrice_vector(NUM_TEST,NUM_FEATURES,testing_data,parameter,prediction);
    int count=0;
    for(int j = 0; j < NUM_TEST; ++j)
		{    
            if(prediction[j]>=0.5)
            {
              prediction[j]=1.0;  
            }
            else{
                prediction[j]=0.0; 
            }
            if(prediction[j]!=testing_labels[j]){
                count = count+1;
            }
        }
    //float error = (float)count/(float)num_test; //Testing error rate
    //fprintf(stdout, "%f  ", error);
    
    trigger_low();
    read_count++;
    simpleserial_put('r', 2, (uint8_t*) &read_count);
    return 0x00;  
}


/*
 * 34 layer INT8 Inference Engine
 * 
 * Usage:
 *   float input[MAX_INPUT_LENGTH];  // Preprocessed audio
 *   float output[2];                // [no_murmur_prob, murmur_prob]
 *   int input_length = 60000;       // Actual length (will need to tune)
 *   heart_inference(input, input_length, output);
 */

#include <stdint.h>
#include <string.h>
#include <math.h>
#include "dummy_weights.h"

// ---------------------------------------------
//  Memory buffers (have to adjust sizes based on our constraints idk how much streaming we can
// fo at once)

// Double buffering for layer outputs
#define MAX_CHANNELS 256
#define MAX_LENGTH 15000  // After initial pooling: 60000/4 = 15000

static float buffer_a[MAX_CHANNELS * MAX_LENGTH];
static float buffer_b[MAX_CHANNELS * MAX_LENGTH];

// ---------------------------------------------
//  Utility functions

static inline float relu(float x) {
    return x > 0.0f ? x : 0.0f;
}

static inline float max_f(float a, float b) {
    return a > b ? a : b;
}

// ---------------------------------------------
//  Perform Conv1D with INT8 weights (dequantize during operation)

static void conv1d_int8(
    const float* input,          // Input: [in_channels, in_length]
    float* output,               // Output: [out_channels, out_length]
    const int8_t* weight,        // Weight: [out_ch, in_ch, kernel]
    const float* weight_scale,   // Per-channel scales
    const float* bias,           // Bias: [out_ch]
    int in_channels,
    int out_channels,
    int in_length,
    int kernel_size,
    int padding,
    int stride,
    int apply_relu,
    int* out_length
) {
    int out_len = (in_length + 2 * padding - kernel_size) / stride + 1;
    *out_length = out_len;
    
    // For each output channel
    for (int oc = 0; oc < out_channels; oc++) {
        float scale = weight_scale[oc];
        
        // For each output position
        for (int ox = 0; ox < out_len; ox++) {
            float sum = bias[oc];
            int in_start = ox * stride - padding;
            
            // Convolution
            for (int ic = 0; ic < in_channels; ic++) {
                for (int k = 0; k < kernel_size; k++) {
                    int in_x = in_start + k;
                    
                    if (in_x >= 0 && in_x < in_length) {
                        int w_idx = oc * in_channels * kernel_size + ic * kernel_size + k;
                        float w_val = (float)weight[w_idx] * scale;  // Dequantize
                        float in_val = input[ic * in_length + in_x];
                        sum += w_val * in_val;
                    }
                }
            }
            
            // ReLU
            if (apply_relu) {
                sum = relu(sum);
            }
            
            output[oc * out_len + ox] = sum;
        }
    }
}

//  MaxPool1D

static void maxpool1d(
    const float* input,   // [channels, in_length]
    float* output,        // [channels, out_length]
    int channels,
    int in_length,
    int pool_size,
    int* out_length
) {
    int out_len = in_length / pool_size;
    *out_length = out_len;
    
    for (int c = 0; c < channels; c++) {
        for (int ox = 0; ox < out_len; ox++) {
            float max_val = -1e30f;
            for (int p = 0; p < pool_size; p++) {
                int in_x = ox * pool_size + p;
                if (in_x < in_length) {
                    float val = input[c * in_length + in_x];
                    max_val = max_f(max_val, val);
                }
            }
            output[c * out_len + ox] = max_val;
        }
    }
}

// ---------------------------------------------
//  Global Average Pooling

static void global_avg_pool1d(
    const float* input,   // [channels, length]
    float* output,        // [channels]
    int channels,
    int length
) {
    for (int c = 0; c < channels; c++) {
        float sum = 0.0f;
        for (int i = 0; i < length; i++) {
            sum += input[c * length + i];
        }
        output[c] = sum / (float)length;
    }
}

// ---------------------------------------------
//  Fully Connected with INT8 weights

static void linear_int8(
    const float* input,          // [in_features]
    float* output,               // [out_features]
    const int8_t* weight,        // [out_features, in_features]
    const float* weight_scale,   // Per-output scales
    const float* bias,           // [out_features]
    int in_features,
    int out_features
) {
    for (int o = 0; o < out_features; o++) {
        float sum = bias[o];
        float scale = weight_scale[o];
        
        for (int i = 0; i < in_features; i++) {
            int w_idx = o * in_features + i;
            float w_val = (float)weight[w_idx] * scale;
            sum += w_val * input[i];
        }
        
        output[o] = sum;
    }
}

// ---------------------------------------------
//  Softmax funciton

static void softmax(float* x, int n) {
    float max_val = x[0];
    for (int i = 1; i < n; i++) {
        if (x[i] > max_val) max_val = x[i];
    }
    
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    
    for (int i = 0; i < n; i++) {
        x[i] /= sum;
    }
}

// ---------------------------------------------
//  Single Conv+ReLU+Pool layer helper

static void apply_conv_layer(
    const float* input,
    float* output,
    const int8_t* weight,
    const float* scale,
    const float* bias,
    const Conv1dConfig* cfg,
    int in_length,
    int* out_length
) {
    int conv_out_len;
    
    // Conv + ReLU
    conv1d_int8(
        input, output,
        weight, scale, bias,
        cfg->in_channels, cfg->out_channels,
        in_length, cfg->kernel_size,
        cfg->padding, cfg->stride,
        cfg->has_relu,
        &conv_out_len
    );
    
    // MaxPool if needed
    if (cfg->has_pool && cfg->pool_size > 1) {
        // Pool in-place (output to temp, then back)
        // For simplicity, we'll use a portion of the buffer
        int pooled_len = conv_out_len / cfg->pool_size;
        
        // Simple in-place pooling by overwriting
        for (int c = 0; c < cfg->out_channels; c++) {
            for (int ox = 0; ox < pooled_len; ox++) {
                float max_val = -1e30f;
                for (int p = 0; p < cfg->pool_size; p++) {
                    int idx = c * conv_out_len + ox * cfg->pool_size + p;
                    if (output[idx] > max_val) {
                        max_val = output[idx];
                    }
                }
                output[c * pooled_len + ox] = max_val;
            }
        }
        *out_length = pooled_len;
    } else {
        *out_length = conv_out_len;
    }
}

// ---------------------------------------------
//  Main Inference Function

int heart_inference(const float* input, int input_length, float* output) {
    float* buf_in = buffer_a;
    float* buf_out = buffer_b;
    float* temp;
    int current_length = input_length;
    int current_channels = 1;

    // Copy input to buffer
    memcpy(buf_in, input, input_length * sizeof(float));

    // Layer 0: initial_conv
    apply_conv_layer(
        buf_in, buf_out,
        initial_conv_weight,
        initial_conv_scale,
        initial_conv_bias,
        &initial_conv_config,
        current_length,
        &current_length
    );
    current_channels = 16;
    temp = buf_in; buf_in = buf_out; buf_out = temp;  // Swap buffers

    // Layer 1: layer_0
    apply_conv_layer(
        buf_in, buf_out,
        layer_0_weight,
        layer_0_scale,
        layer_0_bias,
        &layer_0_config,
        current_length,
        &current_length
    );
    current_channels = 32;
    temp = buf_in; buf_in = buf_out; buf_out = temp;  // Swap buffers

    // Layer 2: layer_1
    apply_conv_layer(
        buf_in, buf_out,
        layer_1_weight,
        layer_1_scale,
        layer_1_bias,
        &layer_1_config,
        current_length,
        &current_length
    );
    current_channels = 32;
    temp = buf_in; buf_in = buf_out; buf_out = temp;  // Swap buffers

    // Layer 3: layer_2
    apply_conv_layer(
        buf_in, buf_out,
        layer_2_weight,
        layer_2_scale,
        layer_2_bias,
        &layer_2_config,
        current_length,
        &current_length
    );
    current_channels = 32;
    temp = buf_in; buf_in = buf_out; buf_out = temp;  // Swap buffers

    // Layer 4: layer_3
    apply_conv_layer(
        buf_in, buf_out,
        layer_3_weight,
        layer_3_scale,
        layer_3_bias,
        &layer_3_config,
        current_length,
        &current_length
    );
    current_channels = 32;
    temp = buf_in; buf_in = buf_out; buf_out = temp;  // Swap buffers

    // Layer 5: layer_4
    apply_conv_layer(
        buf_in, buf_out,
        layer_4_weight,
        layer_4_scale,
        layer_4_bias,
        &layer_4_config,
        current_length,
        &current_length
    );
    current_channels = 32;
    temp = buf_in; buf_in = buf_out; buf_out = temp;  // Swap buffers

    // Layer 6: layer_5
    apply_conv_layer(
        buf_in, buf_out,
        layer_5_weight,
        layer_5_scale,
        layer_5_bias,
        &layer_5_config,
        current_length,
        &current_length
    );
    current_channels = 32;
    temp = buf_in; buf_in = buf_out; buf_out = temp;  // Swap buffers

    // Layer 7: layer_6
    apply_conv_layer(
        buf_in, buf_out,
        layer_6_weight,
        layer_6_scale,
        layer_6_bias,
        &layer_6_config,
        current_length,
        &current_length
    );
    current_channels = 64;
    temp = buf_in; buf_in = buf_out; buf_out = temp;  // Swap buffers

    // Layer 8: layer_7
    apply_conv_layer(
        buf_in, buf_out,
        layer_7_weight,
        layer_7_scale,
        layer_7_bias,
        &layer_7_config,
        current_length,
        &current_length
    );
    current_channels = 64;
    temp = buf_in; buf_in = buf_out; buf_out = temp;  // Swap buffers

    // Layer 9: layer_8
    apply_conv_layer(
        buf_in, buf_out,
        layer_8_weight,
        layer_8_scale,
        layer_8_bias,
        &layer_8_config,
        current_length,
        &current_length
    );
    current_channels = 64;
    temp = buf_in; buf_in = buf_out; buf_out = temp;  // Swap buffers

    // Layer 10: layer_9
    apply_conv_layer(
        buf_in, buf_out,
        layer_9_weight,
        layer_9_scale,
        layer_9_bias,
        &layer_9_config,
        current_length,
        &current_length
    );
    current_channels = 64;
    temp = buf_in; buf_in = buf_out; buf_out = temp;  // Swap buffers

    // Layer 11: layer_10
    apply_conv_layer(
        buf_in, buf_out,
        layer_10_weight,
        layer_10_scale,
        layer_10_bias,
        &layer_10_config,
        current_length,
        &current_length
    );
    current_channels = 64;
    temp = buf_in; buf_in = buf_out; buf_out = temp;  // Swap buffers

    // Layer 12: layer_11
    apply_conv_layer(
        buf_in, buf_out,
        layer_11_weight,
        layer_11_scale,
        layer_11_bias,
        &layer_11_config,
        current_length,
        &current_length
    );
    current_channels = 64;
    temp = buf_in; buf_in = buf_out; buf_out = temp;  // Swap buffers

    // Layer 13: layer_12
    apply_conv_layer(
        buf_in, buf_out,
        layer_12_weight,
        layer_12_scale,
        layer_12_bias,
        &layer_12_config,
        current_length,
        &current_length
    );
    current_channels = 64;
    temp = buf_in; buf_in = buf_out; buf_out = temp;  // Swap buffers

    // Layer 14: layer_13
    apply_conv_layer(
        buf_in, buf_out,
        layer_13_weight,
        layer_13_scale,
        layer_13_bias,
        &layer_13_config,
        current_length,
        &current_length
    );
    current_channels = 128;
    temp = buf_in; buf_in = buf_out; buf_out = temp;  // Swap buffers

    // Layer 15: layer_14
    apply_conv_layer(
        buf_in, buf_out,
        layer_14_weight,
        layer_14_scale,
        layer_14_bias,
        &layer_14_config,
        current_length,
        &current_length
    );
    current_channels = 128;
    temp = buf_in; buf_in = buf_out; buf_out = temp;  // Swap buffers

    // Layer 16: layer_15
    apply_conv_layer(
        buf_in, buf_out,
        layer_15_weight,
        layer_15_scale,
        layer_15_bias,
        &layer_15_config,
        current_length,
        &current_length
    );
    current_channels = 128;
    temp = buf_in; buf_in = buf_out; buf_out = temp;  // Swap buffers

    // Layer 17: layer_16
    apply_conv_layer(
        buf_in, buf_out,
        layer_16_weight,
        layer_16_scale,
        layer_16_bias,
        &layer_16_config,
        current_length,
        &current_length
    );
    current_channels = 128;
    temp = buf_in; buf_in = buf_out; buf_out = temp;  // Swap buffers

    // Layer 18: layer_17
    apply_conv_layer(
        buf_in, buf_out,
        layer_17_weight,
        layer_17_scale,
        layer_17_bias,
        &layer_17_config,
        current_length,
        &current_length
    );
    current_channels = 128;
    temp = buf_in; buf_in = buf_out; buf_out = temp;  // Swap buffers

    // Layer 19: layer_18
    apply_conv_layer(
        buf_in, buf_out,
        layer_18_weight,
        layer_18_scale,
        layer_18_bias,
        &layer_18_config,
        current_length,
        &current_length
    );
    current_channels = 128;
    temp = buf_in; buf_in = buf_out; buf_out = temp;  // Swap buffers

    // Layer 20: layer_19
    apply_conv_layer(
        buf_in, buf_out,
        layer_19_weight,
        layer_19_scale,
        layer_19_bias,
        &layer_19_config,
        current_length,
        &current_length
    );
    current_channels = 128;
    temp = buf_in; buf_in = buf_out; buf_out = temp;  // Swap buffers

    // Layer 21: layer_20
    apply_conv_layer(
        buf_in, buf_out,
        layer_20_weight,
        layer_20_scale,
        layer_20_bias,
        &layer_20_config,
        current_length,
        &current_length
    );
    current_channels = 256;
    temp = buf_in; buf_in = buf_out; buf_out = temp;  // Swap buffers

    // Layer 22: layer_21
    apply_conv_layer(
        buf_in, buf_out,
        layer_21_weight,
        layer_21_scale,
        layer_21_bias,
        &layer_21_config,
        current_length,
        &current_length
    );
    current_channels = 256;
    temp = buf_in; buf_in = buf_out; buf_out = temp;  // Swap buffers

    // Layer 23: layer_22
    apply_conv_layer(
        buf_in, buf_out,
        layer_22_weight,
        layer_22_scale,
        layer_22_bias,
        &layer_22_config,
        current_length,
        &current_length
    );
    current_channels = 256;
    temp = buf_in; buf_in = buf_out; buf_out = temp;  // Swap buffers

    // Layer 24: layer_23
    apply_conv_layer(
        buf_in, buf_out,
        layer_23_weight,
        layer_23_scale,
        layer_23_bias,
        &layer_23_config,
        current_length,
        &current_length
    );
    current_channels = 256;
    temp = buf_in; buf_in = buf_out; buf_out = temp;  // Swap buffers

    // Layer 25: layer_24
    apply_conv_layer(
        buf_in, buf_out,
        layer_24_weight,
        layer_24_scale,
        layer_24_bias,
        &layer_24_config,
        current_length,
        &current_length
    );
    current_channels = 256;
    temp = buf_in; buf_in = buf_out; buf_out = temp;  // Swap buffers

    // Layer 26: layer_25
    apply_conv_layer(
        buf_in, buf_out,
        layer_25_weight,
        layer_25_scale,
        layer_25_bias,
        &layer_25_config,
        current_length,
        &current_length
    );
    current_channels = 256;
    temp = buf_in; buf_in = buf_out; buf_out = temp;  // Swap buffers

    // Layer 27: layer_26
    apply_conv_layer(
        buf_in, buf_out,
        layer_26_weight,
        layer_26_scale,
        layer_26_bias,
        &layer_26_config,
        current_length,
        &current_length
    );
    current_channels = 256;
    temp = buf_in; buf_in = buf_out; buf_out = temp;  // Swap buffers

    // Layer 28: layer_27
    apply_conv_layer(
        buf_in, buf_out,
        layer_27_weight,
        layer_27_scale,
        layer_27_bias,
        &layer_27_config,
        current_length,
        &current_length
    );
    current_channels = 256;
    temp = buf_in; buf_in = buf_out; buf_out = temp;  // Swap buffers

    // Layer 29: layer_28
    apply_conv_layer(
        buf_in, buf_out,
        layer_28_weight,
        layer_28_scale,
        layer_28_bias,
        &layer_28_config,
        current_length,
        &current_length
    );
    current_channels = 256;
    temp = buf_in; buf_in = buf_out; buf_out = temp;  // Swap buffers

    // Layer 30: layer_29
    apply_conv_layer(
        buf_in, buf_out,
        layer_29_weight,
        layer_29_scale,
        layer_29_bias,
        &layer_29_config,
        current_length,
        &current_length
    );
    current_channels = 256;
    temp = buf_in; buf_in = buf_out; buf_out = temp;  // Swap buffers

    // Layer 31: layer_30
    apply_conv_layer(
        buf_in, buf_out,
        layer_30_weight,
        layer_30_scale,
        layer_30_bias,
        &layer_30_config,
        current_length,
        &current_length
    );
    current_channels = 256;
    temp = buf_in; buf_in = buf_out; buf_out = temp;  // Swap buffers

    // Layer 32: layer_31
    apply_conv_layer(
        buf_in, buf_out,
        layer_31_weight,
        layer_31_scale,
        layer_31_bias,
        &layer_31_config,
        current_length,
        &current_length
    );
    current_channels = 256;
    temp = buf_in; buf_in = buf_out; buf_out = temp;  // Swap buffers

    // Layer 33: layer_32
    apply_conv_layer(
        buf_in, buf_out,
        layer_32_weight,
        layer_32_scale,
        layer_32_bias,
        &layer_32_config,
        current_length,
        &current_length
    );
    current_channels = 256;
    temp = buf_in; buf_in = buf_out; buf_out = temp;  // Swap buffers

    // Global Average Pooling
    float pooled[256];
    global_avg_pool1d(buf_in, pooled, current_channels, current_length);

    // Fully Connected Layer
    float logits[NUM_CLASSES];
    linear_int8(
        pooled, logits,
        fc_weight,
        fc_scale,
        fc_bias,
        FC_IN_FEATURES, FC_OUT_FEATURES
    );

    // Perform Softmax for probabilities 
    softmax(logits, NUM_CLASSES);

    // Copy output
    output[0] = logits[0];  // No Murmur probability
    output[1] = logits[1];  // Murmur probability

    return 0;  // Finished running
}


// ---------------------------------------------
//  High-level prediction function

typedef enum {
    PREDICTION_NO_MURMUR = 0,
    PREDICTION_MURMUR = 1
} MurmurPrediction;

MurmurPrediction heart_predict(const float* input, int input_length, float threshold) {
    float probs[2];
    heart_inference(input, input_length, probs);
    
    return (probs[1] >= threshold) ? PREDICTION_MURMUR : PREDICTION_NO_MURMUR;
}

float heart_get_murmur_probability(const float* input, int input_length) {
    float probs[2];
    heart_inference(input, input_length, probs);
    return probs[1];
}

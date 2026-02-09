/*
 * Eko ResNet34 INT8 Inference Engine
 * CORRECTED VERSION: BatchNorm applied separately (not folded)
 * 
 * Layer order: Conv -> ReLU -> BatchNorm -> Pool
 */

#include "heart_inference.h"
#include "model_weights.h"

#include <cstring>
#include <cmath>
#include <algorithm>
#include <numeric>

// For ESP32
#ifdef ESP_PLATFORM
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#define YIELD() vTaskDelay(1)
#define LOG_INFO(tag, fmt, ...) ESP_LOGI(tag, fmt, ##__VA_ARGS__)
#else
#include <cstdio>
#define YIELD()
#define LOG_INFO(tag, fmt, ...) std::printf("[%s] " fmt "\n", tag, ##__VA_ARGS__)
#endif

namespace heart_inference {

using namespace heart_model;

// ---------------------------------------------
//  Utility functions
// ---------------------------------------------

namespace {

inline float relu(float x) {
    return std::max(0.0f, x);
}

// ---------------------------------------------
//  BatchNorm1D (applied after ReLU)
//  y = gamma * (x - mean) / sqrt(var + eps) + beta
// ---------------------------------------------

void batchnorm1d(
    float* data,           // In-place: [channels, length]
    const float* gamma,    // Scale: [channels]
    const float* beta,     // Shift: [channels]
    const float* mean,     // Running mean: [channels]
    const float* var,      // Running variance: [channels]
    float eps,
    int channels,
    int length
) {
    for (int c = 0; c < channels; ++c) {
        const float scale = gamma[c] / std::sqrt(var[c] + eps);
        const float shift = beta[c] - mean[c] * scale;
        
        float* channel_data = data + c * length;
        for (int i = 0; i < length; ++i) {
            channel_data[i] = channel_data[i] * scale + shift;
        }
    }
}

// ---------------------------------------------
//  Conv1D with INT8 weights (no BN, no pool)
// ---------------------------------------------

void conv1d_int8(
    const float* input,
    float* output,
    const int8_t* weight,
    const float* weight_scale,
    const float* bias,
    int in_channels,
    int out_channels,
    int in_length,
    int kernel_size,
    int padding,
    int stride,
    bool apply_relu,
    int& out_length
) {
    out_length = (in_length + 2 * padding - kernel_size) / stride + 1;

    int total_elements = out_channels * out_length;
    if (total_elements > 400000) {
        ESP_LOGE("AI", "CRITICAL: Output buffer too small! Need %d, have 400000", total_elements);
        return; // Stop the layer from running
    }
    
    for (int oc = 0; oc < out_channels; ++oc) {
        if (oc % 4 == 0) {
            YIELD();
        }
        
        const float scale = weight_scale[oc];
        float* out_channel = output + oc * out_length;
        
        for (int ox = 0; ox < out_length; ++ox) {
            float sum = bias[oc];
            const int in_start = ox * stride - padding;
            
            for (int ic = 0; ic < in_channels; ++ic) {
                const float* in_channel = input + ic * in_length;
                const int8_t* w_ptr = weight + (oc * in_channels + ic) * kernel_size;
                
                for (int k = 0; k < kernel_size; ++k) {
                    const int in_x = in_start + k;
                    
                    if (in_x >= 0 && in_x < in_length) {
                        const float w_val = static_cast<float>(w_ptr[k]) * scale;
                        sum += w_val * in_channel[in_x];
                    }
                }
            }
            
            out_channel[ox] = apply_relu ? relu(sum) : sum;
        }
    }
}

// ---------------------------------------------
//  MaxPool1D (in-place capable)
// ---------------------------------------------

void maxpool1d_inplace(
    float* data,
    int channels,
    int in_length,
    int pool_size,
    int& out_length
) {
    out_length = in_length / pool_size;
    
    for (int c = 0; c < channels; ++c) {
        float* channel_data = data + c * in_length;
        float* out_ptr = data + c * out_length;
        
        for (int ox = 0; ox < out_length; ++ox) {
            float max_val = -1e30f;
            const int base = ox * pool_size;
            
            for (int p = 0; p < pool_size; ++p) {
                max_val = std::max(max_val, channel_data[base + p]);
            }
            out_ptr[ox] = max_val;
        }
    }
}

// ---------------------------------------------
//  Global Average Pooling
// ---------------------------------------------

void global_avg_pool1d(
    const float* input,
    float* output,
    int channels,
    int length
) {
    const float inv_length = 1.0f / static_cast<float>(length);
    
    for (int c = 0; c < channels; ++c) {
        const float* channel_data = input + c * length;
        float sum = 0.0f;
        
        for (int i = 0; i < length; ++i) {
            sum += channel_data[i];
        }
        output[c] = sum * inv_length;
    }
}

// ---------------------------------------------
//  Fully Connected with INT8 weights
// ---------------------------------------------

void linear_int8(
    const float* input,
    float* output,
    const int8_t* weight,
    const float* weight_scale,
    const float* bias,
    int in_features,
    int out_features
) {
    for (int o = 0; o < out_features; ++o) {
        float sum = bias[o];
        const float scale = weight_scale[o];
        const int8_t* w_row = weight + o * in_features;
        
        for (int i = 0; i < in_features; ++i) {
            sum += static_cast<float>(w_row[i]) * scale * input[i];
        }
        
        output[o] = sum;
    }
}

// ---------------------------------------------
//  Softmax
// ---------------------------------------------

void softmax(float* x, int n) {
    const float max_val = *std::max_element(x, x + n);
    
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        x[i] = std::exp(x[i] - max_val);
        sum += x[i];
    }
    
    const float inv_sum = 1.0f / sum;
    for (int i = 0; i < n; ++i) {
        x[i] *= inv_sum;
    }
}

// ---------------------------------------------
//  Layer application helper
// ---------------------------------------------

template<typename LayerWeights>
void apply_layer(
    float*& buf_in,
    float*& buf_out,
    int& current_length,
    const Conv1dConfig& config,
    const int8_t* weight,
    const float* scale,
    const float* bias,
    const float* bn_gamma = nullptr,
    const float* bn_beta = nullptr,
    const float* bn_mean = nullptr,
    const float* bn_var = nullptr,
    float bn_eps = 1e-5f
) {
    YIELD();
    
    conv1d_int8(
        buf_in, buf_out,
        weight, scale, bias,
        config.in_channels, config.out_channels,
        current_length, config.kernel_size,
        config.padding, config.stride,
        config.has_relu,
        current_length
    );
    
    if (config.has_bn && bn_gamma != nullptr) {
        batchnorm1d(
            buf_out,
            bn_gamma, bn_beta,
            bn_mean, bn_var,
            bn_eps,
            config.out_channels, current_length
        );
    }
    
    if (config.has_pool && config.pool_size > 1) {
        maxpool1d_inplace(buf_out, config.out_channels, current_length, config.pool_size, current_length);
    }
    
    std::swap(buf_in, buf_out);
}

} // anonymous namespace

// ---------------------------------------------
//  Main Inference Function
// ---------------------------------------------

int run_inference(const float* input, int input_length, float* output, float* buffer_a, float* buffer_b) {
    
    if (buffer_a == nullptr || buffer_b == nullptr) {
        return -1;
    }
    
    if (input_length > 160000) {
        return -1;
    }

    LOG_INFO("AI", "Starting inference with %d samples", input_length);

    float* buf_in = buffer_a;
    float* buf_out = buffer_b;
    int current_length = input_length;

    // Copy input to buffer
    std::memcpy(buf_in, input, input_length * sizeof(float));

    // Layer 0: initial_conv (1->16) + pool(4)
    apply_layer<void>(
        buf_in, buf_out, current_length,
        initial_conv_config,
        initial_conv_weight, initial_conv_scale, initial_conv_bias,
        initial_conv_bn_gamma, initial_conv_bn_beta,
        initial_conv_bn_mean, initial_conv_bn_var,
        initial_conv_bn_eps
    );
    LOG_INFO("AI", "After initial_conv: length=%d", current_length);

    // Layer 1: layer_0 (16->32) + pool(2)
    apply_layer<void>(
        buf_in, buf_out, current_length,
        layer_0_config,
        layer_0_weight, layer_0_scale, layer_0_bias,
        layer_0_bn_gamma, layer_0_bn_beta,
        layer_0_bn_mean, layer_0_bn_var,
        layer_0_bn_eps
    );

    // Layer 2: layer_1 (32->32)
    apply_layer<void>(
        buf_in, buf_out, current_length,
        layer_1_config,
        layer_1_weight, layer_1_scale, layer_1_bias,
        layer_1_bn_gamma, layer_1_bn_beta,
        layer_1_bn_mean, layer_1_bn_var,
        layer_1_bn_eps
    );

    // Layer 3: layer_2 (32->32)
    apply_layer<void>(
        buf_in, buf_out, current_length,
        layer_2_config,
        layer_2_weight, layer_2_scale, layer_2_bias,
        layer_2_bn_gamma, layer_2_bn_beta,
        layer_2_bn_mean, layer_2_bn_var,
        layer_2_bn_eps
    );

    // Layer 4: layer_3 (32->32) + pool(2)
    apply_layer<void>(
        buf_in, buf_out, current_length,
        layer_3_config,
        layer_3_weight, layer_3_scale, layer_3_bias,
        layer_3_bn_gamma, layer_3_bn_beta,
        layer_3_bn_mean, layer_3_bn_var,
        layer_3_bn_eps
    );

    // Layer 5: layer_4 (32->32)
    apply_layer<void>(
        buf_in, buf_out, current_length,
        layer_4_config,
        layer_4_weight, layer_4_scale, layer_4_bias,
        layer_4_bn_gamma, layer_4_bn_beta,
        layer_4_bn_mean, layer_4_bn_var,
        layer_4_bn_eps
    );
    LOG_INFO("AI", "After layer_4: length=%d", current_length);

    // Layer 6: layer_5 (32->32)
    apply_layer<void>(
        buf_in, buf_out, current_length,
        layer_5_config,
        layer_5_weight, layer_5_scale, layer_5_bias,
        layer_5_bn_gamma, layer_5_bn_beta,
        layer_5_bn_mean, layer_5_bn_var,
        layer_5_bn_eps
    );

    // Layer 7: layer_6 (32->64) + pool(2)
    apply_layer<void>(
        buf_in, buf_out, current_length,
        layer_6_config,
        layer_6_weight, layer_6_scale, layer_6_bias,
        layer_6_bn_gamma, layer_6_bn_beta,
        layer_6_bn_mean, layer_6_bn_var,
        layer_6_bn_eps
    );

    // Layer 8: layer_7 (64->64)
    apply_layer<void>(
        buf_in, buf_out, current_length,
        layer_7_config,
        layer_7_weight, layer_7_scale, layer_7_bias,
        layer_7_bn_gamma, layer_7_bn_beta,
        layer_7_bn_mean, layer_7_bn_var,
        layer_7_bn_eps
    );

    // Layer 9: layer_8 (64->64)
    apply_layer<void>(
        buf_in, buf_out, current_length,
        layer_8_config,
        layer_8_weight, layer_8_scale, layer_8_bias,
        layer_8_bn_gamma, layer_8_bn_beta,
        layer_8_bn_mean, layer_8_bn_var,
        layer_8_bn_eps
    );

    // Layer 10: layer_9 (64->64)
    apply_layer<void>(
        buf_in, buf_out, current_length,
        layer_9_config,
        layer_9_weight, layer_9_scale, layer_9_bias,
        layer_9_bn_gamma, layer_9_bn_beta,
        layer_9_bn_mean, layer_9_bn_var,
        layer_9_bn_eps
    );
    LOG_INFO("AI", "After layer_9: length=%d", current_length);

    // Layer 11: layer_10 (64->64) + pool(2)
    apply_layer<void>(
        buf_in, buf_out, current_length,
        layer_10_config,
        layer_10_weight, layer_10_scale, layer_10_bias,
        layer_10_bn_gamma, layer_10_bn_beta,
        layer_10_bn_mean, layer_10_bn_var,
        layer_10_bn_eps
    );

    // Layer 12: layer_11 (64->64)
    apply_layer<void>(
        buf_in, buf_out, current_length,
        layer_11_config,
        layer_11_weight, layer_11_scale, layer_11_bias,
        layer_11_bn_gamma, layer_11_bn_beta,
        layer_11_bn_mean, layer_11_bn_var,
        layer_11_bn_eps
    );

    // Layer 13: layer_12 (64->64)
    apply_layer<void>(
        buf_in, buf_out, current_length,
        layer_12_config,
        layer_12_weight, layer_12_scale, layer_12_bias,
        layer_12_bn_gamma, layer_12_bn_beta,
        layer_12_bn_mean, layer_12_bn_var,
        layer_12_bn_eps
    );

    // Layer 14: layer_13 (64->128) + pool(2)
    apply_layer<void>(
        buf_in, buf_out, current_length,
        layer_13_config,
        layer_13_weight, layer_13_scale, layer_13_bias,
        layer_13_bn_gamma, layer_13_bn_beta,
        layer_13_bn_mean, layer_13_bn_var,
        layer_13_bn_eps
    );

    // Layer 15: layer_14 (128->128)
    apply_layer<void>(
        buf_in, buf_out, current_length,
        layer_14_config,
        layer_14_weight, layer_14_scale, layer_14_bias,
        layer_14_bn_gamma, layer_14_bn_beta,
        layer_14_bn_mean, layer_14_bn_var,
        layer_14_bn_eps
    );
    LOG_INFO("AI", "After layer_14: length=%d", current_length);

    // Layer 16: layer_15 (128->128)
    apply_layer<void>(
        buf_in, buf_out, current_length,
        layer_15_config,
        layer_15_weight, layer_15_scale, layer_15_bias,
        layer_15_bn_gamma, layer_15_bn_beta,
        layer_15_bn_mean, layer_15_bn_var,
        layer_15_bn_eps
    );

    // Layer 17: layer_16 (128->128)
    apply_layer<void>(
        buf_in, buf_out, current_length,
        layer_16_config,
        layer_16_weight, layer_16_scale, layer_16_bias,
        layer_16_bn_gamma, layer_16_bn_beta,
        layer_16_bn_mean, layer_16_bn_var,
        layer_16_bn_eps
    );

    // Layer 18: layer_17 (128->128) + pool(2)
    apply_layer<void>(
        buf_in, buf_out, current_length,
        layer_17_config,
        layer_17_weight, layer_17_scale, layer_17_bias,
        layer_17_bn_gamma, layer_17_bn_beta,
        layer_17_bn_mean, layer_17_bn_var,
        layer_17_bn_eps
    );

    // Layer 19: layer_18 (128->128)
    apply_layer<void>(
        buf_in, buf_out, current_length,
        layer_18_config,
        layer_18_weight, layer_18_scale, layer_18_bias,
        layer_18_bn_gamma, layer_18_bn_beta,
        layer_18_bn_mean, layer_18_bn_var,
        layer_18_bn_eps
    );

    // Layer 20: layer_19 (128->128)
    apply_layer<void>(
        buf_in, buf_out, current_length,
        layer_19_config,
        layer_19_weight, layer_19_scale, layer_19_bias,
        layer_19_bn_gamma, layer_19_bn_beta,
        layer_19_bn_mean, layer_19_bn_var,
        layer_19_bn_eps
    );
    LOG_INFO("AI", "After layer_19: length=%d", current_length);

    // Layer 21: layer_20 (128->256) + pool(2)
    apply_layer<void>(
        buf_in, buf_out, current_length,
        layer_20_config,
        layer_20_weight, layer_20_scale, layer_20_bias,
        layer_20_bn_gamma, layer_20_bn_beta,
        layer_20_bn_mean, layer_20_bn_var,
        layer_20_bn_eps
    );

    // Layer 22: layer_21 (256->256)
    apply_layer<void>(
        buf_in, buf_out, current_length,
        layer_21_config,
        layer_21_weight, layer_21_scale, layer_21_bias,
        layer_21_bn_gamma, layer_21_bn_beta,
        layer_21_bn_mean, layer_21_bn_var,
        layer_21_bn_eps
    );

    // Layer 23: layer_22 (256->256)
    apply_layer<void>(
        buf_in, buf_out, current_length,
        layer_22_config,
        layer_22_weight, layer_22_scale, layer_22_bias,
        layer_22_bn_gamma, layer_22_bn_beta,
        layer_22_bn_mean, layer_22_bn_var,
        layer_22_bn_eps
    );

    // Layer 24: layer_23 (256->256)
    apply_layer<void>(
        buf_in, buf_out, current_length,
        layer_23_config,
        layer_23_weight, layer_23_scale, layer_23_bias,
        layer_23_bn_gamma, layer_23_bn_beta,
        layer_23_bn_mean, layer_23_bn_var,
        layer_23_bn_eps
    );

    // Layer 25: layer_24 (256->256) + pool(2)
    apply_layer<void>(
        buf_in, buf_out, current_length,
        layer_24_config,
        layer_24_weight, layer_24_scale, layer_24_bias,
        layer_24_bn_gamma, layer_24_bn_beta,
        layer_24_bn_mean, layer_24_bn_var,
        layer_24_bn_eps
    );
    LOG_INFO("AI", "After layer_24: length=%d", current_length);

    // Layer 26: layer_25 (256->256)
    apply_layer<void>(
        buf_in, buf_out, current_length,
        layer_25_config,
        layer_25_weight, layer_25_scale, layer_25_bias,
        layer_25_bn_gamma, layer_25_bn_beta,
        layer_25_bn_mean, layer_25_bn_var,
        layer_25_bn_eps
    );

    // Layer 27: layer_26 (256->256)
    apply_layer<void>(
        buf_in, buf_out, current_length,
        layer_26_config,
        layer_26_weight, layer_26_scale, layer_26_bias,
        layer_26_bn_gamma, layer_26_bn_beta,
        layer_26_bn_mean, layer_26_bn_var,
        layer_26_bn_eps
    );

    // Layer 28: layer_27 (256->256) + pool(2)
    apply_layer<void>(
        buf_in, buf_out, current_length,
        layer_27_config,
        layer_27_weight, layer_27_scale, layer_27_bias,
        layer_27_bn_gamma, layer_27_bn_beta,
        layer_27_bn_mean, layer_27_bn_var,
        layer_27_bn_eps
    );

    // Layer 29: layer_28 (256->256)
    apply_layer<void>(
        buf_in, buf_out, current_length,
        layer_28_config,
        layer_28_weight, layer_28_scale, layer_28_bias,
        layer_28_bn_gamma, layer_28_bn_beta,
        layer_28_bn_mean, layer_28_bn_var,
        layer_28_bn_eps
    );

    // Layer 30: layer_29 (256->256)
    apply_layer<void>(
        buf_in, buf_out, current_length,
        layer_29_config,
        layer_29_weight, layer_29_scale, layer_29_bias,
        layer_29_bn_gamma, layer_29_bn_beta,
        layer_29_bn_mean, layer_29_bn_var,
        layer_29_bn_eps
    );
    LOG_INFO("AI", "After layer_29: length=%d", current_length);

    // Layer 31: layer_30 (256->256) + pool(2)
    apply_layer<void>(
        buf_in, buf_out, current_length,
        layer_30_config,
        layer_30_weight, layer_30_scale, layer_30_bias,
        layer_30_bn_gamma, layer_30_bn_beta,
        layer_30_bn_mean, layer_30_bn_var,
        layer_30_bn_eps
    );

    // Layer 32: layer_31 (256->256)
    apply_layer<void>(
        buf_in, buf_out, current_length,
        layer_31_config,
        layer_31_weight, layer_31_scale, layer_31_bias,
        layer_31_bn_gamma, layer_31_bn_beta,
        layer_31_bn_mean, layer_31_bn_var,
        layer_31_bn_eps
    );

    // Layer 33: layer_32 (256->256)
    apply_layer<void>(
        buf_in, buf_out, current_length,
        layer_32_config,
        layer_32_weight, layer_32_scale, layer_32_bias,
        layer_32_bn_gamma, layer_32_bn_beta,
        layer_32_bn_mean, layer_32_bn_var,
        layer_32_bn_eps
    );


    LOG_INFO("AI", "Final length before GAP: %d", current_length);

    // Global Average Pooling
    static std::array<float, 256> pooled{};
    global_avg_pool1d(buf_in, pooled.data(), 256, current_length);

    LOG_INFO("AI", "After GAP: pooled[0]=%f, pooled[1]=%f", pooled[0], pooled[1]);

    // Fully Connected Layer
    static std::array<float, NUM_CLASSES> logits{};
    linear_int8(pooled.data(), logits.data(), fc_weight, fc_scale, fc_bias, FC_IN_FEATURES, FC_OUT_FEATURES);

    LOG_INFO("AI", "After FC: logits[0]=%f, logits[1]=%f", logits[0], logits[1]);

    // Softmax
    softmax(logits.data(), NUM_CLASSES);

    LOG_INFO("AI", "After softmax: %f, %f", logits[0], logits[1]);

    output[0] = logits[0];
    output[1] = logits[1];

    return 0;
}

} // namespace heart_inference

#ifndef ESP_PLATFORM

#include <fstream>
#include <cstdint>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// Buffer size after initial pooling: 16 channels * 15000 length = 240,000 floats
// Worst case mid-network: 32 channels * 7500 = 240,000 floats
#define MAX_BUFFER_FLOATS 240000
#define MAX_INPUT_SAMPLES 80000

// ---------------------------------------------
//  WAV File Loading

#pragma pack(push, 1)
struct WAVHeader {
    char riff[4];           // "RIFF"
    uint32_t fileSize;      // File size - 8
    char wave[4];           // "WAVE"
    char fmt[4];            // "fmt "
    uint32_t fmtSize;       // Format chunk size
    uint16_t audioFormat;   // 1 = PCM
    uint16_t numChannels;   // 1 = mono, 2 = stereo
    uint32_t sampleRate;    // e.g., 44100
    uint32_t byteRate;      // sampleRate * numChannels * bitsPerSample/8
    uint16_t blockAlign;    // numChannels * bitsPerSample/8
    uint16_t bitsPerSample; // 8, 16, 24, or 32
};
#pragma pack(pop)

static int load_wav(const char* filename, float* output, int max_samples, int* out_length, int* out_sample_rate) {
    FILE* fp = fopen(filename, "rb");
    if (!fp) {
        fprintf(stderr, "Error: Cannot open file %s\n", filename);
        return -1;
    }
    
    WAVHeader header;
    if (fread(&header, sizeof(WAVHeader), 1, fp) != 1) {
        fprintf(stderr, "Error: Cannot read WAV header\n");
        fclose(fp);
        return -1;
    }
    
    // Validate RIFF/WAVE
    if (strncmp(header.riff, "RIFF", 4) != 0 || strncmp(header.wave, "WAVE", 4) != 0) {
        fprintf(stderr, "Error: Not a valid WAV file\n");
        fclose(fp);
        return -1;
    }
    
    printf("WAV Info: %d channels, %d Hz, %d bits/sample\n", 
           header.numChannels, header.sampleRate, header.bitsPerSample);
    
    // Skip any extra format bytes (with padding if odd)
    if (header.fmtSize > 16) {
        uint32_t extraBytes = header.fmtSize - 16;
        fseek(fp, extraBytes + (extraBytes & 1), SEEK_CUR);
    }
    
    // Find data chunk
    char chunkId[4];
    uint32_t chunkSize;
    int foundData = 0;
    
    while (fread(chunkId, 4, 1, fp) == 1) {
        if (fread(&chunkSize, 4, 1, fp) != 1) break;
        
        printf("Found chunk: '%.4s' size=%u\n", chunkId, chunkSize);
        
        if (strncmp(chunkId, "data", 4) == 0) {
            foundData = 1;
            break;
        }
        
        // Skip chunk data + padding byte if odd size
        fseek(fp, chunkSize + (chunkSize & 1), SEEK_CUR);
    }
    
    if (!foundData) {
        fprintf(stderr, "Error: Cannot find data chunk\n");
        fclose(fp);
        return -1;
    }
    
    int bytesPerSample = header.bitsPerSample / 8;
    int totalSamples = chunkSize / (bytesPerSample * header.numChannels);
    
    printf("Data chunk: %d bytes, %d samples\n", chunkSize, totalSamples);
    
    if (totalSamples > max_samples) {
        printf("Warning: Truncating from %d to %d samples\n", totalSamples, max_samples);
        totalSamples = max_samples;
    }
    
    // Read and convert samples
    for (int i = 0; i < totalSamples; i++) {
        float sample = 0.0f;
        
        for (int ch = 0; ch < header.numChannels; ch++) {
            float chSample = 0.0f;
            
            if (header.bitsPerSample == 8) {
                uint8_t val;
                fread(&val, 1, 1, fp);
                chSample = (float)(val - 128) / 128.0f;
            } else if (header.bitsPerSample == 16) {
                int16_t val;
                fread(&val, 2, 1, fp);
                chSample = (float)val / 32768.0f;
            } else if (header.bitsPerSample == 24) {
                uint8_t bytes[3];
                fread(bytes, 3, 1, fp);
                int32_t val = (bytes[2] << 24) | (bytes[1] << 16) | (bytes[0] << 8);
                val >>= 8; // Sign extend
                chSample = (float)val / 8388608.0f;
            } else if (header.bitsPerSample == 32) {
                int32_t val;
                fread(&val, 4, 1, fp);
                chSample = (float)val / 2147483648.0f;
            }
            
            sample += chSample;
        }
        
        // Average channels for mono output
        output[i] = sample / header.numChannels;
    }
    
    fclose(fp);
    *out_length = totalSamples;
    *out_sample_rate = header.sampleRate;
    return 0;
}

// #include <vector>
// #include <iostream>

// int main(int argc, char** argv) {
//     if (argc < 2) {
//         std::cerr << "Usage: " << argv[0] << " file.wav\n";
//         return 1;
//     }

//     // Pre-allocate buffer for samples
//     std::vector<float> samples(MAX_INPUT_SAMPLES);
//     int sample_count = 0;
//     int sample_rate = 0;

//     // Call with correct signature
//     if (load_wav(argv[1], samples.data(), MAX_INPUT_SAMPLES, &sample_count, &sample_rate) != 0) {
//         std::cerr << "Failed to load WAV file\n";
//         return 1;
//     }

//     std::cout << "Loaded " << sample_count
//               << " samples @ " << sample_rate << " Hz\n";

//     std::vector<float> output(2);
    
//     // Buffer needs to hold: max_channels * max_length
//     // Worst case before first pool: 16 channels * 60000 samples = 960,000 floats
//     // After first pool: 16 * 15000 = 240,000
//     // To be safe, use 1,000,000 floats (~4MB each)
//     const size_t BUFFER_SIZE = 1000000;
//     std::vector<float> buffer_a(BUFFER_SIZE);
//     std::vector<float> buffer_b(BUFFER_SIZE);

//     int ret = heart_inference::run_inference(
//         samples.data(),
//         sample_count,
//         output.data(),
//         buffer_a.data(),
//         buffer_b.data()
//     );

//     std::cout << "ret=" << ret
//               << " Normal=" << output[0]
//               << " Abnormal=" << output[1] << std::endl;

//     return 0;
// }


#endif // ESP_PLATFORM
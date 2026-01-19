/*
 * 34 layer Inference Header
 */

#ifndef HEART_INFERENCE_H
#define HEART_INFERENCE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Prediction results
typedef enum {
    PREDICTION_NO_MURMUR = 0,
    PREDICTION_MURMUR = 1
} MurmurPrediction;

/**
 * Run full inference on preprocessed audio
 * 
 * @param input  Preprocessed audio samples (normalized to [-1, 1])
 * @param input_length  Number of samples (max 60000 for 30s at 2000Hz)
 * @param output  Output probabilities [no_murmur, murmur]
 * @return 0 on success
 */
int heart_inference(const float* input, int input_length, float* output);

/**
 * Get prediction with threshold
 * 
 * @param input  Preprocessed audio samples
 * @param input_length  Number of samples
 * @param threshold  Decision threshold (0.76 for optimal, 0.65 for high sensitivity)
 * @return PREDICTION_MURMUR or PREDICTION_NO_MURMUR
 */
MurmurPrediction heart_predict(const float* input, int input_length, float threshold);

/**
 * Get murmur probability
 */
float heart_get_murmur_probability(const float* input, int input_length);

#ifdef __cplusplus
}
#endif

#endif // HEART_INFERENCE_H

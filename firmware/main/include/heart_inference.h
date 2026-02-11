/*
 * Heart Murmur Detection Inference Header
 */

#pragma once

#include <cstdint>

namespace heart_inference {

/**
 * Run inference on preprocessed audio
 * 
 * @param input  Preprocessed audio samples (normalized to [-1, 1])
 * @param input_length  Number of samples (max 60000 for 30s at 2000Hz)
 * @param output  Output probabilities [no_murmur, murmur]
 * @param buffer_a  Working buffer (at least 256*15000 floats)
 * @param buffer_b  Working buffer (at least 256*15000 floats)
 * @return 0 on success
 */
int run_inference(const float* input, int input_length, float* output, float* buffer_a, float* buffer_b);

} // namespace heart_inference

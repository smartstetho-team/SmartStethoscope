/*
 * Heart Murmur Detection - QAT INT8 Inference API
 *
 * Model: Eko ResNet34 with residual connections
 * Quantization: QAT (Quantization-Aware Training)
 */

#pragma once

#include "dsp_ml_setup.h"
#include <cstdint>

namespace heart_inference {

/**
 * Run inference on preprocessed audio.
 *
 * @param input         Preprocessed audio (normalized [-1, 1])
 * @param input_length  Number of samples (max 80000)
 * @param output        Output probabilities [P(no_murmur), P(murmur)]
 * @param buffer_a      Working buffer (at least 256 * max_temporal_length floats)
 * @param buffer_b      Working buffer (same size as buffer_a)
 * @param buffer_skip   Skip connection buffer (same size as buffer_a)
 * @param ui            LCD UI Handle (for UI updates)
 * @return 0 on success, -1 on error
 *
 * Buffer sizing guide:
 *   For 30s at 2000Hz (60000 samples):
 *     After initial pool(4): 15000 length, max 256 channels
 *     Buffer size: 256 * 15000 = 3,840,000 floats = ~14.6 MB each
 *
 *   For 8s at 4000Hz (32000 samples):
 *     After initial pool(4): 8000 length, max 256 channels
 *     Buffer size: 256 * 8000 = 2,048,000 floats = ~7.8 MB each
 *
 * Example:
 *   constexpr int BUF_SIZE = 256 * 15000;
 *   auto buf_a = std::make_unique<float[]>(BUF_SIZE);
 *   auto buf_b = std::make_unique<float[]>(BUF_SIZE);
 *   auto buf_s = std::make_unique<float[]>(BUF_SIZE);
 *
 *   float probs[2];
 *   int ret = run_inference(audio, num_samples, probs,
 *                           buf_a.get(), buf_b.get(), buf_s.get());
 *   if (ret == 0) {
 *       bool has_murmur = probs[1] >= threshold;
 *   }
 */
int run_inference(
    const float* input, int input_length,
    float* output,
    float* buffer_a, float* buffer_b, float* buffer_skip, ui_update_handle_t *ui
);

} // namespace heart_inference

/*
 * Eko ResNet34 QAT INT8 Inference
 * Conv -> ReLU -> BN -> Pool with residual skip connections
 *
 * Memory layout: data is stored as [channels, length]
 * Three buffers are used:
 *   buf_in  - current layer input
 *   buf_out - current layer output / main path
 *   buf_skip - stores input for residual addition
 */

#include "dsp_ml_setup.h"
#include "heart_inference.h"
#include "model_weights.h"

#include <cstring>
#include <cmath>
#include <algorithm>
#include <array>

#ifdef ESP_PLATFORM
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "lvgl.h"
#define YIELD() vTaskDelay(1)
#define LOG_INFO(tag, fmt, ...) ESP_LOGI(tag, fmt, ##__VA_ARGS__)
#else
#include <cstdio>
#define YIELD()
#define LOG_INFO(tag, fmt, ...) std::printf("[%s] " fmt "\n", tag, ##__VA_ARGS__)
#endif

namespace heart_inference {
using namespace heart_model;

namespace {

// -----------------------------------------------
//  Primitive operations
// -----------------------------------------------

inline float relu(float x) { return std::max(0.0f, x); }

// Conv1D with INT8 weights, float activations
void conv1d_int8(
    const float* input, float* output,
    const int8_t* weight, const float* scale, const float* bias,
    int ic, int oc, int in_len, int ks, int pad, int stride,
    bool apply_relu, int& out_len
) {
    out_len = (in_len + 2 * pad - ks) / stride + 1;
    for (int o = 0; o < oc; ++o) {
        if (o % 4 == 0) YIELD();
        const float s = scale[o];
        float* out_ch = output + o * out_len;
        for (int ox = 0; ox < out_len; ++ox) {
            float sum = bias[o];
            const int base = ox * stride - pad;
            for (int i = 0; i < ic; ++i) {
                const float* in_ch = input + i * in_len;
                const int8_t* w = weight + (o * ic + i) * ks;
                for (int k = 0; k < ks; ++k) {
                    const int ix = base + k;
                    if (ix >= 0 && ix < in_len)
                        sum += static_cast<float>(w[k]) * s * in_ch[ix];
                }
            }
            out_ch[ox] = apply_relu ? relu(sum) : sum;
        }
    }
}

// BatchNorm1D: y = gamma * (x - mean) / sqrt(var + eps) + beta
void batchnorm1d(
    float* data,
    const float* gamma, const float* beta,
    const float* mean, const float* var, float eps,
    int channels, int length
) {
    for (int c = 0; c < channels; ++c) {
        const float s = gamma[c] / std::sqrt(var[c] + eps);
        const float b = beta[c] - mean[c] * s;
        float* ch = data + c * length;
        for (int i = 0; i < length; ++i)
            ch[i] = ch[i] * s + b;
    }
}

// MaxPool1D (in-place)
void maxpool1d(float* data, int channels, int in_len, int pool_size, int& out_len) {
    out_len = in_len / pool_size;
    for (int c = 0; c < channels; ++c) {
        float* ch_in = data + c * in_len;
        float* ch_out = data + c * out_len;
        for (int ox = 0; ox < out_len; ++ox) {
            float mx = -1e30f;
            const int base = ox * pool_size;
            for (int p = 0; p < pool_size; ++p)
                mx = std::max(mx, ch_in[base + p]);
            ch_out[ox] = mx;
        }
    }
}

// Element-wise addition: out[i] += skip[i]
void residual_add(float* out, const float* skip, int total_elements) {
    for (int i = 0; i < total_elements; ++i)
        out[i] += skip[i];
}

// 1x1 Conv projection for skip connections (INT8 weights)
void conv1x1_int8(
    const float* input, float* output,
    const int8_t* weight, const float* scale, const float* bias,
    int ic, int oc, int length
) {
    for (int o = 0; o < oc; ++o) {
        const float s = scale[o];
        float* out_ch = output + o * length;
        for (int x = 0; x < length; ++x) {
            float sum = bias[o];
            for (int i = 0; i < ic; ++i) {
                sum += static_cast<float>(weight[o * ic + i]) * s * input[i * length + x];
            }
            out_ch[x] = sum;
        }
    }
}

// Global Average Pooling
void global_avg_pool1d(const float* input, float* output, int channels, int length) {
    const float inv = 1.0f / static_cast<float>(length);
    for (int c = 0; c < channels; ++c) {
        const float* ch = input + c * length;
        float sum = 0.0f;
        for (int i = 0; i < length; ++i)
            sum += ch[i];
        output[c] = sum * inv;
    }
}

// FC layer with INT8 weights
void linear_int8(
    const float* input, float* output,
    const int8_t* weight, const float* scale, const float* bias,
    int in_feat, int out_feat
) {
    for (int o = 0; o < out_feat; ++o) {
        float sum = bias[o];
        const float s = scale[o];
        const int8_t* w = weight + o * in_feat;
        for (int i = 0; i < in_feat; ++i)
            sum += static_cast<float>(w[i]) * s * input[i];
        output[o] = sum;
    }
}

// Softmax
void softmax(float* x, int n) {
    const float mx = *std::max_element(x, x + n);
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        x[i] = std::exp(x[i] - mx);
        sum += x[i];
    }
    const float inv = 1.0f / sum;
    for (int i = 0; i < n; ++i)
        x[i] *= inv;
}

// Copy buffer region
void copy_buffer(const float* src, float* dst, int count) {
    std::memcpy(dst, src, count * sizeof(float));
}

} // anonymous namespace

// -----------------------------------------------
//  Main Inference Function
// -----------------------------------------------

int run_inference(
    const float* input, int input_length,
    float* output,
    float* buffer_a, float* buffer_b, float* buffer_skip, ui_update_handle_t *ui
) {
    if (!buffer_a || !buffer_b || !buffer_skip) return -1;
    if (input_length > 80000) return -1;

    LOG_INFO("AI", "Starting inference: %d samples", input_length);

    float* buf_in = buffer_a;
    float* buf_out = buffer_b;
    int cur_len = input_length;

    // Copy input
    std::memcpy(buf_in, input, input_length * sizeof(float));

    // Layer 0: initial_conv (1->16) [pool=4]
    {
        conv1d_int8(
            buf_in, buf_out,
            initial_conv_weight, initial_conv_scale, initial_conv_bias,
            initial_conv_config.in_channels, initial_conv_config.out_channels,
            cur_len, initial_conv_config.kernel_size,
            initial_conv_config.padding, initial_conv_config.stride,
            initial_conv_config.has_relu, cur_len
        );

        batchnorm1d(
            buf_out,
            initial_conv_bn_gamma, initial_conv_bn_beta,
            initial_conv_bn_mean, initial_conv_bn_var, initial_conv_bn_eps,
            initial_conv_config.out_channels, cur_len
        );

        maxpool1d(buf_out, initial_conv_config.out_channels, cur_len, initial_conv_config.pool_size, cur_len);

        std::swap(buf_in, buf_out);
    }
    LOG_INFO("AI", "After initial_conv: len=%d", cur_len);

    _lock_acquire(ui->lvgl_lock);
    lv_bar_set_value(ui->progress_bar, 30, LV_ANIM_ON);
    _lock_release(ui->lvgl_lock);

    // Layer 1: layer_0 (16->32) [residual] [proj] [pool=2]
    {
        const int skip_channels = 16;
        const int skip_len = cur_len;
        copy_buffer(buf_in, buffer_skip, skip_channels * skip_len);

        conv1d_int8(
            buf_in, buf_out,
            layer_0_weight, layer_0_scale, layer_0_bias,
            layer_0_config.in_channels, layer_0_config.out_channels,
            cur_len, layer_0_config.kernel_size,
            layer_0_config.padding, layer_0_config.stride,
            layer_0_config.has_relu, cur_len
        );

        batchnorm1d(
            buf_out,
            layer_0_bn_gamma, layer_0_bn_beta,
            layer_0_bn_mean, layer_0_bn_var, layer_0_bn_eps,
            layer_0_config.out_channels, cur_len
        );

        maxpool1d(buf_out, layer_0_config.out_channels, cur_len, layer_0_config.pool_size, cur_len);

        // Skip connection
        conv1x1_int8(
            buffer_skip, buf_in,
            layer_0_skip_weight, layer_0_skip_scale, layer_0_skip_bias,
            16, 32, skip_len
        );
        {
            int proj_len = skip_len;
            maxpool1d(buf_in, 32, proj_len, layer_0_config.pool_size, proj_len);
            residual_add(buf_out, buf_in, layer_0_config.out_channels * cur_len);
        }

        std::swap(buf_in, buf_out);
    }

    // Layer 2: layer_1 (32->32) [residual]
    {
        const int skip_channels = 32;
        const int skip_len = cur_len;
        copy_buffer(buf_in, buffer_skip, skip_channels * skip_len);

        conv1d_int8(
            buf_in, buf_out,
            layer_1_weight, layer_1_scale, layer_1_bias,
            layer_1_config.in_channels, layer_1_config.out_channels,
            cur_len, layer_1_config.kernel_size,
            layer_1_config.padding, layer_1_config.stride,
            layer_1_config.has_relu, cur_len
        );

        batchnorm1d(
            buf_out,
            layer_1_bn_gamma, layer_1_bn_beta,
            layer_1_bn_mean, layer_1_bn_var, layer_1_bn_eps,
            layer_1_config.out_channels, cur_len
        );

        // Skip connection
        residual_add(buf_out, buffer_skip, layer_1_config.out_channels * cur_len);

        std::swap(buf_in, buf_out);
    }

    // Layer 3: layer_2 (32->32) [residual]
    {
        const int skip_channels = 32;
        const int skip_len = cur_len;
        copy_buffer(buf_in, buffer_skip, skip_channels * skip_len);

        conv1d_int8(
            buf_in, buf_out,
            layer_2_weight, layer_2_scale, layer_2_bias,
            layer_2_config.in_channels, layer_2_config.out_channels,
            cur_len, layer_2_config.kernel_size,
            layer_2_config.padding, layer_2_config.stride,
            layer_2_config.has_relu, cur_len
        );

        batchnorm1d(
            buf_out,
            layer_2_bn_gamma, layer_2_bn_beta,
            layer_2_bn_mean, layer_2_bn_var, layer_2_bn_eps,
            layer_2_config.out_channels, cur_len
        );

        // Skip connection
        residual_add(buf_out, buffer_skip, layer_2_config.out_channels * cur_len);

        std::swap(buf_in, buf_out);
    }

    // Layer 4: layer_3 (32->32) [residual] [pool=2]
    {
        const int skip_channels = 32;
        const int skip_len = cur_len;
        copy_buffer(buf_in, buffer_skip, skip_channels * skip_len);

        conv1d_int8(
            buf_in, buf_out,
            layer_3_weight, layer_3_scale, layer_3_bias,
            layer_3_config.in_channels, layer_3_config.out_channels,
            cur_len, layer_3_config.kernel_size,
            layer_3_config.padding, layer_3_config.stride,
            layer_3_config.has_relu, cur_len
        );

        batchnorm1d(
            buf_out,
            layer_3_bn_gamma, layer_3_bn_beta,
            layer_3_bn_mean, layer_3_bn_var, layer_3_bn_eps,
            layer_3_config.out_channels, cur_len
        );

        maxpool1d(buf_out, layer_3_config.out_channels, cur_len, layer_3_config.pool_size, cur_len);

        // Skip connection
        {
            int skip_pool_len = skip_len;
            maxpool1d(buffer_skip, skip_channels, skip_pool_len, layer_3_config.pool_size, skip_pool_len);
            residual_add(buf_out, buffer_skip, layer_3_config.out_channels * cur_len);
        }

        std::swap(buf_in, buf_out);
    }

    // Layer 5: layer_4 (32->32) [residual]
    {
        const int skip_channels = 32;
        const int skip_len = cur_len;
        copy_buffer(buf_in, buffer_skip, skip_channels * skip_len);

        conv1d_int8(
            buf_in, buf_out,
            layer_4_weight, layer_4_scale, layer_4_bias,
            layer_4_config.in_channels, layer_4_config.out_channels,
            cur_len, layer_4_config.kernel_size,
            layer_4_config.padding, layer_4_config.stride,
            layer_4_config.has_relu, cur_len
        );

        batchnorm1d(
            buf_out,
            layer_4_bn_gamma, layer_4_bn_beta,
            layer_4_bn_mean, layer_4_bn_var, layer_4_bn_eps,
            layer_4_config.out_channels, cur_len
        );

        // Skip connection
        residual_add(buf_out, buffer_skip, layer_4_config.out_channels * cur_len);

        std::swap(buf_in, buf_out);
    }

    // Layer 6: layer_5 (32->32) [residual]
    {
        const int skip_channels = 32;
        const int skip_len = cur_len;
        copy_buffer(buf_in, buffer_skip, skip_channels * skip_len);

        conv1d_int8(
            buf_in, buf_out,
            layer_5_weight, layer_5_scale, layer_5_bias,
            layer_5_config.in_channels, layer_5_config.out_channels,
            cur_len, layer_5_config.kernel_size,
            layer_5_config.padding, layer_5_config.stride,
            layer_5_config.has_relu, cur_len
        );

        batchnorm1d(
            buf_out,
            layer_5_bn_gamma, layer_5_bn_beta,
            layer_5_bn_mean, layer_5_bn_var, layer_5_bn_eps,
            layer_5_config.out_channels, cur_len
        );

        // Skip connection
        residual_add(buf_out, buffer_skip, layer_5_config.out_channels * cur_len);

        std::swap(buf_in, buf_out);
    }

    // Layer 7: layer_6 (32->64) [residual] [proj] [pool=2]
    {
        const int skip_channels = 32;
        const int skip_len = cur_len;
        copy_buffer(buf_in, buffer_skip, skip_channels * skip_len);

        conv1d_int8(
            buf_in, buf_out,
            layer_6_weight, layer_6_scale, layer_6_bias,
            layer_6_config.in_channels, layer_6_config.out_channels,
            cur_len, layer_6_config.kernel_size,
            layer_6_config.padding, layer_6_config.stride,
            layer_6_config.has_relu, cur_len
        );

        batchnorm1d(
            buf_out,
            layer_6_bn_gamma, layer_6_bn_beta,
            layer_6_bn_mean, layer_6_bn_var, layer_6_bn_eps,
            layer_6_config.out_channels, cur_len
        );

        maxpool1d(buf_out, layer_6_config.out_channels, cur_len, layer_6_config.pool_size, cur_len);

        // Skip connection
        conv1x1_int8(
            buffer_skip, buf_in,
            layer_6_skip_weight, layer_6_skip_scale, layer_6_skip_bias,
            32, 64, skip_len
        );
        {
            int proj_len = skip_len;
            maxpool1d(buf_in, 64, proj_len, layer_6_config.pool_size, proj_len);
            residual_add(buf_out, buf_in, layer_6_config.out_channels * cur_len);
        }

        std::swap(buf_in, buf_out);
    }

    // Layer 8: layer_7 (64->64) [residual]
    {
        const int skip_channels = 64;
        const int skip_len = cur_len;
        copy_buffer(buf_in, buffer_skip, skip_channels * skip_len);

        conv1d_int8(
            buf_in, buf_out,
            layer_7_weight, layer_7_scale, layer_7_bias,
            layer_7_config.in_channels, layer_7_config.out_channels,
            cur_len, layer_7_config.kernel_size,
            layer_7_config.padding, layer_7_config.stride,
            layer_7_config.has_relu, cur_len
        );

        batchnorm1d(
            buf_out,
            layer_7_bn_gamma, layer_7_bn_beta,
            layer_7_bn_mean, layer_7_bn_var, layer_7_bn_eps,
            layer_7_config.out_channels, cur_len
        );

        // Skip connection
        residual_add(buf_out, buffer_skip, layer_7_config.out_channels * cur_len);

        std::swap(buf_in, buf_out);
    }
    LOG_INFO("AI", "After layer_7: len=%d", cur_len);

    _lock_acquire(ui->lvgl_lock);
    lv_bar_set_value(ui->progress_bar, 40, LV_ANIM_ON);
    _lock_release(ui->lvgl_lock);

    // Layer 9: layer_8 (64->64) [residual]
    {
        const int skip_channels = 64;
        const int skip_len = cur_len;
        copy_buffer(buf_in, buffer_skip, skip_channels * skip_len);

        conv1d_int8(
            buf_in, buf_out,
            layer_8_weight, layer_8_scale, layer_8_bias,
            layer_8_config.in_channels, layer_8_config.out_channels,
            cur_len, layer_8_config.kernel_size,
            layer_8_config.padding, layer_8_config.stride,
            layer_8_config.has_relu, cur_len
        );

        batchnorm1d(
            buf_out,
            layer_8_bn_gamma, layer_8_bn_beta,
            layer_8_bn_mean, layer_8_bn_var, layer_8_bn_eps,
            layer_8_config.out_channels, cur_len
        );

        // Skip connection
        residual_add(buf_out, buffer_skip, layer_8_config.out_channels * cur_len);

        std::swap(buf_in, buf_out);
    }

    // Layer 10: layer_9 (64->64) [residual]
    {
        const int skip_channels = 64;
        const int skip_len = cur_len;
        copy_buffer(buf_in, buffer_skip, skip_channels * skip_len);

        conv1d_int8(
            buf_in, buf_out,
            layer_9_weight, layer_9_scale, layer_9_bias,
            layer_9_config.in_channels, layer_9_config.out_channels,
            cur_len, layer_9_config.kernel_size,
            layer_9_config.padding, layer_9_config.stride,
            layer_9_config.has_relu, cur_len
        );

        batchnorm1d(
            buf_out,
            layer_9_bn_gamma, layer_9_bn_beta,
            layer_9_bn_mean, layer_9_bn_var, layer_9_bn_eps,
            layer_9_config.out_channels, cur_len
        );

        // Skip connection
        residual_add(buf_out, buffer_skip, layer_9_config.out_channels * cur_len);

        std::swap(buf_in, buf_out);
    }

    // Layer 11: layer_10 (64->64) [residual] [pool=2]
    {
        const int skip_channels = 64;
        const int skip_len = cur_len;
        copy_buffer(buf_in, buffer_skip, skip_channels * skip_len);

        conv1d_int8(
            buf_in, buf_out,
            layer_10_weight, layer_10_scale, layer_10_bias,
            layer_10_config.in_channels, layer_10_config.out_channels,
            cur_len, layer_10_config.kernel_size,
            layer_10_config.padding, layer_10_config.stride,
            layer_10_config.has_relu, cur_len
        );

        batchnorm1d(
            buf_out,
            layer_10_bn_gamma, layer_10_bn_beta,
            layer_10_bn_mean, layer_10_bn_var, layer_10_bn_eps,
            layer_10_config.out_channels, cur_len
        );

        maxpool1d(buf_out, layer_10_config.out_channels, cur_len, layer_10_config.pool_size, cur_len);

        // Skip connection
        {
            int skip_pool_len = skip_len;
            maxpool1d(buffer_skip, skip_channels, skip_pool_len, layer_10_config.pool_size, skip_pool_len);
            residual_add(buf_out, buffer_skip, layer_10_config.out_channels * cur_len);
        }

        std::swap(buf_in, buf_out);
    }

    // Layer 12: layer_11 (64->64) [residual]
    {
        const int skip_channels = 64;
        const int skip_len = cur_len;
        copy_buffer(buf_in, buffer_skip, skip_channels * skip_len);

        conv1d_int8(
            buf_in, buf_out,
            layer_11_weight, layer_11_scale, layer_11_bias,
            layer_11_config.in_channels, layer_11_config.out_channels,
            cur_len, layer_11_config.kernel_size,
            layer_11_config.padding, layer_11_config.stride,
            layer_11_config.has_relu, cur_len
        );

        batchnorm1d(
            buf_out,
            layer_11_bn_gamma, layer_11_bn_beta,
            layer_11_bn_mean, layer_11_bn_var, layer_11_bn_eps,
            layer_11_config.out_channels, cur_len
        );

        // Skip connection
        residual_add(buf_out, buffer_skip, layer_11_config.out_channels * cur_len);

        std::swap(buf_in, buf_out);
    }

    // Layer 13: layer_12 (64->64) [residual]
    {
        const int skip_channels = 64;
        const int skip_len = cur_len;
        copy_buffer(buf_in, buffer_skip, skip_channels * skip_len);

        conv1d_int8(
            buf_in, buf_out,
            layer_12_weight, layer_12_scale, layer_12_bias,
            layer_12_config.in_channels, layer_12_config.out_channels,
            cur_len, layer_12_config.kernel_size,
            layer_12_config.padding, layer_12_config.stride,
            layer_12_config.has_relu, cur_len
        );

        batchnorm1d(
            buf_out,
            layer_12_bn_gamma, layer_12_bn_beta,
            layer_12_bn_mean, layer_12_bn_var, layer_12_bn_eps,
            layer_12_config.out_channels, cur_len
        );

        // Skip connection
        residual_add(buf_out, buffer_skip, layer_12_config.out_channels * cur_len);

        std::swap(buf_in, buf_out);
    }

    // Layer 14: layer_13 (64->128) [residual] [proj] [pool=2]
    {
        const int skip_channels = 64;
        const int skip_len = cur_len;
        copy_buffer(buf_in, buffer_skip, skip_channels * skip_len);

        conv1d_int8(
            buf_in, buf_out,
            layer_13_weight, layer_13_scale, layer_13_bias,
            layer_13_config.in_channels, layer_13_config.out_channels,
            cur_len, layer_13_config.kernel_size,
            layer_13_config.padding, layer_13_config.stride,
            layer_13_config.has_relu, cur_len
        );

        batchnorm1d(
            buf_out,
            layer_13_bn_gamma, layer_13_bn_beta,
            layer_13_bn_mean, layer_13_bn_var, layer_13_bn_eps,
            layer_13_config.out_channels, cur_len
        );

        maxpool1d(buf_out, layer_13_config.out_channels, cur_len, layer_13_config.pool_size, cur_len);

        // Skip connection
        conv1x1_int8(
            buffer_skip, buf_in,
            layer_13_skip_weight, layer_13_skip_scale, layer_13_skip_bias,
            64, 128, skip_len
        );
        {
            int proj_len = skip_len;
            maxpool1d(buf_in, 128, proj_len, layer_13_config.pool_size, proj_len);
            residual_add(buf_out, buf_in, layer_13_config.out_channels * cur_len);
        }

        std::swap(buf_in, buf_out);
    }

    // Layer 15: layer_14 (128->128) [residual]
    {
        const int skip_channels = 128;
        const int skip_len = cur_len;
        copy_buffer(buf_in, buffer_skip, skip_channels * skip_len);

        conv1d_int8(
            buf_in, buf_out,
            layer_14_weight, layer_14_scale, layer_14_bias,
            layer_14_config.in_channels, layer_14_config.out_channels,
            cur_len, layer_14_config.kernel_size,
            layer_14_config.padding, layer_14_config.stride,
            layer_14_config.has_relu, cur_len
        );

        batchnorm1d(
            buf_out,
            layer_14_bn_gamma, layer_14_bn_beta,
            layer_14_bn_mean, layer_14_bn_var, layer_14_bn_eps,
            layer_14_config.out_channels, cur_len
        );

        // Skip connection
        residual_add(buf_out, buffer_skip, layer_14_config.out_channels * cur_len);

        std::swap(buf_in, buf_out);
    }

    // Layer 16: layer_15 (128->128) [residual]
    {
        const int skip_channels = 128;
        const int skip_len = cur_len;
        copy_buffer(buf_in, buffer_skip, skip_channels * skip_len);

        conv1d_int8(
            buf_in, buf_out,
            layer_15_weight, layer_15_scale, layer_15_bias,
            layer_15_config.in_channels, layer_15_config.out_channels,
            cur_len, layer_15_config.kernel_size,
            layer_15_config.padding, layer_15_config.stride,
            layer_15_config.has_relu, cur_len
        );

        batchnorm1d(
            buf_out,
            layer_15_bn_gamma, layer_15_bn_beta,
            layer_15_bn_mean, layer_15_bn_var, layer_15_bn_eps,
            layer_15_config.out_channels, cur_len
        );

        // Skip connection
        residual_add(buf_out, buffer_skip, layer_15_config.out_channels * cur_len);

        std::swap(buf_in, buf_out);
    }
    LOG_INFO("AI", "After layer_15: len=%d", cur_len);

    _lock_acquire(ui->lvgl_lock);
    lv_bar_set_value(ui->progress_bar, 50, LV_ANIM_ON);
    _lock_release(ui->lvgl_lock);

    // Layer 17: layer_16 (128->128) [residual]
    {
        const int skip_channels = 128;
        const int skip_len = cur_len;
        copy_buffer(buf_in, buffer_skip, skip_channels * skip_len);

        conv1d_int8(
            buf_in, buf_out,
            layer_16_weight, layer_16_scale, layer_16_bias,
            layer_16_config.in_channels, layer_16_config.out_channels,
            cur_len, layer_16_config.kernel_size,
            layer_16_config.padding, layer_16_config.stride,
            layer_16_config.has_relu, cur_len
        );

        batchnorm1d(
            buf_out,
            layer_16_bn_gamma, layer_16_bn_beta,
            layer_16_bn_mean, layer_16_bn_var, layer_16_bn_eps,
            layer_16_config.out_channels, cur_len
        );

        // Skip connection
        residual_add(buf_out, buffer_skip, layer_16_config.out_channels * cur_len);

        std::swap(buf_in, buf_out);
    }

    // Layer 18: layer_17 (128->128) [residual] [pool=2]
    {
        const int skip_channels = 128;
        const int skip_len = cur_len;
        copy_buffer(buf_in, buffer_skip, skip_channels * skip_len);

        conv1d_int8(
            buf_in, buf_out,
            layer_17_weight, layer_17_scale, layer_17_bias,
            layer_17_config.in_channels, layer_17_config.out_channels,
            cur_len, layer_17_config.kernel_size,
            layer_17_config.padding, layer_17_config.stride,
            layer_17_config.has_relu, cur_len
        );

        batchnorm1d(
            buf_out,
            layer_17_bn_gamma, layer_17_bn_beta,
            layer_17_bn_mean, layer_17_bn_var, layer_17_bn_eps,
            layer_17_config.out_channels, cur_len
        );

        maxpool1d(buf_out, layer_17_config.out_channels, cur_len, layer_17_config.pool_size, cur_len);

        // Skip connection
        {
            int skip_pool_len = skip_len;
            maxpool1d(buffer_skip, skip_channels, skip_pool_len, layer_17_config.pool_size, skip_pool_len);
            residual_add(buf_out, buffer_skip, layer_17_config.out_channels * cur_len);
        }

        std::swap(buf_in, buf_out);
    }

    // Layer 19: layer_18 (128->128) [residual]
    {
        const int skip_channels = 128;
        const int skip_len = cur_len;
        copy_buffer(buf_in, buffer_skip, skip_channels * skip_len);

        conv1d_int8(
            buf_in, buf_out,
            layer_18_weight, layer_18_scale, layer_18_bias,
            layer_18_config.in_channels, layer_18_config.out_channels,
            cur_len, layer_18_config.kernel_size,
            layer_18_config.padding, layer_18_config.stride,
            layer_18_config.has_relu, cur_len
        );

        batchnorm1d(
            buf_out,
            layer_18_bn_gamma, layer_18_bn_beta,
            layer_18_bn_mean, layer_18_bn_var, layer_18_bn_eps,
            layer_18_config.out_channels, cur_len
        );

        // Skip connection
        residual_add(buf_out, buffer_skip, layer_18_config.out_channels * cur_len);

        std::swap(buf_in, buf_out);
    }

    // Layer 20: layer_19 (128->128) [residual]
    {
        const int skip_channels = 128;
        const int skip_len = cur_len;
        copy_buffer(buf_in, buffer_skip, skip_channels * skip_len);

        conv1d_int8(
            buf_in, buf_out,
            layer_19_weight, layer_19_scale, layer_19_bias,
            layer_19_config.in_channels, layer_19_config.out_channels,
            cur_len, layer_19_config.kernel_size,
            layer_19_config.padding, layer_19_config.stride,
            layer_19_config.has_relu, cur_len
        );

        batchnorm1d(
            buf_out,
            layer_19_bn_gamma, layer_19_bn_beta,
            layer_19_bn_mean, layer_19_bn_var, layer_19_bn_eps,
            layer_19_config.out_channels, cur_len
        );

        // Skip connection
        residual_add(buf_out, buffer_skip, layer_19_config.out_channels * cur_len);

        std::swap(buf_in, buf_out);
    }

    // Layer 21: layer_20 (128->256) [residual] [proj] [pool=2]
    {
        const int skip_channels = 128;
        const int skip_len = cur_len;
        copy_buffer(buf_in, buffer_skip, skip_channels * skip_len);

        conv1d_int8(
            buf_in, buf_out,
            layer_20_weight, layer_20_scale, layer_20_bias,
            layer_20_config.in_channels, layer_20_config.out_channels,
            cur_len, layer_20_config.kernel_size,
            layer_20_config.padding, layer_20_config.stride,
            layer_20_config.has_relu, cur_len
        );

        batchnorm1d(
            buf_out,
            layer_20_bn_gamma, layer_20_bn_beta,
            layer_20_bn_mean, layer_20_bn_var, layer_20_bn_eps,
            layer_20_config.out_channels, cur_len
        );

        maxpool1d(buf_out, layer_20_config.out_channels, cur_len, layer_20_config.pool_size, cur_len);

        // Skip connection
        conv1x1_int8(
            buffer_skip, buf_in,
            layer_20_skip_weight, layer_20_skip_scale, layer_20_skip_bias,
            128, 256, skip_len
        );
        {
            int proj_len = skip_len;
            maxpool1d(buf_in, 256, proj_len, layer_20_config.pool_size, proj_len);
            residual_add(buf_out, buf_in, layer_20_config.out_channels * cur_len);
        }

        std::swap(buf_in, buf_out);
    }

    // Layer 22: layer_21 (256->256) [residual]
    {
        const int skip_channels = 256;
        const int skip_len = cur_len;
        copy_buffer(buf_in, buffer_skip, skip_channels * skip_len);

        conv1d_int8(
            buf_in, buf_out,
            layer_21_weight, layer_21_scale, layer_21_bias,
            layer_21_config.in_channels, layer_21_config.out_channels,
            cur_len, layer_21_config.kernel_size,
            layer_21_config.padding, layer_21_config.stride,
            layer_21_config.has_relu, cur_len
        );

        batchnorm1d(
            buf_out,
            layer_21_bn_gamma, layer_21_bn_beta,
            layer_21_bn_mean, layer_21_bn_var, layer_21_bn_eps,
            layer_21_config.out_channels, cur_len
        );

        // Skip connection
        residual_add(buf_out, buffer_skip, layer_21_config.out_channels * cur_len);

        std::swap(buf_in, buf_out);
    }

    // Layer 23: layer_22 (256->256) [residual]
    {
        const int skip_channels = 256;
        const int skip_len = cur_len;
        copy_buffer(buf_in, buffer_skip, skip_channels * skip_len);

        conv1d_int8(
            buf_in, buf_out,
            layer_22_weight, layer_22_scale, layer_22_bias,
            layer_22_config.in_channels, layer_22_config.out_channels,
            cur_len, layer_22_config.kernel_size,
            layer_22_config.padding, layer_22_config.stride,
            layer_22_config.has_relu, cur_len
        );

        batchnorm1d(
            buf_out,
            layer_22_bn_gamma, layer_22_bn_beta,
            layer_22_bn_mean, layer_22_bn_var, layer_22_bn_eps,
            layer_22_config.out_channels, cur_len
        );

        // Skip connection
        residual_add(buf_out, buffer_skip, layer_22_config.out_channels * cur_len);

        std::swap(buf_in, buf_out);
    }

    // Layer 24: layer_23 (256->256) [residual]
    {
        const int skip_channels = 256;
        const int skip_len = cur_len;
        copy_buffer(buf_in, buffer_skip, skip_channels * skip_len);

        conv1d_int8(
            buf_in, buf_out,
            layer_23_weight, layer_23_scale, layer_23_bias,
            layer_23_config.in_channels, layer_23_config.out_channels,
            cur_len, layer_23_config.kernel_size,
            layer_23_config.padding, layer_23_config.stride,
            layer_23_config.has_relu, cur_len
        );

        batchnorm1d(
            buf_out,
            layer_23_bn_gamma, layer_23_bn_beta,
            layer_23_bn_mean, layer_23_bn_var, layer_23_bn_eps,
            layer_23_config.out_channels, cur_len
        );

        // Skip connection
        residual_add(buf_out, buffer_skip, layer_23_config.out_channels * cur_len);

        std::swap(buf_in, buf_out);
    }
    LOG_INFO("AI", "After layer_23: len=%d", cur_len);

    _lock_acquire(ui->lvgl_lock);
    lv_bar_set_value(ui->progress_bar, 60, LV_ANIM_ON);
    _lock_release(ui->lvgl_lock);

    // Layer 25: layer_24 (256->256) [residual] [pool=2]
    {
        const int skip_channels = 256;
        const int skip_len = cur_len;
        copy_buffer(buf_in, buffer_skip, skip_channels * skip_len);

        conv1d_int8(
            buf_in, buf_out,
            layer_24_weight, layer_24_scale, layer_24_bias,
            layer_24_config.in_channels, layer_24_config.out_channels,
            cur_len, layer_24_config.kernel_size,
            layer_24_config.padding, layer_24_config.stride,
            layer_24_config.has_relu, cur_len
        );

        batchnorm1d(
            buf_out,
            layer_24_bn_gamma, layer_24_bn_beta,
            layer_24_bn_mean, layer_24_bn_var, layer_24_bn_eps,
            layer_24_config.out_channels, cur_len
        );

        maxpool1d(buf_out, layer_24_config.out_channels, cur_len, layer_24_config.pool_size, cur_len);

        // Skip connection
        {
            int skip_pool_len = skip_len;
            maxpool1d(buffer_skip, skip_channels, skip_pool_len, layer_24_config.pool_size, skip_pool_len);
            residual_add(buf_out, buffer_skip, layer_24_config.out_channels * cur_len);
        }

        std::swap(buf_in, buf_out);
    }

    // Layer 26: layer_25 (256->256) [residual]
    {
        const int skip_channels = 256;
        const int skip_len = cur_len;
        copy_buffer(buf_in, buffer_skip, skip_channels * skip_len);

        conv1d_int8(
            buf_in, buf_out,
            layer_25_weight, layer_25_scale, layer_25_bias,
            layer_25_config.in_channels, layer_25_config.out_channels,
            cur_len, layer_25_config.kernel_size,
            layer_25_config.padding, layer_25_config.stride,
            layer_25_config.has_relu, cur_len
        );

        batchnorm1d(
            buf_out,
            layer_25_bn_gamma, layer_25_bn_beta,
            layer_25_bn_mean, layer_25_bn_var, layer_25_bn_eps,
            layer_25_config.out_channels, cur_len
        );

        // Skip connection
        residual_add(buf_out, buffer_skip, layer_25_config.out_channels * cur_len);

        std::swap(buf_in, buf_out);
    }

    // Layer 27: layer_26 (256->256) [residual]
    {
        const int skip_channels = 256;
        const int skip_len = cur_len;
        copy_buffer(buf_in, buffer_skip, skip_channels * skip_len);

        conv1d_int8(
            buf_in, buf_out,
            layer_26_weight, layer_26_scale, layer_26_bias,
            layer_26_config.in_channels, layer_26_config.out_channels,
            cur_len, layer_26_config.kernel_size,
            layer_26_config.padding, layer_26_config.stride,
            layer_26_config.has_relu, cur_len
        );

        batchnorm1d(
            buf_out,
            layer_26_bn_gamma, layer_26_bn_beta,
            layer_26_bn_mean, layer_26_bn_var, layer_26_bn_eps,
            layer_26_config.out_channels, cur_len
        );

        // Skip connection
        residual_add(buf_out, buffer_skip, layer_26_config.out_channels * cur_len);

        std::swap(buf_in, buf_out);
    }

    // Layer 28: layer_27 (256->256) [residual] [pool=2]
    {
        const int skip_channels = 256;
        const int skip_len = cur_len;
        copy_buffer(buf_in, buffer_skip, skip_channels * skip_len);

        conv1d_int8(
            buf_in, buf_out,
            layer_27_weight, layer_27_scale, layer_27_bias,
            layer_27_config.in_channels, layer_27_config.out_channels,
            cur_len, layer_27_config.kernel_size,
            layer_27_config.padding, layer_27_config.stride,
            layer_27_config.has_relu, cur_len
        );

        batchnorm1d(
            buf_out,
            layer_27_bn_gamma, layer_27_bn_beta,
            layer_27_bn_mean, layer_27_bn_var, layer_27_bn_eps,
            layer_27_config.out_channels, cur_len
        );

        maxpool1d(buf_out, layer_27_config.out_channels, cur_len, layer_27_config.pool_size, cur_len);

        // Skip connection
        {
            int skip_pool_len = skip_len;
            maxpool1d(buffer_skip, skip_channels, skip_pool_len, layer_27_config.pool_size, skip_pool_len);
            residual_add(buf_out, buffer_skip, layer_27_config.out_channels * cur_len);
        }

        std::swap(buf_in, buf_out);
    }

    // Layer 29: layer_28 (256->256) [residual]
    {
        const int skip_channels = 256;
        const int skip_len = cur_len;
        copy_buffer(buf_in, buffer_skip, skip_channels * skip_len);

        conv1d_int8(
            buf_in, buf_out,
            layer_28_weight, layer_28_scale, layer_28_bias,
            layer_28_config.in_channels, layer_28_config.out_channels,
            cur_len, layer_28_config.kernel_size,
            layer_28_config.padding, layer_28_config.stride,
            layer_28_config.has_relu, cur_len
        );

        batchnorm1d(
            buf_out,
            layer_28_bn_gamma, layer_28_bn_beta,
            layer_28_bn_mean, layer_28_bn_var, layer_28_bn_eps,
            layer_28_config.out_channels, cur_len
        );

        // Skip connection
        residual_add(buf_out, buffer_skip, layer_28_config.out_channels * cur_len);

        std::swap(buf_in, buf_out);
    }

    // Layer 30: layer_29 (256->256) [residual]
    {
        const int skip_channels = 256;
        const int skip_len = cur_len;
        copy_buffer(buf_in, buffer_skip, skip_channels * skip_len);

        conv1d_int8(
            buf_in, buf_out,
            layer_29_weight, layer_29_scale, layer_29_bias,
            layer_29_config.in_channels, layer_29_config.out_channels,
            cur_len, layer_29_config.kernel_size,
            layer_29_config.padding, layer_29_config.stride,
            layer_29_config.has_relu, cur_len
        );

        batchnorm1d(
            buf_out,
            layer_29_bn_gamma, layer_29_bn_beta,
            layer_29_bn_mean, layer_29_bn_var, layer_29_bn_eps,
            layer_29_config.out_channels, cur_len
        );

        // Skip connection
        residual_add(buf_out, buffer_skip, layer_29_config.out_channels * cur_len);

        std::swap(buf_in, buf_out);
    }

    // Layer 31: layer_30 (256->256) [residual] [pool=2]
    {
        const int skip_channels = 256;
        const int skip_len = cur_len;
        copy_buffer(buf_in, buffer_skip, skip_channels * skip_len);

        conv1d_int8(
            buf_in, buf_out,
            layer_30_weight, layer_30_scale, layer_30_bias,
            layer_30_config.in_channels, layer_30_config.out_channels,
            cur_len, layer_30_config.kernel_size,
            layer_30_config.padding, layer_30_config.stride,
            layer_30_config.has_relu, cur_len
        );

        batchnorm1d(
            buf_out,
            layer_30_bn_gamma, layer_30_bn_beta,
            layer_30_bn_mean, layer_30_bn_var, layer_30_bn_eps,
            layer_30_config.out_channels, cur_len
        );

        maxpool1d(buf_out, layer_30_config.out_channels, cur_len, layer_30_config.pool_size, cur_len);

        // Skip connection
        {
            int skip_pool_len = skip_len;
            maxpool1d(buffer_skip, skip_channels, skip_pool_len, layer_30_config.pool_size, skip_pool_len);
            residual_add(buf_out, buffer_skip, layer_30_config.out_channels * cur_len);
        }

        std::swap(buf_in, buf_out);
    }

    // Layer 32: layer_31 (256->256) [residual]
    {
        const int skip_channels = 256;
        const int skip_len = cur_len;
        copy_buffer(buf_in, buffer_skip, skip_channels * skip_len);

        conv1d_int8(
            buf_in, buf_out,
            layer_31_weight, layer_31_scale, layer_31_bias,
            layer_31_config.in_channels, layer_31_config.out_channels,
            cur_len, layer_31_config.kernel_size,
            layer_31_config.padding, layer_31_config.stride,
            layer_31_config.has_relu, cur_len
        );

        batchnorm1d(
            buf_out,
            layer_31_bn_gamma, layer_31_bn_beta,
            layer_31_bn_mean, layer_31_bn_var, layer_31_bn_eps,
            layer_31_config.out_channels, cur_len
        );

        // Skip connection
        residual_add(buf_out, buffer_skip, layer_31_config.out_channels * cur_len);

        std::swap(buf_in, buf_out);
    }
    LOG_INFO("AI", "After layer_31: len=%d", cur_len);

    _lock_acquire(ui->lvgl_lock);
    lv_bar_set_value(ui->progress_bar, 70, LV_ANIM_ON);
    _lock_release(ui->lvgl_lock);

    // Layer 33: layer_32 (256->256) [residual]
    {
        const int skip_channels = 256;
        const int skip_len = cur_len;
        copy_buffer(buf_in, buffer_skip, skip_channels * skip_len);

        conv1d_int8(
            buf_in, buf_out,
            layer_32_weight, layer_32_scale, layer_32_bias,
            layer_32_config.in_channels, layer_32_config.out_channels,
            cur_len, layer_32_config.kernel_size,
            layer_32_config.padding, layer_32_config.stride,
            layer_32_config.has_relu, cur_len
        );

        batchnorm1d(
            buf_out,
            layer_32_bn_gamma, layer_32_bn_beta,
            layer_32_bn_mean, layer_32_bn_var, layer_32_bn_eps,
            layer_32_config.out_channels, cur_len
        );

        // Skip connection
        residual_add(buf_out, buffer_skip, layer_32_config.out_channels * cur_len);

        std::swap(buf_in, buf_out);
    }


    LOG_INFO("AI", "Before GAP: len=%d", cur_len);

    _lock_acquire(ui->lvgl_lock);
    lv_bar_set_value(ui->progress_bar, 80, LV_ANIM_ON);
    _lock_release(ui->lvgl_lock);

    // Global Average Pooling
    std::array<float, 256> pooled{};
    global_avg_pool1d(buf_in, pooled.data(), 256, cur_len);

    // Fully Connected
    std::array<float, NUM_CLASSES> logits{};
    linear_int8(pooled.data(), logits.data(),
                fc_weight, fc_scale, fc_bias,
                FC_IN, FC_OUT);

    LOG_INFO("AI", "Logits: %f, %f", logits[0], logits[1]);

    // Softmax
    softmax(logits.data(), NUM_CLASSES);

    LOG_INFO("AI", "Probabilities: %f, %f", logits[0], logits[1]);

    output[0] = logits[0];  // P(no murmur)
    output[1] = logits[1];  // P(murmur)

    return 0;
}

} // namespace heart_inference

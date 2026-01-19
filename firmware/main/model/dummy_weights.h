#ifndef DUMMY_MODEL_WEIGHTS_H
#define DUMMY_MODEL_WEIGHTS_H

#include <stdint.h>

#define INPUT_CHANNELS 1
#define NUM_CLASSES 2
#define SAMPLE_RATE 2000
#define MAX_INPUT_LENGTH 60000  // 30 seconds * 2000 Hz

#define FC_IN_FEATURES 256
#define FC_OUT_FEATURES 2

// Layer configuration
typedef struct {
    int in_channels;
    int out_channels;
    int kernel_size;
    int padding;
    int stride;
    int has_relu;
    int has_pool;
    int pool_size;
} Conv1dConfig;

static const Conv1dConfig initial_conv_config = {
    .in_channels = 1,
    .out_channels = 16,
    .kernel_size = 7,
    .padding = 3,
    .stride = 1,
    .has_relu = 1,
    .has_pool = 1,
    .pool_size = 4
};
static const int8_t initial_conv_weight[112] = {0};
static const float initial_conv_scale[16] = {0};
static const float initial_conv_bias[16] = {0};

// ---------- layer_0 ----------
static const int8_t layer_0_weight[1536] = {0};
static const float layer_0_scale[32] = {0};
static const float layer_0_bias[32] = {0};
static const Conv1dConfig layer_0_config = { .in_channels = 16, .out_channels = 32, .kernel_size = 3,
    .padding = 1, .stride = 1, .has_relu = 1, .has_pool = 1, .pool_size = 2 };

// ---------- layer_1 ----------
static const int8_t layer_1_weight[3072] = {0};
static const float layer_1_scale[32] = {0};
static const float layer_1_bias[32] = {0};
static const Conv1dConfig layer_1_config = { .in_channels = 32, .out_channels = 32, .kernel_size = 3, 
    .padding = 1, .stride = 1, .has_relu = 1, .has_pool = 0, .pool_size = 2 };

// ---------- layer_2 ----------
static const int8_t layer_2_weight[3072] = {0};
static const float layer_2_scale[32] = {0};
static const float layer_2_bias[32] = {0};
static const Conv1dConfig layer_2_config = { .in_channels = 32, .out_channels = 32, .kernel_size = 3, .padding = 1, .stride = 1, .has_relu = 1, .has_pool = 0, .pool_size = 2 };

// ---------- layer_3 ----------
static const int8_t layer_3_weight[3072] = {0};
static const float layer_3_scale[32] = {0};
static const float layer_3_bias[32] = {0};
static const Conv1dConfig layer_3_config = { .in_channels = 32, .out_channels = 32, .kernel_size = 3, .padding = 1, .stride = 1, .has_relu = 1, .has_pool = 0, .pool_size = 2 };

// ---------- layer_4 ----------
static const int8_t layer_4_weight[3072] = {0};
static const float layer_4_scale[32] = {0};
static const float layer_4_bias[32] = {0};
static const Conv1dConfig layer_4_config = { .in_channels = 32, .out_channels = 32, .kernel_size = 3, .padding = 1, .stride = 1, .has_relu = 1, .has_pool = 0, .pool_size = 2 };

// ---------- layer_5 ----------
static const int8_t layer_5_weight[3072] = {0};
static const float layer_5_scale[32] = {0};
static const float layer_5_bias[32] = {0};
static const Conv1dConfig layer_5_config = { .in_channels = 32, .out_channels = 32, .kernel_size = 3, 
    .padding = 1, .stride = 1, .has_relu = 1, .has_pool = 0, .pool_size = 2 };

// ---------- layer_6 ---------- (Transition 32 -> 64)
static const int8_t layer_6_weight[6144] = {0};
static const float layer_6_scale[64] = {0};
static const float layer_6_bias[64] = {0};
static const Conv1dConfig layer_6_config = { .in_channels = 32, .out_channels = 64, .kernel_size = 3, 
    .padding = 1, .stride = 1, .has_relu = 1, .has_pool = 0, .pool_size = 2 };

// ---------- layer_7 ----------
static const int8_t layer_7_weight[12288] = {0};
static const float layer_7_scale[64] = {0};
static const float layer_7_bias[64] = {0};
static const Conv1dConfig layer_7_config = { .in_channels = 64, .out_channels = 64, .kernel_size = 3, 
    .padding = 1, .stride = 1, .has_relu = 1, .has_pool = 0, .pool_size = 2 };

// ---------- layer_8 ----------
static const int8_t layer_8_weight[12288] = {0};
static const float layer_8_scale[64] = {0};
static const float layer_8_bias[64] = {0};
static const Conv1dConfig layer_8_config = { .in_channels = 64, .out_channels = 64, .kernel_size = 3, 
    .padding = 1, .stride = 1, .has_relu = 1, .has_pool = 0, .pool_size = 2 };

// ---------- layer_9 ----------
static const int8_t layer_9_weight[12288] = {0};
static const float layer_9_scale[64] = {0};
static const float layer_9_bias[64] = {0};
static const Conv1dConfig layer_9_config = { .in_channels = 64, .out_channels = 64, .kernel_size = 3, 
    .padding = 1, .stride = 1, .has_relu = 1, .has_pool = 0, .pool_size = 2 };

// ---------- layer_10 ----------
static const int8_t layer_10_weight[12288] = {0};
static const float layer_10_scale[64] = {0};
static const float layer_10_bias[64] = {0};
static const Conv1dConfig layer_10_config = { .in_channels = 64, .out_channels = 64, .kernel_size = 3, 
    .padding = 1, .stride = 1, .has_relu = 1, .has_pool = 0, .pool_size = 2 };

// ---------- layer_11 ----------
static const int8_t layer_11_weight[12288] = {0};
static const float layer_11_scale[64] = {0};
static const float layer_11_bias[64] = {0};
static const Conv1dConfig layer_11_config = { .in_channels = 64, .out_channels = 64, .kernel_size = 3, 
    .padding = 1, .stride = 1, .has_relu = 1, .has_pool = 0, .pool_size = 2 };

// ---------- layer_12 ----------
static const int8_t layer_12_weight[12288] = {0};
static const float layer_12_scale[64] = {0};
static const float layer_12_bias[64] = {0};
static const Conv1dConfig layer_12_config = { .in_channels = 64, .out_channels = 64, .kernel_size = 3, 
    .padding = 1, .stride = 1, .has_relu = 1, .has_pool = 0, .pool_size = 2 };

// ---------- layer_13 ---------- (Transition 64 -> 128)
static const int8_t layer_13_weight[24576] = {0};
static const float layer_13_scale[128] = {0};
static const float layer_13_bias[128] = {0};
static const Conv1dConfig layer_13_config = { .in_channels = 64, .out_channels = 128, .kernel_size = 3, 
    .padding = 1, .stride = 1, .has_relu = 1, .has_pool = 0, .pool_size = 2 };

// ---------- layer_14 ----------
static const int8_t layer_14_weight[49152] = {0};
static const float layer_14_scale[128] = {0};
static const float layer_14_bias[128] = {0};
static const Conv1dConfig layer_14_config = { .in_channels = 128, .out_channels = 128, .kernel_size = 3, 
    .padding = 1, .stride = 1, .has_relu = 1, .has_pool = 0, .pool_size = 2 };

// ---------- layer_15 ----------
static const int8_t layer_15_weight[49152] = {0};
static const float layer_15_scale[128] = {0};
static const float layer_15_bias[128] = {0};
static const Conv1dConfig layer_15_config = { .in_channels = 128, .out_channels = 128, .kernel_size = 3, 
    .padding = 1, .stride = 1, .has_relu = 1, .has_pool = 0, .pool_size = 2 };

// ---------- layer_16 ----------
static const int8_t layer_16_weight[49152] = {0};
static const float layer_16_scale[128] = {0};
static const float layer_16_bias[128] = {0};
static const Conv1dConfig layer_16_config = { .in_channels = 128, .out_channels = 128, .kernel_size = 3, 
    .padding = 1, .stride = 1, .has_relu = 1, .has_pool = 0, .pool_size = 2 };

// ---------- layer_17 ----------
static const int8_t layer_17_weight[49152] = {0};
static const float layer_17_scale[128] = {0};
static const float layer_17_bias[128] = {0};
static const Conv1dConfig layer_17_config = { .in_channels = 128, .out_channels = 128, .kernel_size = 3, 
    .padding = 1, .stride = 1, .has_relu = 1, .has_pool = 0, .pool_size = 2 };

// ---------- layer_18 ----------
static const int8_t layer_18_weight[49152] = {0};
static const float layer_18_scale[128] = {0};
static const float layer_18_bias[128] = {0};
static const Conv1dConfig layer_18_config = { .in_channels = 128, .out_channels = 128, .kernel_size = 3, 
    .padding = 1, .stride = 1, .has_relu = 1, .has_pool = 0, .pool_size = 2 };

// ---------- layer_19 ----------
static const int8_t layer_19_weight[49152] = {0};
static const float layer_19_scale[128] = {0};
static const float layer_19_bias[128] = {0};
static const Conv1dConfig layer_19_config = { .in_channels = 128, .out_channels = 128, .kernel_size = 3, 
    .padding = 1, .stride = 1, .has_relu = 1, .has_pool = 0, .pool_size = 2 };

// ---------- layer_20 ---------- (Transition 128 -> 256)
static const int8_t layer_20_weight[98304] = {0};
static const float layer_20_scale[256] = {0};
static const float layer_20_bias[256] = {0};
static const Conv1dConfig layer_20_config = { .in_channels = 128, .out_channels = 256, .kernel_size = 3, 
    .padding = 1, .stride = 1, .has_relu = 1, .has_pool = 0, .pool_size = 2 };

// ---------- layer_21 ----------
static const int8_t layer_21_weight[196608] = {0};
static const float layer_21_scale[256] = {0};
static const float layer_21_bias[256] = {0};
static const Conv1dConfig layer_21_config = { .in_channels = 256, .out_channels = 256, .kernel_size = 3, 
    .padding = 1, .stride = 1, .has_relu = 1, .has_pool = 0, .pool_size = 2 };

// ---------- layer_22 ----------
static const int8_t layer_22_weight[196608] = {0};
static const float layer_22_scale[256] = {0};
static const float layer_22_bias[256] = {0};
static const Conv1dConfig layer_22_config = { .in_channels = 256, .out_channels = 256, .kernel_size = 3, 
    .padding = 1, .stride = 1, .has_relu = 1, .has_pool = 0, .pool_size = 2 };

// ---------- layer_23 ----------
static const int8_t layer_23_weight[196608] = {0};
static const float layer_23_scale[256] = {0};
static const float layer_23_bias[256] = {0};
static const Conv1dConfig layer_23_config = { .in_channels = 256, .out_channels = 256, .kernel_size = 3, 
    .padding = 1, .stride = 1, .has_relu = 1, .has_pool = 0, .pool_size = 2 };

// ---------- layer_24 ----------
static const int8_t layer_24_weight[196608] = {0};
static const float layer_24_scale[256] = {0};
static const float layer_24_bias[256] = {0};
static const Conv1dConfig layer_24_config = { .in_channels = 256, .out_channels = 256, .kernel_size = 3, 
    .padding = 1, .stride = 1, .has_relu = 1, .has_pool = 0, .pool_size = 2 };

// ---------- layer_25 ----------
static const int8_t layer_25_weight[196608] = {0};
static const float layer_25_scale[256] = {0};
static const float layer_25_bias[256] = {0};
static const Conv1dConfig layer_25_config = { .in_channels = 256, .out_channels = 256, .kernel_size = 3, 
    .padding = 1, .stride = 1, .has_relu = 1, .has_pool = 0, .pool_size = 2 };

// ---------- layer_26 ----------
static const int8_t layer_26_weight[196608] = {0};
static const float layer_26_scale[256] = {0};
static const float layer_26_bias[256] = {0};
static const Conv1dConfig layer_26_config = { .in_channels = 256, .out_channels = 256, .kernel_size = 3, 
    .padding = 1, .stride = 1, .has_relu = 1, .has_pool = 0, .pool_size = 2 };

// ---------- layer_27 ----------
static const int8_t layer_27_weight[196608] = {0};
static const float layer_27_scale[256] = {0};
static const float layer_27_bias[256] = {0};
static const Conv1dConfig layer_27_config = { .in_channels = 256, .out_channels = 256, .kernel_size = 3, 
    .padding = 1, .stride = 1, .has_relu = 1, .has_pool = 0, .pool_size = 2 };

// ---------- layer_28 ----------
static const int8_t layer_28_weight[196608] = {0};
static const float layer_28_scale[256] = {0};
static const float layer_28_bias[256] = {0};
static const Conv1dConfig layer_28_config = { .in_channels = 256, .out_channels = 256, .kernel_size = 3, 
    .padding = 1, .stride = 1, .has_relu = 1, .has_pool = 0, .pool_size = 2 };

// ---------- layer_29 ----------
static const int8_t layer_29_weight[196608] = {0};
static const float layer_29_scale[256] = {0};
static const float layer_29_bias[256] = {0};
static const Conv1dConfig layer_29_config = { .in_channels = 256, .out_channels = 256, .kernel_size = 3, 
    .padding = 1, .stride = 1, .has_relu = 1, .has_pool = 0, .pool_size = 2 };

// ---------- layer_30 ----------
static const int8_t layer_30_weight[196608] = {0};
static const float layer_30_scale[256] = {0};
static const float layer_30_bias[256] = {0};
static const Conv1dConfig layer_30_config = { .in_channels = 256, .out_channels = 256, .kernel_size = 3, 
    .padding = 1, .stride = 1, .has_relu = 1, .has_pool = 0, .pool_size = 2 };

// ---------- layer_31 ----------
static const int8_t layer_31_weight[196608] = {0};
static const float layer_31_scale[256] = {0};
static const float layer_31_bias[256] = {0};
static const Conv1dConfig layer_31_config = { .in_channels = 256, .out_channels = 256, .kernel_size = 3, 
    .padding = 1, .stride = 1, .has_relu = 1, .has_pool = 0, .pool_size = 2 };

// ---------- layer_32 ----------
static const int8_t layer_32_weight[196608] = {0};
static const float layer_32_scale[256] = {0};
static const float layer_32_bias[256] = {0};
static const Conv1dConfig layer_32_config = { .in_channels = 256, .out_channels = 256, .kernel_size = 3, 
    .padding = 1, .stride = 1, .has_relu = 1, .has_pool = 0, .pool_size = 2 };

static const int8_t fc_weight[512] = {0};
static const float fc_scale[2] = {
    8.45421338e-04f, 7.53340370e-04f
};

static const float fc_bias[2] = {
    -1.83131814e-01f, 1.79806054e-01f
};

#endif // DUMMY_MODEL_WEIGHTS_H

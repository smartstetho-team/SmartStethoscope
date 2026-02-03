#include "dsp_ml_setup.h"
#include "mic_setup.h"

#include "cmn.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "esp_dsp.h"

static const char *ML_CLASSIFICATION_TASK_TAG = "ML_CLASSIFICATION_TASK";

// Filter coefficients (generated from python)
static float state_s1[2] = {0}; 
static float state_s2[2] = {0};

// static float state_notch[2] = {0};
// Original SciPy: a1 = -1.997864, a2 = 0.998429
// ESP-DSP Stable: a1 = 1.997864, a2 = -0.998429
// static float coeffs_notch[5] = {0.999215f, -1.997864f, 0.999215f, -1.997864f, 0.998429f};

// static float coeffs_s1[5] = {0.002081f, 0.004161f, 0.002081f, -1.889040f, 0.899332f};
// static float coeffs_s2[5] = {1.000000f, -2.000000f, 1.000000f, -1.972482f, 0.973183f};

// static float coeffs_s1[5] = {0.037725f, 0.075450f, 0.037725f, -1.394347f, 0.549097f};
// static float coeffs_s2[5] = {1.000000f, -2.000000f, 1.000000f, -1.966975f, 0.967573f};

// static float coeffs_s1[5] = {0.004067f, 0.008135f, 0.004067f, -1.834586f, 0.852926f};
// static float coeffs_s2[5] = {1.000000f, -2.000000f, 1.000000f, -1.970018f, 0.970699f};

// static float coeffs_s1[5] = {0.001460f, 0.002921f, 0.001460f, -1.913178f, 0.933004f};
// static float coeffs_s2[5] = {1.000000f, -2.000000f, 1.000000f, -1.951878f, 0.959133f};

// static float coeffs_s1[5] = {0.000375f, 0.000750f, 0.000375f, -1.960858f, 0.965904f};
// static float coeffs_s2[5] = {1.000000f, -2.000000f, 1.000000f, -1.977537f, 0.979370f};

// static float coeffs_s1[5] = {0.000727f, 0.001454f, 0.000727f, -1.940787f, 0.945451f};
// static float coeffs_s2[5] = {1.000000f, -2.000000f, 1.000000f, -1.977869f, 0.978576f};

// static float coeffs_s1[5] = {0.001460f, 0.002921f, 0.001460f, -1.910222f, 0.917972f};
// static float coeffs_s2[5] = {1.000000f, -2.000000f, 1.000000f, -1.974132f, 0.974839f};

static float coeffs_s1[5] = {0.002081f, 0.004161f, 0.002081f, -1.889040f, 0.899332f};
static float coeffs_s2[5] = {1.000000f, -2.000000f, 1.000000f, -1.972482f, 0.973183f};

void ml_classification_task(void *dsp_ml_parameters)
{
    ESP_LOGI(ML_CLASSIFICATION_TASK_TAG, "Starting ML classification task");
    
    task_params* params = (task_params*)dsp_ml_parameters;
    uint8_t* master_audio_buffer = params->master_audio_buffer;
    float* filtered_audio_buffer = params->filtered_audio_buffer;
    EventGroupHandle_t event_group_handle = params->event_group_handle;

    while (1)
    {
        xEventGroupWaitBits(event_group_handle,
                            AUDIO_RECORDING_DONE_BIT | ML_CLASSIFICATION_START_BIT, 
                            pdFALSE, pdTRUE, portMAX_DELAY);

        ESP_LOGI(ML_CLASSIFICATION_TASK_TAG, "Filtering audio..");

        memset(state_s1, 0, sizeof(state_s1));
        memset(state_s2, 0, sizeof(state_s2));

        uint32_t adc_sum = 0;
        for (int i = 0; i < MASTER_AUDIO_BUFFER_SIZE; i += ADC_OUTPUT_LEN)
        {
            adc_digi_output_data_t *sample = (adc_digi_output_data_t*)&master_audio_buffer[i];
            adc_sum += sample->type2.data;
        }
        
        params->audio_dc_offset = (float)(adc_sum/NUM_OF_SAMPLES);
        ESP_LOGI(ML_CLASSIFICATION_TASK_TAG, "DC Offset: %.2f", params->audio_dc_offset);

        for (int i = 0; i < MASTER_AUDIO_BUFFER_SIZE; i += ADC_OUTPUT_LEN)
        {
            adc_digi_output_data_t *sample = (adc_digi_output_data_t*)&master_audio_buffer[i];
            uint16_t adc_val = (uint16_t)sample->type2.data;

            float centered_val = (float)adc_val - params->audio_dc_offset;
            float normalized_val = centered_val/2048.0f;

            if (normalized_val > 1.0f)
            {
                normalized_val = 1.0f;
            }
            else if (normalized_val < -1.0f)
            {
                normalized_val = -1.0f;
            }

            filtered_audio_buffer[i/ADC_OUTPUT_LEN] = normalized_val;
        }

        // dsps_biquad_f32(filtered_audio_buffer, filtered_audio_buffer, NUM_OF_SAMPLES, coeffs_notch, state_notch);

        // Apply bandpass filter to get frequencies between 30-150 Hz
        dsps_biquad_f32(filtered_audio_buffer, filtered_audio_buffer, NUM_OF_SAMPLES, coeffs_s1, state_s1);
        dsps_biquad_f32(filtered_audio_buffer, filtered_audio_buffer, NUM_OF_SAMPLES, coeffs_s2, state_s2);

        ESP_LOGI(ML_CLASSIFICATION_TASK_TAG, "Audio filtering complete.");
        
        // TODO: Add different filtering modes?
        // TODO: Classify using MFCC

        xEventGroupSetBits(event_group_handle, ML_CLASSIFICATION_END_BIT);
        xEventGroupClearBits(event_group_handle, ML_CLASSIFICATION_START_BIT);
        vTaskDelay(pdMS_TO_TICKS(100));
    }
}
#include "dsp_ml_setup.h"
#include "mic_setup.h"
#include "heart_inference.h"

#include "cmn.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "esp_dsp.h"
#include "lvgl.h"

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
    float* inference_buffer_a = params->inference_buffer_a;
    float* inference_buffer_b = params->inference_buffer_b;

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

        // // Apply bandpass filter to get frequencies between 30-150 Hz
        // dsps_biquad_f32(filtered_audio_buffer, filtered_audio_buffer, NUM_OF_SAMPLES, coeffs_s1, state_s1);
        // dsps_biquad_f32(filtered_audio_buffer, filtered_audio_buffer, NUM_OF_SAMPLES, coeffs_s2, state_s2);

        // ESP_LOGI(ML_CLASSIFICATION_TASK_TAG, "Audio filtering complete.");
        
        // float output[2];
        // heart_inference::run_inference(filtered_audio_buffer, NUM_OF_SAMPLES, 
        // output, inference_buffer_a, inference_buffer_b );

        // // heart_inference(filtered_audio_buffer, 80000, output, inference_buffer_a, inference_buffer_b);
        // ESP_LOGI(ML_CLASSIFICATION_TASK_TAG, "Output 1: %.2f", output[0]);
        // ESP_LOGI(ML_CLASSIFICATION_TASK_TAG, "Output 2: %.2f", output[1]);

        _lock_acquire(&params->lcd_params.lvgl_api_lock);

        lv_obj_t * active_scr = lv_screen_active();
        lv_obj_clean(active_scr);

        // 1. Status Label
        lv_obj_t *proc_label = lv_label_create(active_scr);
        lv_label_set_text(proc_label, "Filtering Audio..."); // Initial stage text
        lv_obj_set_style_text_font(proc_label, &lv_font_montserrat_22, 0); 
        lv_obj_set_style_text_color(proc_label, lv_color_hex(0x000000), 0);
        lv_obj_align(proc_label, LV_ALIGN_CENTER, 0, -50);

        // 2. Progress Bar
        lv_obj_t *bar = lv_bar_create(active_scr);
        lv_obj_set_size(bar, 220, 15);
        lv_obj_center(bar);
        lv_obj_set_style_bg_color(bar, lv_color_hex(0x1A1A1A), LV_PART_MAIN);
        lv_obj_set_style_bg_color(bar, lv_palette_main(LV_PALETTE_CYAN), LV_PART_INDICATOR);
        
        // Start at 0
        lv_bar_set_value(bar, 0, LV_ANIM_OFF);
        
        _lock_release(&params->lcd_params.lvgl_api_lock);

        // --- STEP 1: FILTERING (0% -> 20%) ---
        dsps_biquad_f32(filtered_audio_buffer, filtered_audio_buffer, NUM_OF_SAMPLES, coeffs_s1, state_s1);
        dsps_biquad_f32(filtered_audio_buffer, filtered_audio_buffer, NUM_OF_SAMPLES, coeffs_s2, state_s2);

        _lock_acquire(&params->lcd_params.lvgl_api_lock);
        lv_bar_set_value(bar, 20, LV_ANIM_ON);
        lv_label_set_text(proc_label, "Extracting...");
        _lock_release(&params->lcd_params.lvgl_api_lock);

        // --- STEP 2: INFERENCE (20% -> 90%) ---
        // This is the "Minute-Long" part. 
        float output[2];
        heart_inference::run_inference(filtered_audio_buffer, NUM_OF_SAMPLES, 
                                       output, inference_buffer_a, inference_buffer_b);

        _lock_acquire(&params->lcd_params.lvgl_api_lock);
        lv_bar_set_value(bar, 90, LV_ANIM_ON);
        lv_label_set_text(proc_label, "Finalizing...");
        _lock_release(&params->lcd_params.lvgl_api_lock);

        // --- STEP 3: RESULT (100%) ---
        vTaskDelay(pdMS_TO_TICKS(500)); // Small pause to let the user see "Finalizing"

        _lock_acquire(&params->lcd_params.lvgl_api_lock);
        lv_obj_add_flag(bar, LV_OBJ_FLAG_HIDDEN); 
        
        if (output[1] > output[0]) {
            lv_label_set_text(proc_label, "Abnormal");
        } else {
            lv_label_set_text(proc_label, "Normal");
        }
        lv_obj_align(proc_label, LV_ALIGN_CENTER, 0, 0); 
        _lock_release(&params->lcd_params.lvgl_api_lock);

        xEventGroupSetBits(event_group_handle, ML_CLASSIFICATION_END_BIT);
        xEventGroupClearBits(event_group_handle, ML_CLASSIFICATION_START_BIT);
        vTaskDelay(pdMS_TO_TICKS(100));
    }
}
#include "dsp_ml_setup.h"
#include "mic_setup.h"
#include "heart_inference.h"

#include "cmn.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "esp_dsp.h"
#include "lvgl.h"
#include <cmath>

static const char *ML_CLASSIFICATION_TASK_TAG = "ML_CLASSIFICATION_TASK";

// Filter coefficients (generated from python)
static float state_s1[2] = {0}; 
static float state_s2[2] = {0};

// static float state_notch[2] = {0};
// Original SciPy: a1 = -1.997864, a2 = 0.998429
// ESP-DSP Stable: a1 = 1.997864, a2 = -0.998429
// static float coeffs_notch[5] = {0.999215f, -1.997864f, 0.999215f, -1.997864f, 0.998429f};

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
    float* inference_buffer_skip = params->inference_buffer_skip;

    EventGroupHandle_t event_group_handle = params->event_group_handle;

    // ADDED: Extract characteristic for BLE communication
    NimBLECharacteristic* pAudioDataChar = params->pAudioDataChar;

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
            float normalized_val = (centered_val/2048.0f) * DIGITAL_GAIN;

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

        _lock_acquire(&params->lcd_params.lvgl_api_lock);

        lv_obj_t * active_scr = lv_screen_active();
        lv_obj_clean(active_scr);

        // Status Label
        lv_obj_t *proc_label = lv_label_create(active_scr);
        lv_label_set_text(proc_label, "Filtering Audio...");
        lv_obj_set_style_text_font(proc_label, &lv_font_montserrat_22, 0); 
        lv_obj_align(proc_label, LV_ALIGN_CENTER, 0, -60);

        // Progress Bar
        lv_obj_t *bar = lv_bar_create(active_scr);
        lv_obj_set_size(bar, 200, 12);
        lv_obj_align(bar, LV_ALIGN_CENTER, 0, -20);
        lv_bar_set_value(bar, 5, LV_ANIM_OFF);

        // Pulsing Heart
        lv_obj_t *heart = lv_label_create(active_scr);
        lv_label_set_text(heart, LV_SYMBOL_VOLUME_MID);
        lv_obj_set_style_text_color(heart, lv_palette_main(LV_PALETTE_RED), 0);
        lv_obj_set_style_text_font(heart, &lv_font_montserrat_30, 0); // Large and visible
        lv_obj_align(heart, LV_ALIGN_CENTER, 0, 50);

        // "Lub-Dub" Heartbeat Animation
        lv_anim_t a;
        lv_anim_init(&a);
        lv_anim_set_var(&a, heart);
        lv_anim_set_values(&a, 256, 380);      // 256 = 100% scale
        lv_anim_set_time(&a, 200);             // Quick "Thump"
        lv_anim_set_playback_time(&a, 400);    // Gentle return
        lv_anim_set_repeat_count(&a, LV_ANIM_REPEAT_INFINITE);
        
        // Create a more organic, bouncy "thump"
        lv_anim_set_path_cb(&a, lv_anim_path_overshoot); 
        
        lv_anim_set_exec_cb(&a, [](void * var, int32_t v) {
            lv_obj_set_style_transform_scale((lv_obj_t *)var, v, 0);
        });
        lv_anim_start(&a);
        _lock_release(&params->lcd_params.lvgl_api_lock);

        // --- STEP 1: FILTERING (20% -> 40%) ---
        dsps_biquad_f32(filtered_audio_buffer, filtered_audio_buffer, NUM_OF_SAMPLES, coeffs_s1, state_s1);
        dsps_biquad_f32(filtered_audio_buffer, filtered_audio_buffer, NUM_OF_SAMPLES, coeffs_s2, state_s2);

        // Calculate BPM metric
        float max_peak = 0;
        for (int i = 0; i < NUM_OF_SAMPLES; i++) 
        {
            float abs_val = fabsf(filtered_audio_buffer[i]);
            if (abs_val > max_peak) 
            { 
                max_peak = abs_val;
            }
        }

        int final_bpm = 0;
        int beat_count = 0;

        // If the room is silent, max_peak will be tiny. 
        // Ignore everything below a 0.05 amplitude floor (adjust based on your mic).
        const float NOISE_FLOOR = 0.05f; 

        if (max_peak < NOISE_FLOOR) 
        {
            ESP_LOGW(ML_CLASSIFICATION_TASK_TAG, "Silence detected (Peak: %.4f). Setting BPM to 0.", max_peak);
            final_bpm = 0;
        }
        else 
        {
            // Dynamic Thresholding
            float threshold = max_peak * 0.65f; 
            
            // Refractory Period (Cooldown)
            // Ignore everything for 300ms after a peak to skip the S2 "Dub" and noise.
            int refractory_samples = (int)(SAMPLE_FREQ_HZ * 0.30); 
            int last_beat_index = -refractory_samples; 

            for (int i = 0; i < NUM_OF_SAMPLES; i++) 
            {
                float current_val = fabsf(filtered_audio_buffer[i]);

                if (current_val > threshold && (i - last_beat_index) > refractory_samples) 
                {
                    beat_count++;
                    last_beat_index = i;
                }
            }

            float total_seconds = (float)NUM_OF_SAMPLES / SAMPLE_FREQ_HZ;
            final_bpm = (int)(beat_count * (60.0f / total_seconds));

            // Sanity Check
            // If the math results in 250+ BPM, it's likely electronic noise, not a heart.
            if (final_bpm > 220 || final_bpm < 40) 
            {
                ESP_LOGW(ML_CLASSIFICATION_TASK_TAG, "BPM Out of Range (%d). Masking as 0.", final_bpm);
                final_bpm = 0;
            }
        }

        ESP_LOGI(ML_CLASSIFICATION_TASK_TAG, "Calculated BPM: %d", final_bpm);

        vTaskDelay(pdMS_TO_TICKS(1000));

        ui_update_handle_t ui_handle = {
            .progress_bar = bar,
            .status_label = proc_label,
            .lvgl_lock = &params->lcd_params.lvgl_api_lock
        };

        _lock_acquire(&params->lcd_params.lvgl_api_lock);
        lv_bar_set_value(bar, 20, LV_ANIM_ON);
        lv_label_set_text(proc_label, "Analyzing...");
        _lock_release(&params->lcd_params.lvgl_api_lock);

        // --- STEP 2: INFERENCE (40% -> 90%) ---
        float output[2];
        heart_inference::run_inference(filtered_audio_buffer, NUM_OF_SAMPLES, 
                                       output, inference_buffer_a, inference_buffer_b, 
                                       inference_buffer_skip, &ui_handle);

        _lock_acquire(&params->lcd_params.lvgl_api_lock);
        lv_bar_set_value(bar, 90, LV_ANIM_ON);
        lv_label_set_text(proc_label, "Finalizing...");
        _lock_release(&params->lcd_params.lvgl_api_lock);

        vTaskDelay(pdMS_TO_TICKS(500));

        // --- STEP 3: RESULT & CLEANUP ---
        _lock_acquire(&params->lcd_params.lvgl_api_lock);
        lv_obj_clean(active_scr); // Automatically stops the heart animation
        
        if (output[1] > MURMUR_THRESHOLD)
        {
            lv_obj_t *err_icon = lv_label_create(active_scr);
            lv_label_set_text(err_icon, LV_SYMBOL_WARNING);
            lv_obj_align(err_icon, LV_ALIGN_CENTER, 0, -50);

            lv_obj_t *end_label = lv_label_create(active_scr);
            lv_label_set_text(end_label, "Abnormal");
            lv_obj_set_style_text_font(end_label, &lv_font_montserrat_30, 0);
            lv_obj_align(end_label, LV_ALIGN_CENTER, 0, 0);

            lv_obj_t *end_sub1 = lv_label_create(active_scr);
            lv_label_set_text(end_sub1, "BPM: --");
            lv_obj_align(end_sub1, LV_ALIGN_CENTER, 0, 45);
            
            lv_obj_t *end_sub2 = lv_label_create(active_scr);
            lv_label_set_text(end_sub2, "Check CardioScope App");
            lv_obj_align(end_sub2, LV_ALIGN_CENTER, 0, 75);
        } 
        else 
        {
            lv_obj_t *ok_icon = lv_label_create(active_scr);
            lv_label_set_text(ok_icon, LV_SYMBOL_OK);
            lv_obj_align(ok_icon, LV_ALIGN_CENTER, 0, -50);

            lv_obj_t *end_label = lv_label_create(active_scr);
            lv_label_set_text(end_label, "Normal");
            lv_obj_set_style_text_font(end_label, &lv_font_montserrat_30, 0);
            lv_obj_align(end_label, LV_ALIGN_CENTER, 0, 0);

            lv_obj_t *end_sub = lv_label_create(active_scr);
            lv_label_set_text_fmt(end_sub, "BPM: %d", final_bpm);
            lv_obj_align(end_sub, LV_ALIGN_CENTER, 0, 45);

            lv_obj_t *restart_label = lv_label_create(active_scr);
            lv_label_set_text(restart_label, LV_SYMBOL_REFRESH " Press button to record again");
            lv_obj_set_style_text_font(restart_label, &lv_font_montserrat_16, 0);
            lv_obj_align(restart_label, LV_ALIGN_CENTER, 0, 75);
        }
        _lock_release(&params->lcd_params.lvgl_api_lock);

        // Set BPM & Abnormal/Normal (to be sent via BLE)
        params->calculated_bpm = final_bpm;
        params->classification_result = (output[1] > MURMUR_THRESHOLD) ? 1 : 0;

        // ADDED: Formatting and sending metadata packet [Header, Status, BPM, Padding]
        uint8_t metadata[4];
        metadata[0] = 0xFF;                          // SYNC BYTE: Tells phone this is metadata
        metadata[1] = params->classification_result; // 1 for Abnormal, 0 for Normal
        metadata[2] = (uint8_t)params->calculated_bpm;
        metadata[3] = 0x00;                          // Padding

        ESP_LOGI(ML_CLASSIFICATION_TASK_TAG, ">>> BLE TRIAGE SENT: Status=%d, BPM=%d", 
                 metadata[1], metadata[2]);
        
        // Notify the phone characteristic immediately
        pAudioDataChar->setValue(metadata, 4);
        pAudioDataChar->notify();

        xEventGroupSetBits(event_group_handle, ML_CLASSIFICATION_END_BIT);
        xEventGroupClearBits(event_group_handle, ML_CLASSIFICATION_START_BIT);
        vTaskDelay(pdMS_TO_TICKS(100));
    }
}
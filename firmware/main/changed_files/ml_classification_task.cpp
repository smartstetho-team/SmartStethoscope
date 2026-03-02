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
        // lv_anim_delete(heart, NULL);

        lv_obj_t * active_scr = lv_screen_active();
        // lv_anim_delete(NULL, NULL);
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
        lv_bar_set_value(bar, 20, LV_ANIM_OFF);

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
        // 1. Dynamic Thresholding
        float max_peak = 0;
        for (int i = 0; i < NUM_OF_SAMPLES; i++) 
        {
            if (abs(filtered_audio_buffer[i]) > max_peak) 
            {
                max_peak = abs(filtered_audio_buffer[i]);
            }
        }

        // Set threshold at 65% of max to catch S1 but ignore background hiss
        float threshold = max_peak * 0.65f; 
        int beat_count = 0;

        // 2. Refractory Period (Cooldown)
        // A human heart won't beat faster than 220 BPM (~270ms per beat).
        // We'll ignore everything for 300ms after a peak to skip the S2 "Dub".
        int refractory_samples = (int)(SAMPLE_FREQ_HZ * 0.30); 
        int last_beat_index = -refractory_samples; 

        for (int i = 0; i < NUM_OF_SAMPLES; i++) 
        {
            float current_val = abs(filtered_audio_buffer[i]);

            // Trigger if: Signal > Threshold AND we are outside the cooldown window
            if (current_val > threshold && (i - last_beat_index) > refractory_samples) 
            {
                beat_count++;
                last_beat_index = i; // Reset cooldown
            }
        }

        // 3. Final Conversion
        float total_seconds = (float)NUM_OF_SAMPLES / SAMPLE_FREQ_HZ;
        int final_bpm = (int)(beat_count * (60.0f / total_seconds));

        ESP_LOGI(ML_CLASSIFICATION_TASK_TAG, "Calculated BPM: %d", final_bpm);

        vTaskDelay(pdMS_TO_TICKS(1000));

        _lock_acquire(&params->lcd_params.lvgl_api_lock);
        lv_bar_set_value(bar, 40, LV_ANIM_ON);
        lv_label_set_text(proc_label, "Analyzing...");
        _lock_release(&params->lcd_params.lvgl_api_lock);

        // --- STEP 2: INFERENCE (40% -> 90%) ---
        float output[2];
        heart_inference::run_inference(filtered_audio_buffer, NUM_OF_SAMPLES, 
                                       output, inference_buffer_a, inference_buffer_b, 
                                       inference_buffer_skip);

        ESP_LOGI(ML_CLASSIFICATION_TASK_TAG, "Stack HWM: %u bytes remaining", 
        uxTaskGetStackHighWaterMark(NULL) * sizeof(StackType_t));

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
        }
        _lock_release(&params->lcd_params.lvgl_api_lock);

        xEventGroupSetBits(event_group_handle, ML_CLASSIFICATION_END_BIT);
        xEventGroupClearBits(event_group_handle, ML_CLASSIFICATION_START_BIT);
        vTaskDelay(pdMS_TO_TICKS(100));
    }
}

// ================================ TESTING SCRIPT ===========================================
// #include "dsp_ml_setup.h"
// #include "mic_setup.h"
// #include "heart_inference.h"
// #include "wav_loader.h"

// #include "cmn.h"
// #include "freertos/FreeRTOS.h"
// #include "freertos/task.h"
// #include "esp_log.h"
// #include "esp_dsp.h"
// #include "lvgl.h"

// #include <stdio.h>
// #include <stdlib.h>
// #include <string.h>

// static const char *ML_CLASSIFICATION_TASK_TAG = "ML_CLASSIFICATION_TASK";

// // Filter coefficients (generated from python)
// static float state_s1[2] = {0}; 
// static float state_s2[2] = {0};

// static float coeffs_s1[5] = {0.002081f, 0.004161f, 0.002081f, -1.889040f, 0.899332f};
// static float coeffs_s2[5] = {1.000000f, -2.000000f, 1.000000f, -1.972482f, 0.973183f};

// // Path to WAV file — update this to your actual file path
// #define WAV_FILE_PATH "./13918_PV.wav"

// /**
//  * @brief Decode WAV PCM data into normalized float samples using wav_loader.
//  * @return Number of samples written, or -1 on error.
//  */
// static int load_wav_samples(const uint8_t *wav_data, uint32_t wav_len,
//                             float *output_buffer, int max_samples)
// {
//     wav_info_t info;
//     if (parse_wav(wav_data, wav_len, &info) != 0) {
//         ESP_LOGE(ML_CLASSIFICATION_TASK_TAG, "Failed to parse WAV");
//         return -1;
//     }

//     ESP_LOGI(ML_CLASSIFICATION_TASK_TAG, "WAV: %lu Hz, %u-bit, %u ch",
//              info.sample_rate, info.bits_per_sample, info.num_channels);

//     const uint8_t *data = wav_data + info.data_offset;
//     int bytes_per_sample = info.bits_per_sample / 8;
//     int frame_size = bytes_per_sample * info.num_channels;
//     int total_frames = info.data_size / frame_size;
//     int samples_to_read = (total_frames < max_samples) ? total_frames : max_samples;

//     for (int i = 0; i < samples_to_read; i++) {
//         const uint8_t *frame = data + (i * frame_size);

//         if (info.bits_per_sample == 16) {
//             int16_t raw;
//             memcpy(&raw, frame, sizeof(int16_t));
//             output_buffer[i] = (float)raw / 32768.0f;
//         } else if (info.bits_per_sample == 8) {
//             output_buffer[i] = ((float)frame[0] - 128.0f) / 128.0f;
//         } else {
//             ESP_LOGE(ML_CLASSIFICATION_TASK_TAG, "Unsupported bit depth: %u", info.bits_per_sample);
//             return -1;
//         }
//     }

//     return samples_to_read;
// }

// void ml_classification_task(void *dsp_ml_parameters)
// {
//     ESP_LOGI(ML_CLASSIFICATION_TASK_TAG, "Starting ML classification task");
    
//     task_params* params = (task_params*)dsp_ml_parameters;
//     float* filtered_audio_buffer = params->filtered_audio_buffer;
//     float* inference_buffer_a = params->inference_buffer_a;
//     float* inference_buffer_b = params->inference_buffer_b;
//     float* inference_buffer_skip = params->inference_buffer_skip;

//     EventGroupHandle_t event_group_handle = params->event_group_handle;

//     while (1)
//     {
//         xEventGroupWaitBits(event_group_handle,
//                             AUDIO_RECORDING_DONE_BIT | ML_CLASSIFICATION_START_BIT, 
//                             pdFALSE, pdTRUE, portMAX_DELAY);

//         ESP_LOGI(ML_CLASSIFICATION_TASK_TAG, "Loading WAV file...");

//         memset(state_s1, 0, sizeof(state_s1));
//         memset(state_s2, 0, sizeof(state_s2));

//         // --- Read WAV file from filesystem ---
//         FILE *f = fopen(WAV_FILE_PATH, "rb");
//         if (!f) {
//             ESP_LOGE(ML_CLASSIFICATION_TASK_TAG, "Failed to open WAV: %s", WAV_FILE_PATH);
//             xEventGroupSetBits(event_group_handle, ML_CLASSIFICATION_END_BIT);
//             xEventGroupClearBits(event_group_handle, ML_CLASSIFICATION_START_BIT);
//             continue;
//         }

//         fseek(f, 0, SEEK_END);
//         uint32_t wav_len = ftell(f);
//         fseek(f, 0, SEEK_SET);

//         uint8_t *wav_buf = (uint8_t *)malloc(wav_len);
//         if (!wav_buf) {
//             ESP_LOGE(ML_CLASSIFICATION_TASK_TAG, "Failed to allocate %lu bytes for WAV", wav_len);
//             fclose(f);
//             xEventGroupSetBits(event_group_handle, ML_CLASSIFICATION_END_BIT);
//             xEventGroupClearBits(event_group_handle, ML_CLASSIFICATION_START_BIT);
//             continue;
//         }

//         fread(wav_buf, 1, wav_len, f);
//         fclose(f);

//         int samples_read = load_wav_samples(wav_buf, wav_len,
//                                             filtered_audio_buffer, NUM_OF_SAMPLES);
//         free(wav_buf);

//         if (samples_read <= 0) {
//             ESP_LOGE(ML_CLASSIFICATION_TASK_TAG, "WAV load failed, skipping inference");
//             xEventGroupSetBits(event_group_handle, ML_CLASSIFICATION_END_BIT);
//             xEventGroupClearBits(event_group_handle, ML_CLASSIFICATION_START_BIT);
//             continue;
//         }

//         // Zero-pad if WAV had fewer samples than expected
//         if (samples_read < NUM_OF_SAMPLES) {
//             ESP_LOGW(ML_CLASSIFICATION_TASK_TAG, "WAV had %d samples, expected %d. Zero-padding.",
//                      samples_read, NUM_OF_SAMPLES);
//             memset(&filtered_audio_buffer[samples_read], 0, 
//                    (NUM_OF_SAMPLES - samples_read) * sizeof(float));
//         }

//         // Apply digital gain and clipping
//         for (int i = 0; i < NUM_OF_SAMPLES; i++) {
//             filtered_audio_buffer[i] *= DIGITAL_GAIN;
//             if (filtered_audio_buffer[i] > 1.0f) filtered_audio_buffer[i] = 1.0f;
//             else if (filtered_audio_buffer[i] < -1.0f) filtered_audio_buffer[i] = -1.0f;
//         }

//         _lock_acquire(&params->lcd_params.lvgl_api_lock);

//         lv_obj_t * active_scr = lv_screen_active();
//         lv_obj_clean(active_scr);

//         // Status Label
//         lv_obj_t *proc_label = lv_label_create(active_scr);
//         lv_label_set_text(proc_label, "Filtering Audio...");
//         lv_obj_set_style_text_font(proc_label, &lv_font_montserrat_22, 0); 
//         lv_obj_align(proc_label, LV_ALIGN_CENTER, 0, -60);

//         // Progress Bar
//         lv_obj_t *bar = lv_bar_create(active_scr);
//         lv_obj_set_size(bar, 200, 12);
//         lv_obj_align(bar, LV_ALIGN_CENTER, 0, -20);
//         lv_bar_set_value(bar, 20, LV_ANIM_OFF);

//         // Pulsing Heart
//         lv_obj_t *heart = lv_label_create(active_scr);
//         lv_label_set_text(heart, LV_SYMBOL_VOLUME_MID);
//         lv_obj_set_style_text_color(heart, lv_palette_main(LV_PALETTE_RED), 0);
//         lv_obj_set_style_text_font(heart, &lv_font_montserrat_30, 0);
//         lv_obj_align(heart, LV_ALIGN_CENTER, 0, 50);

//         // "Lub-Dub" Heartbeat Animation
//         lv_anim_t a;
//         lv_anim_init(&a);
//         lv_anim_set_var(&a, heart);
//         lv_anim_set_values(&a, 256, 380);
//         lv_anim_set_time(&a, 200);
//         lv_anim_set_playback_time(&a, 400);
//         lv_anim_set_repeat_count(&a, LV_ANIM_REPEAT_INFINITE);
//         lv_anim_set_path_cb(&a, lv_anim_path_overshoot); 
        
//         lv_anim_set_exec_cb(&a, [](void * var, int32_t v) {
//             lv_obj_set_style_transform_scale((lv_obj_t *)var, v, 0);
//         });
//         lv_anim_start(&a);
//         _lock_release(&params->lcd_params.lvgl_api_lock);

//         // --- STEP 1: FILTERING (20% -> 40%) ---
//         dsps_biquad_f32(filtered_audio_buffer, filtered_audio_buffer, NUM_OF_SAMPLES, coeffs_s1, state_s1);
//         dsps_biquad_f32(filtered_audio_buffer, filtered_audio_buffer, NUM_OF_SAMPLES, coeffs_s2, state_s2);

//         // Calculate BPM metric
//         float max_peak = 0;
//         for (int i = 0; i < NUM_OF_SAMPLES; i++) 
//         {
//             if (abs(filtered_audio_buffer[i]) > max_peak) 
//             {
//                 max_peak = abs(filtered_audio_buffer[i]);
//             }
//         }

//         float threshold = max_peak * 0.65f; 
//         int beat_count = 0;

//         int refractory_samples = (int)(SAMPLE_FREQ_HZ * 0.30); 
//         int last_beat_index = -refractory_samples; 

//         for (int i = 0; i < NUM_OF_SAMPLES; i++) 
//         {
//             float current_val = abs(filtered_audio_buffer[i]);

//             if (current_val > threshold && (i - last_beat_index) > refractory_samples) 
//             {
//                 beat_count++;
//                 last_beat_index = i;
//             }
//         }

//         float total_seconds = (float)NUM_OF_SAMPLES / SAMPLE_FREQ_HZ;
//         int final_bpm = (int)(beat_count * (60.0f / total_seconds));

//         ESP_LOGI(ML_CLASSIFICATION_TASK_TAG, "Calculated BPM: %d", final_bpm);

//         vTaskDelay(pdMS_TO_TICKS(1000));

//         _lock_acquire(&params->lcd_params.lvgl_api_lock);
//         lv_bar_set_value(bar, 40, LV_ANIM_ON);
//         lv_label_set_text(proc_label, "Analyzing...");
//         _lock_release(&params->lcd_params.lvgl_api_lock);

//         // --- STEP 2: INFERENCE (40% -> 90%) ---
//         float output[2];
//         heart_inference::run_inference(filtered_audio_buffer, NUM_OF_SAMPLES, 
//                                        output, inference_buffer_a, inference_buffer_b, 
//                                        inference_buffer_skip);

//         _lock_acquire(&params->lcd_params.lvgl_api_lock);
//         lv_bar_set_value(bar, 90, LV_ANIM_ON);
//         lv_label_set_text(proc_label, "Finalizing...");
//         _lock_release(&params->lcd_params.lvgl_api_lock);

//         vTaskDelay(pdMS_TO_TICKS(500));

//         // --- STEP 3: RESULT & CLEANUP ---
//         _lock_acquire(&params->lcd_params.lvgl_api_lock);
//         lv_obj_clean(active_scr);
        
//         if (output[1] > MURMUR_THRESHOLD)
//         {
//             lv_obj_t *err_icon = lv_label_create(active_scr);
//             lv_label_set_text(err_icon, LV_SYMBOL_WARNING);
//             lv_obj_align(err_icon, LV_ALIGN_CENTER, 0, -50);

//             lv_obj_t *end_label = lv_label_create(active_scr);
//             lv_label_set_text(end_label, "Abnormal");
//             lv_obj_set_style_text_font(end_label, &lv_font_montserrat_30, 0);
//             lv_obj_align(end_label, LV_ALIGN_CENTER, 0, 0);

//             lv_obj_t *end_sub1 = lv_label_create(active_scr);
//             lv_label_set_text(end_sub1, "BPM: --");
//             lv_obj_align(end_sub1, LV_ALIGN_CENTER, 0, 45);
            
//             lv_obj_t *end_sub2 = lv_label_create(active_scr);
//             lv_label_set_text(end_sub2, "Check CardioScope App");
//             lv_obj_align(end_sub2, LV_ALIGN_CENTER, 0, 75);
//         } 
//         else 
//         {
//             lv_obj_t *ok_icon = lv_label_create(active_scr);
//             lv_label_set_text(ok_icon, LV_SYMBOL_OK);
//             lv_obj_align(ok_icon, LV_ALIGN_CENTER, 0, -50);

//             lv_obj_t *end_label = lv_label_create(active_scr);
//             lv_label_set_text(end_label, "Normal");
//             lv_obj_set_style_text_font(end_label, &lv_font_montserrat_30, 0);
//             lv_obj_align(end_label, LV_ALIGN_CENTER, 0, 0);

//             lv_obj_t *end_sub = lv_label_create(active_scr);
//             lv_label_set_text_fmt(end_sub, "BPM: %d", final_bpm);
//             lv_obj_align(end_sub, LV_ALIGN_CENTER, 0, 45);
//         }
//         _lock_release(&params->lcd_params.lvgl_api_lock);

//         xEventGroupSetBits(event_group_handle, ML_CLASSIFICATION_END_BIT);
//         xEventGroupClearBits(event_group_handle, ML_CLASSIFICATION_START_BIT);
//         vTaskDelay(pdMS_TO_TICKS(100));
//     }
// }
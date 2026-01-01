#include "mic_setup.h"

#include "cmn.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "lvgl.h"
#include <cstring>

/*
Raw ADC Packet Structure:
- Bits 0-11: ADC Value (12 bit resolution)
- Bits 13-16: ADC Channel (tells you which GPIO pin gave us the sample)
- Bits 17: Tells you if it came from ADC1 or ADC2
*/

static const char *AUDIO_TASK_TAG = "AUDIO_TASK";

void configure_mic_adc(adc_continuous_handle_t *handle)
{
    // Set up ADC DMA Buffer
    adc_continuous_handle_cfg_t adc_config = {
        .max_store_buf_size = 4096,
        .conv_frame_size = READ_LEN,
    };

    ESP_ERROR_CHECK(adc_continuous_new_handle(&adc_config, handle));

    // Configure ADC channel
    adc_continuous_config_t cont_config = {
        .pattern_num = 1,
        .adc_pattern = (adc_digi_pattern_config_t[]) {
            {
                .atten = ADC_ATTENUATION,
                .channel = ADC_CHANNEL,
                .unit = ADC_UNIT,
                .bit_width = ADC_BITWIDTH,
            }
        },
        .sample_freq_hz = SAMPLE_FREQ_HZ,
        .conv_mode = ADC_CONV_SINGLE_UNIT_1,
        .format = ADC_DIGI_OUTPUT_FORMAT_TYPE2,
    };

    ESP_ERROR_CHECK(adc_continuous_config(*handle, &cont_config));
}

void audio_sampling_task(void *audio_parameters)
{
    ESP_LOGI(AUDIO_TASK_TAG, "Starting audio sampling task");

    task_params* params = (task_params*)audio_parameters;
    adc_continuous_handle_t handle = params->mic_adc_handle;
    uint8_t * master_audio_buffer = params->master_audio_buffer;
    EventGroupHandle_t event_group_handle = params->event_group_handle;

    uint8_t read_buffer[READ_LEN];
    uint32_t bytes_read = 0;
        
    while (1)
    {
        // Block audio sampling task until button pressed via hardware interrupt
        ulTaskNotifyTake(pdTRUE, portMAX_DELAY);

        xEventGroupSetBits(event_group_handle, AUDIO_RECORDING_START_BIT);

        ESP_LOGI(AUDIO_TASK_TAG, "Record button pressed! Sampling now..");

        _lock_acquire(&params->lcd_params.lvgl_api_lock);

        lv_obj_clean(lv_screen_active());

        lv_obj_t * record_spinner = lv_spinner_create(lv_screen_active());
        lv_obj_set_size(record_spinner, 100, 100);
        lv_obj_center(record_spinner);
        lv_spinner_set_anim_params(record_spinner, 10000, 200);

        lv_obj_t *start_label = lv_label_create(lv_screen_active());
        lv_label_set_text(start_label, "Recording..");
        lv_obj_set_style_text_font(start_label, &lv_font_montserrat_18, 0);
        lv_obj_align(start_label, LV_ALIGN_CENTER, 0, 70);
        
        _lock_release(&params->lcd_params.lvgl_api_lock);

        int64_t sampling_start = esp_timer_get_time();

        // Start the DMA conversion
        ESP_ERROR_CHECK(adc_continuous_start(handle));
        
        uint32_t total_bytes_read = 0;
        uint32_t bytes_needed = MASTER_AUDIO_BUFFER_SIZE; // 10 seconds of 32-bit data

        while (total_bytes_read < bytes_needed) 
        {
            // Read from the DMA buffer
            esp_err_t err = adc_continuous_read(handle, read_buffer, READ_LEN, &bytes_read, 100);

            if (err == ESP_OK) 
            {
                uint32_t remaining_bytes = bytes_needed - total_bytes_read;

                if (bytes_read > remaining_bytes)
                {
                    bytes_read = remaining_bytes;
                }

                memcpy(master_audio_buffer + total_bytes_read, read_buffer, bytes_read);

                total_bytes_read += bytes_read;
            }
            // No data is available in the DMA buffer, so it times out
            else if (err == ESP_ERR_TIMEOUT) 
            {
                ESP_LOGW(AUDIO_TASK_TAG, "ADC read from DMA buffer timed out, retrying...");
            }
            // occurs when the hardware fills the DMA buffers faster than the software can consume them
            else if (err == ESP_ERR_INVALID_STATE) 
            {
                ESP_LOGE(AUDIO_TASK_TAG, "ADC Overrun! Data was lost.");
            }
        }

        // Stop conversion
        ESP_ERROR_CHECK(adc_continuous_stop(handle));

        int64_t sampling_end = esp_timer_get_time();

        _lock_acquire(&params->lcd_params.lvgl_api_lock);

        lv_obj_clean(lv_screen_active());

        lv_obj_t *end_label = lv_label_create(lv_screen_active());
        lv_label_set_text(end_label, "Done Recording.");
        lv_obj_center(end_label);
        
        lv_obj_t *end_sub_label = lv_label_create(lv_screen_active());
        lv_label_set_text(end_sub_label, "Press button to record again.");
        lv_obj_set_style_text_font(end_sub_label, &lv_font_montserrat_14, 0);
        lv_obj_align(end_sub_label, LV_ALIGN_CENTER, 0, 60);

        _lock_release(&params->lcd_params.lvgl_api_lock);

        ESP_LOGI(AUDIO_TASK_TAG, "Sampling Time (ms): %d", (uint32_t)((sampling_end-sampling_start)/1000.0f));
        ESP_LOGI(AUDIO_TASK_TAG, "Finished sampling.");

        xEventGroupClearBits(event_group_handle, AUDIO_RECORDING_START_BIT);
        xEventGroupSetBits(event_group_handle, AUDIO_RECORDING_DONE_BIT | 
                           BLE_STREAMING_START_BIT | ML_CLASSIFICATION_START_BIT);

        // Wait for BLE streaming and DSP+ML classification to be done before listening to user again
        xEventGroupWaitBits(event_group_handle, 
                            BLE_STREAMING_END_BIT | ML_CLASSIFICATION_END_BIT,  
                            pdTRUE, pdTRUE, portMAX_DELAY);

        xEventGroupClearBits(event_group_handle, AUDIO_RECORDING_DONE_BIT);
        vTaskDelay(pdMS_TO_TICKS(100));
    }
}
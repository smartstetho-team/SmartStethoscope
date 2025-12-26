#include "mic_setup.h"

#include "cmn.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "esp_timer.h"
#include <cstring>

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
                .atten = ADC_ATTEN,
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
    global_params* params = (global_params*)audio_parameters;
    adc_continuous_handle_t handle = params->mic_adc_handle;
    uint8_t * master_audio_buffer = params->master_audio_buffer;
    EventGroupHandle_t event_group_handle = params->event_group_handle;

    uint8_t read_buffer[READ_LEN];
    uint32_t bytes_read = 0;

    while (1)
    {
        ESP_LOGI(AUDIO_TASK_TAG, "Ready for audio sampling.");
        
        // Block audio sampling task until button pressed via hardware interrupt
        ulTaskNotifyTake(pdTRUE, portMAX_DELAY);

        xEventGroupSetBits(event_group_handle, AUDIO_RECORDING_START_BIT);

        ESP_LOGI(AUDIO_TASK_TAG, "Record button pressed! Sampling now..");

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
                memcpy(master_audio_buffer+total_bytes_read, read_buffer, bytes_read);
                total_bytes_read += bytes_read;
            }
            else if (err == ESP_ERR_TIMEOUT) 
            {
                ESP_LOGW(AUDIO_TASK_TAG, "ADC read from DMA buffer timed out, retrying...");
            }
        }

        // Stop conversion
        ESP_ERROR_CHECK(adc_continuous_stop(handle));

        int64_t sampling_end = esp_timer_get_time();

        ESP_LOGI(AUDIO_TASK_TAG, "Sampling Time (ms): %d", (sampling_end-sampling_start)/1000.0f);
        ESP_LOGI(AUDIO_TASK_TAG, "Finished sampling.");

        // Output ADC value every 100th sample (USE FOR DEBUGGING)
        for (int i = 0; i < total_bytes_read; i += 100*ADC_OUTPUT_LEN)
        {
            adc_digi_output_data_t *sample = (adc_digi_output_data_t*)&master_audio_buffer[i];
            ESP_LOGI(AUDIO_TASK_TAG, "%ld", sample->type2.data);
            vTaskDelay(pdMS_TO_TICKS(1)); 
        }

        xEventGroupClearBits(event_group_handle, AUDIO_RECORDING_START_BIT);
        xEventGroupSetBits(event_group_handle, AUDIO_RECORDING_DONE_BIT);

        // Wait for BLE streaming and DSP+ML classification to be done before listening to user again
        xEventGroupWaitBits(event_group_handle, 
                            BLE_STREAMING_END_BIT | ML_CLASSIFICATION_END_BIT,  
                            pdTRUE, pdTRUE, portMAX_DELAY);

        xEventGroupClearBits(event_group_handle, AUDIO_RECORDING_DONE_BIT);
        vTaskDelay(pdMS_TO_TICKS(100));
    }
}
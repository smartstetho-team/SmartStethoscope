#include "mic_setup.h"

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

void init_mic_adc(adc_continuous_handle_t *handle)
{
    // Set up ADC DMA Buffer
    adc_continuous_handle_cfg_t adc_config = {
        .max_store_buf_size = 4096,
        .conv_frame_size = READ_LEN,
    };

    adc_continuous_new_handle(&adc_config, handle);

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

    adc_continuous_config(*handle, &cont_config);
}

void audio_sampling_task(void *audio_parameters)
{
    printf("READY FOR AUDIO SAMPLING\n");

    adc_continuous_handle_t handle = (adc_continuous_handle_t)audio_parameters;

    uint8_t result[READ_LEN];
    uint32_t ret_num = 0;

    while (1)
    {
        ulTaskNotifyTake(pdTRUE, portMAX_DELAY);

        printf("Button pressed! Starting sampling now..\n");

        // Start the DMA conversion
        adc_continuous_start(handle);
        
        uint32_t total_bytes_collected = 0;
        uint32_t bytes_needed = 8000 * 10 * 4; // 10 seconds of 32-bit data

        while (total_bytes_collected < bytes_needed) 
        {
            // Read from the DMA buffer
            esp_err_t err = adc_continuous_read(handle, result, READ_LEN, &ret_num, 100);
        
            if (err == ESP_OK) 
            {
                // Continuous mode Type 2 format provides 4 bytes per sample
                for (int i = 0; i < ret_num; i += 4)
                {
                    // Unpack the 12-bit value from the Type 2 structure
                    adc_digi_output_data_t *p = (adc_digi_output_data_t*)&result[i];
                    uint32_t val = p->type2.data;

                    printf("%ld\n", val); 
                }

                total_bytes_collected += ret_num;
                vTaskDelay(pdMS_TO_TICKS(1));
            }
            else if (err == ESP_ERR_TIMEOUT) 
            {
                // No data ready yet, give other tasks time to run
                vTaskDelay(pdMS_TO_TICKS(1)); 
            }
        }

        // Stop conversion to save power
        adc_continuous_stop(handle);
        vTaskDelay(pdMS_TO_TICKS(100));
    }
}
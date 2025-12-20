#include <stdio.h>
#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_adc/adc_continuous.h"

// Configuration
#define ADC_UNIT            ADC_UNIT_1
#define ADC_CHANNEL         ADC_CHANNEL_0    // GPIO1 on S3
#define ADC_ATTEN           ADC_ATTEN_DB_12  // 0-3.3V range
#define ADC_BITWIDTH        ADC_BITWIDTH_12  
#define SAMPLE_FREQ_HZ      8000             // 8kHz sampling rate
#define READ_LEN            1024             // Bytes to read per DMA block

// Global handle for the ADC continuous driver
adc_continuous_handle_t handle = NULL;

void init_adc_continuous() {
    // 1. Initialize the ADC Continuous Driver
    adc_continuous_handle_cfg_t adc_config = {
        .max_store_buf_size = 4096,
        .conv_frame_size = READ_LEN,
    };
    adc_continuous_new_handle(&adc_config, &handle);

    // 2. Configure the ADC Channel
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
        .conv_mode = ADC_CONV_SINGLE_UNIT_1, // Use Unit 1 only
        .format = ADC_DIGI_OUTPUT_FORMAT_TYPE2, // Type 2 includes channel info
    };
    adc_continuous_config(handle, &cont_config);
}

extern "C" void app_main(void) {
    // Disable stdout buffering for immediate "READY" printing
    setvbuf(stdout, NULL, _IONBF, 0);
    
    init_adc_continuous();

    xTaskCreate([](void* p) {
        uint8_t result[READ_LEN];
        uint32_t ret_num = 0;

        while (1) {
            printf("READY\n");
            
            // Wait for 'r' from Python
            char c = 0;
            while (c != 'r') {
                if (scanf("%c", &c) != 1) vTaskDelay(pdMS_TO_TICKS(10));
            }

            // Start the DMA conversion
            adc_continuous_start(handle);
            
            int total_bytes_collected = 0;
            int bytes_needed = 8000 * 10 * 4; // 10 seconds of 16-bit data

            while (total_bytes_collected < bytes_needed) {
                // Read from the DMA buffer
                esp_err_t err = adc_continuous_read(handle, result, READ_LEN, &ret_num, 100);
                
                if (err == ESP_OK) {
                    // result contains 'ret_num' bytes of raw ADC data
                    fwrite(result, 1, ret_num, stdout);
                    fflush(stdout);
                    total_bytes_collected += ret_num;
                }

                // if (err == ESP_OK) 
                // {
                //     // Continuous mode Type 2 format provides 4 bytes per sample
                //     for (int i = 0; i < ret_num; i += 4) {
                //         // Unpack the 12-bit value from the Type 2 structure
                //         adc_digi_output_data_t *p = (adc_digi_output_data_t*)&result[i];
                //         uint32_t val = p->type2.data; 

                //         // Print as text so you can see it in the Serial Monitor
                //         printf("%ld\n", val); 
                //     }

                //     vTaskDelay(pdMS_TO_TICKS(1));
                // }
                // else if (err == ESP_ERR_TIMEOUT) 
                // {
                //     // No data ready yet, give the IDLE task time to run
                //     vTaskDelay(pdMS_TO_TICKS(1)); 
                // }
            }

            // Stop conversion to save power
            adc_continuous_stop(handle);
            vTaskDelay(pdMS_TO_TICKS(100));
        }
    }, "adc_task", 8192, NULL, 10, NULL);
}
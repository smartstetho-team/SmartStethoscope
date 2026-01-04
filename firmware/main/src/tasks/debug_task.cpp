#include "debug.h"

#include "cmn.h"
#include "mic_setup.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include <driver/usb_serial_jtag.h>

static const char *DEBUG_TASK_TAG = "DEBUG_TASK";

static void send_raw_audio(uint8_t* raw_audio_buffer)
{
    for (size_t i = 0; i < MASTER_AUDIO_BUFFER_SIZE; i += ADC_OUTPUT_LEN)
    {
        adc_digi_output_data_t sample;
        memcpy(&sample, &raw_audio_buffer[i], sizeof(adc_digi_output_data_t));
        uint16_t raw_value = (uint16_t)sample.type2.data;

        // Convert data to big endian
        uint8_t debug_audio_packet_raw[4] = {(uint8_t)(raw_value >> 8), 
                                             (uint8_t)(raw_value & 0xFF), 0, 0};
        usb_serial_jtag_write_bytes(debug_audio_packet_raw, 4, 10);
        
        // Feed the watchdog timer every 1024 bytes (every 256 samples)
        if ((i % 1024) == 0) 
        {
            vTaskDelay(pdMS_TO_TICKS(1));
        }
    }
}

static void send_filtered_audio(float* filtered_audio_buffer, float dc_offset)
{
    for (size_t i = 0; i < MASTER_AUDIO_BUFFER_SIZE; i += ADC_OUTPUT_LEN)
    {
        float adc_value = filtered_audio_buffer[i/ADC_OUTPUT_LEN];
        float denormalized_val = (adc_value * 2048.0f) + dc_offset;

        if (denormalized_val > 4095.0f) 
        {
            denormalized_val = 4095.0f;
        }
        else if (denormalized_val < 0.0f)
        {
            denormalized_val = 0.0f;
        } 
        
        uint16_t filtered_value = (uint16_t)denormalized_val;

        // Convert data to big endian
        uint8_t debug_audio_packet_filtered[4] = {(uint8_t)(filtered_value >> 8), 
                                                  (uint8_t)(filtered_value & 0xFF), 0, 0};
        usb_serial_jtag_write_bytes(debug_audio_packet_filtered, 4, 10);
        
        // Feed the watchdog timer every 1024 bytes (every 256 samples)
        if ((i % 1024) == 0) 
        {
            vTaskDelay(pdMS_TO_TICKS(1));
        }
    }
}

void debug_init()
{
    usb_serial_jtag_driver_config_t cfg = USB_SERIAL_JTAG_DRIVER_CONFIG_DEFAULT();
    usb_serial_jtag_driver_install(&cfg);
}

void debug_task(void *debug_parameters)
{
    ESP_LOGI(DEBUG_TASK_TAG, "Starting Debug task");
    
    task_params* params = (task_params*)debug_parameters;
    uint8_t * master_audio_buffer = params->master_audio_buffer;
    float* filtered_audio_buffer = params->filtered_audio_buffer;

    uint8_t serial_cmd = 0;
    while (1)
    {
        if (usb_serial_jtag_read_bytes(&serial_cmd, 1, 100) > 0)
        {
            // Send 'r' from python script to send raw audio bytes
            if (serial_cmd == 'r')
            {
                send_raw_audio(master_audio_buffer);
            }

            // Send 'f' from python script to send filtered audio bytes
            else if (serial_cmd == 'f')
            {
                send_filtered_audio(filtered_audio_buffer, 
                                    params->audio_dc_offset);
            }
            serial_cmd = 0;
        }
        vTaskDelay(pdMS_TO_TICKS(10));
    }
}
#include "debug.h"

#include "cmn.h"
#include "mic_setup.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include <driver/uart.h>
#include <driver/usb_serial_jtag.h>
#include <hal/uart_types.h>

static const char *DEBUG_TASK_TAG = "DEBUG_TASK";
static const uart_port_t uart_num = UART_NUM_0;

uart_config_t uart_config = {
    .baud_rate = 921600,
    .data_bits = UART_DATA_8_BITS,
    .parity = UART_PARITY_DISABLE,
    .stop_bits = UART_STOP_BITS_1,
    .flow_ctrl = UART_HW_FLOWCTRL_DISABLE
};

void debug_init()
{
    // ESP_ERROR_CHECK(uart_param_config(uart_num, &uart_config));
    // ESP_ERROR_CHECK(uart_driver_install(uart_num, 1024, 1024, 0, NULL, 0));
    usb_serial_jtag_driver_config_t cfg = USB_SERIAL_JTAG_DRIVER_CONFIG_DEFAULT();
    usb_serial_jtag_driver_install(&cfg);
}

void debug_task(void *debug_parameters)
{
    ESP_LOGI(DEBUG_TASK_TAG, "Starting Debug task");
    
    task_params* params = (task_params*)debug_parameters;
    uint8_t * master_audio_buffer = params->master_audio_buffer;

    uint8_t serial_cmd = 0;
    while (1)
    {
        if (usb_serial_jtag_read_bytes(&serial_cmd, 1, 100) > 0)
        {
            if (serial_cmd == 'r')
            {
                for (size_t i = 0; i < MASTER_AUDIO_BUFFER_SIZE; i += ADC_OUTPUT_LEN)
                {
                    adc_digi_output_data_t sample;
                    memcpy(&sample, &master_audio_buffer[i], sizeof(adc_digi_output_data_t));
                    uint16_t value = (uint16_t)sample.type2.data;

                    // Convert data to big endian
                    uint8_t debug_audio_packet[4] = {(uint8_t)(value >> 8), (uint8_t)(value & 0xFF), 0, 0};
                    usb_serial_jtag_write_bytes(debug_audio_packet, 4, 10);
                    
                    // Feed the watchdog timer every 1024 bytes (every 256 samples)
                    if ((i % 1024) == 0) 
                    {
                        vTaskDelay(pdMS_TO_TICKS(1));
                    }
                }
                serial_cmd = 0;
            }
        }
        vTaskDelay(pdMS_TO_TICKS(10));
    }
}
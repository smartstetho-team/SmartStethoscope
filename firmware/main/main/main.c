#include "setup.h"

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"
#include "driver/adc.h"
#include "esp_timer.h"

#include <stdio.h>

void app_main(void)
{
    adc1_config_width(MIC_ADC_WIDTH_BIT);
    adc1_config_channel_atten(MIC_ADC_CHANNEL, MIC_ADC_ATTEN);

    // raw_audio_buffer = xQueueCreate(4096, sizeof(uint8_t));
    // dsp_audio_buffer = xQueueCreate(4096, sizeof(uint8_t));

    /* Set up all relevant tasks */
    xTaskCreate(audio_sampling_task, "audio_sampling_task", 8192, NULL, 1, NULL);
    xTaskCreate(ble_streaming_task, "ble_streaming_task", 8192, NULL, 5, NULL);
    xTaskCreate(dsp_ml_processing_task, "dsp_ml_processing_task", 8192, NULL, 5, NULL);
    xTaskCreate(lcd_ui_task, "lcd_ui_task", 8192, NULL, 5, NULL);
}
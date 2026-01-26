#include "drivers/button.h"
#include "drivers/lcd_display.h"
#include "cmn.h"
#include "mic_setup.h"
#include "ble_streaming_task.h"
#include "dsp_ml_setup.h"
#include "lcd_ui_setup.h"
#include "debug.h"

#include <stdio.h>
#include <string.h>
#include <esp_heap_caps.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_adc/adc_continuous.h"
#include "esp_log.h"
#include "lvgl.h"

// Global variables
static const char *MAIN_TAG = "MAIN";

task_params task_parameters = 
{
    .master_audio_buffer = NULL,
    .filtered_audio_buffer = NULL,
    .audio_dc_offset = 0,
    .mic_adc_handle = NULL,
    .event_group_handle = NULL
};

TaskHandle_t audio_sampling_task_handle = NULL;
TaskHandle_t ble_streaming_task_handle = NULL;
TaskHandle_t ml_classification_task_handle = NULL;
TaskHandle_t lcd_ui_task_handle = NULL;
TaskHandle_t debug_task_handle = NULL;

extern "C" void app_main(void) 
{
    // Set up driver for serial debugging
    debug_init();

    // Set up mutex for LVGL resources
    _lock_init(&task_parameters.lcd_params.lvgl_api_lock);

    // Configure LCD display
    configure_lcd_display(&task_parameters.lcd_params);

    // Configure LVGL library and timer
    configure_lcd_lvgl(&task_parameters.lcd_params);

    // Initialize bootup screen
    bootup_screen_init((void*)&task_parameters.lcd_params);

    // Create event group so tasks can talk with each other
    task_parameters.event_group_handle = xEventGroupCreate();

    if (task_parameters.event_group_handle == NULL)
    {
        ESP_LOGE(MAIN_TAG, "Can't create group event handle!");
    }

    // Configure Mic ADC
    configure_mic_adc(&task_parameters.mic_adc_handle);

    // Allocate space for audio buffer in external RAM
    task_parameters.master_audio_buffer = (uint8_t*)heap_caps_malloc
                                          (MASTER_AUDIO_BUFFER_SIZE, MALLOC_CAP_SPIRAM);

    // Allocate space for filtered audio buffer in external RAM
    // Note: Size is halved since we only need ADC value instead of the whole packet
    task_parameters.filtered_audio_buffer = (float*)heap_caps_malloc
                                            (MASTER_AUDIO_BUFFER_SIZE/2 * sizeof(float), MALLOC_CAP_SPIRAM);

    if (task_parameters.master_audio_buffer == nullptr || 
        task_parameters.filtered_audio_buffer == nullptr)
    {
        ESP_LOGE(MAIN_TAG, "PSRAM Allocation Failed! Critical Error. \\
                 Current Free PSRAM: %d bytes", heap_caps_get_free_size(MALLOC_CAP_SPIRAM));

        while(1) 
        { 
            vTaskDelay(pdMS_TO_TICKS(1000)); 
        }
    }

    // Create all tasks
    xTaskCreate(audio_sampling_task, "audio_sampling_task", 8192, 
                (void*)&task_parameters, 7, &audio_sampling_task_handle);

    xTaskCreate(ble_streaming_task, "ble_streaming_task", 8192, 
                (void*)&task_parameters, 4, &ble_streaming_task_handle);

    xTaskCreate(ml_classification_task, "ml_classification_task", 8192, 
                (void*)&task_parameters, 4, &ml_classification_task_handle);

    xTaskCreate(lcd_ui_task, "lcd_ui_task", 10240, 
                (void*)&task_parameters, 2, &lcd_ui_task_handle);

    xTaskCreate(debug_task, "debug_task", 8192, 
                (void*)&task_parameters, 2, &debug_task_handle);

    // Set up button for the audio task
    configure_push_button(audio_sampling_task_handle, (void*)&task_parameters);
}
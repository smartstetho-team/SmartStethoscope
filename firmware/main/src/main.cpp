#include "drivers/button.h"
#include "drivers/lcd_display.h"
#include "cmn.h"
#include "mic_setup.h"
#include "ble_setup.h"
#include "dsp_ml_setup.h"
#include "lcd_ui_setup.h"

#include <stdio.h>
#include <string.h>
#include <esp_heap_caps.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_adc/adc_continuous.h"
#include "esp_log.h"
#include "lvgl.h"

// Global variables
global_params parameters; // contains the ptr to the master audio buffer

// Common parameters
adc_continuous_handle_t mic_adc_handle = NULL;
uint8_t* master_audio_buffer = NULL;
EventGroupHandle_t group_event_handle = NULL;
LCD_Display_Params lcd_params;

// Task related
TaskHandle_t audio_sampling_task_handle = NULL;
TaskHandle_t ble_streaming_task_handle = NULL;
TaskHandle_t ml_classification_task_handle = NULL;
TaskHandle_t lcd_ui_task_handle = NULL;

static const char *MAIN_TAG = "MAIN";

extern "C" void app_main(void) 
{
    // Set up mutex for lvgl resources
    _lock_init(&lcd_params.lvgl_api_lock);

    // Configure LCD display
    configure_lcd_display(&lcd_params);

    // Configure LVGL library and timer
    configure_lcd_lvgl(&lcd_params);

    // Initialize bootup screen
    bootup_screen_init((void*)&lcd_params);

    // Create event group so tasks can talk with each other
    group_event_handle = xEventGroupCreate();

    if (group_event_handle == NULL)
    {
        ESP_LOGE(MAIN_TAG, "Can't create group event handle!");
    }

    // Configure mic ADC
    configure_mic_adc(&mic_adc_handle);

    // Allocate space for audio buffer in external RAM
    master_audio_buffer = (uint8_t*)heap_caps_malloc(MASTER_AUDIO_BUFFER_SIZE, MALLOC_CAP_SPIRAM);

    if (master_audio_buffer == nullptr) 
    {
        ESP_LOGE(MAIN_TAG, "PSRAM Allocation Failed! Critical Error. \\
                 Current Free PSRAM: %d bytes", heap_caps_get_free_size(MALLOC_CAP_SPIRAM));

        while(1) 
        { 
            vTaskDelay(pdMS_TO_TICKS(1000)); 
        }
    }

    // Set up all important parameters that will be passed around
    parameters.master_audio_buffer = master_audio_buffer;
    parameters.mic_adc_handle = mic_adc_handle;
    parameters.event_group_handle = group_event_handle;

    // Create all tasks
    xTaskCreate(audio_sampling_task, "audio_sampling_task", 8192, 
                (void*)&parameters, 10, &audio_sampling_task_handle);

    xTaskCreate(ble_streaming_task, "ble_streaming_task", 8192, 
                (void*)&parameters, 5, &ble_streaming_task_handle);

    xTaskCreate(ml_classification_task, "ml_classification_task", 8192, 
                (void*)&parameters, 5, &ml_classification_task_handle);

    // xTaskCreate(lcd_ui_task, "lcd_ui_task", 10240, 
    //             (void*)&lcd_params, 3, &lcd_ui_task_handle);

    // Set up button for the audio task
    configure_push_button(audio_sampling_task_handle, (void*)&parameters);
}
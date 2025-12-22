#include "mic_setup.h"
#include "drivers/button.h"

#include <stdio.h>
#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_adc/adc_continuous.h"

// Global variables
adc_continuous_handle_t mic_adc_handle = NULL;
TaskHandle_t audio_sampling_task_handle = NULL;

extern "C" void app_main(void) 
{
    // Initialize components (ex. mic)
    init_mic_adc(&mic_adc_handle);

    // Create all tasks
    xTaskCreate(audio_sampling_task, "audio_sampling_task", 8192, 
                (void*)mic_adc_handle, 10, &audio_sampling_task_handle);

    // Set up button
    setup_push_button(audio_sampling_task_handle);
}
#include "setup.h"

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/adc.h"
#include "esp_timer.h"

void dsp_ml_processing_task(void *ble_parameters)
{
    while (1) {
        printf("Hello from dsp ml task\n");
        fflush(stdout);

        vTaskDelay(pdMS_TO_TICKS(1000));
    }
}
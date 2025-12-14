#include "setup.h"

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/adc.h"
#include "esp_timer.h"

void lcd_ui_task(void *ble_parameters)
{
    while (1) {
        printf("Hello from lcd ui task\n");
        fflush(stdout);

        vTaskDelay(pdMS_TO_TICKS(1000));
    }
}
#include "drivers/power_mgmt.h"

#include "cmn.h"
#include <driver/gpio.h>
#include "esp_timer.h"
#include "esp_log.h"

static TaskHandle_t target_task = NULL;
static gpio_config_t io_conf;

static const char *POWER_DRIVER_TAG = "POWER_DRIVER";

static void IRAM_ATTR lbo_isr_handler(void *args)
{
    BaseType_t xHigherPriorityTaskWoken = pdFALSE;

    vTaskNotifyGiveFromISR(target_task, &xHigherPriorityTaskWoken);

    if (xHigherPriorityTaskWoken) 
    {
        portYIELD_FROM_ISR();
    }
}

void configure_lbo_pin(TaskHandle_t task, void *args)
{
    target_task = task;

    io_conf = {
        .pin_bit_mask = (1 << LBO_GPIO_PIN),
        .mode = GPIO_MODE_INPUT,             
        .pull_up_en = GPIO_PULLUP_ENABLE,       // Enable internal pull-up
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .intr_type = GPIO_INTR_NEGEDGE,         // Interrupt called on falling edge due to pull-up config
    };

    ESP_ERROR_CHECK(gpio_config(&io_conf));
    ESP_ERROR_CHECK(gpio_isr_handler_add(LBO_GPIO_PIN, lbo_isr_handler, (void*)args));

    ESP_LOGI(POWER_DRIVER_TAG, "Low Battery Detection Setup Complete.");
}
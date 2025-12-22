#include "drivers/button.h"

#include <driver/gpio.h>

#define PUSH_BUTTON_PIN GPIO_NUM_4

static TaskHandle_t target_task = NULL;
static gpio_config_t io_conf;

static void IRAM_ATTR push_button_isr(void *arg)
{
    BaseType_t xHigherPriorityTaskWoken = pdFALSE;

    vTaskNotifyGiveFromISR(target_task, &xHigherPriorityTaskWoken);

    if (xHigherPriorityTaskWoken) {
        portYIELD_FROM_ISR();
    }
}

void setup_push_button(TaskHandle_t task)
{
    target_task = task;

    io_conf = {
        .pin_bit_mask = (1 << PUSH_BUTTON_PIN),
        .mode = GPIO_MODE_INPUT,             
        .pull_up_en = GPIO_PULLUP_ENABLE,     // Enable internal pull-up
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .intr_type = GPIO_INTR_NEGEDGE, // Interrupt called on falling edge due to pull-up config
    };

    gpio_config(&io_conf);
    gpio_install_isr_service(0);
    gpio_isr_handler_add(PUSH_BUTTON_PIN, push_button_isr, NULL);
}
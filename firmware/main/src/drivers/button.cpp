#include "cmn.h"
#include "drivers/button.h"
#include "esp_timer.h"

#include <driver/gpio.h>

#define PUSH_BUTTON_PIN GPIO_NUM_4
#define DEBOUNCE_TIME_MS 20

static TaskHandle_t target_task = NULL;
static gpio_config_t io_conf;

static void IRAM_ATTR push_button_isr(void *args)
{
    static int64_t previous_interrupt_time = 0;
    int64_t current_time = esp_timer_get_time();

    if ((current_time - previous_interrupt_time) > DEBOUNCE_TIME_MS)
    {
        BaseType_t xHigherPriorityTaskWoken = pdFALSE;

        global_params* params = (global_params*)args;
        EventGroupHandle_t event_group_handle = params->event_group_handle;
        EventBits_t uxBits = xEventGroupGetBitsFromISR(event_group_handle); // non-blocking call

        // Only invoke hardware interrupt if system is not recording and 
        // not doing dsp, ml processing or streaming BLE packets
        // This means the user must currently wait for a recording to be processed fully

        // TODO: We may want to add functionality to cancel the current audio recording process 
        // and start a new one in the future.
        if (!(uxBits & AUDIO_RECORDING_START_BIT) &&
            !(uxBits & BLE_STREAMING_START_BIT) &&
            !(uxBits & ML_CLASSIFICATION_START_BIT))
        {
            vTaskNotifyGiveFromISR(target_task, &xHigherPriorityTaskWoken);

            if (xHigherPriorityTaskWoken) 
            {
                portYIELD_FROM_ISR();
            }
        }
        previous_interrupt_time = current_time;
    }
}

void setup_push_button(TaskHandle_t task, void *args)
{
    target_task = task;

    io_conf = {
        .pin_bit_mask = (1 << PUSH_BUTTON_PIN),
        .mode = GPIO_MODE_INPUT,             
        .pull_up_en = GPIO_PULLUP_ENABLE,       // Enable internal pull-up
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .intr_type = GPIO_INTR_NEGEDGE,         // Interrupt called on falling edge due to pull-up config
    };

    gpio_config(&io_conf);
    gpio_install_isr_service(0);
    gpio_isr_handler_add(PUSH_BUTTON_PIN, push_button_isr, (void*)args);
}
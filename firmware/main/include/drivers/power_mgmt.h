#ifndef POWER_MGMT_H
#define POWER_MGMT_H

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

// GPIO 15 is RTC enabled so CPU can wake up after sleep
#define LBO_GPIO_PIN GPIO_NUM_15

void configure_lbo_pin(TaskHandle_t task, void *args);

#endif /* POWER_MGMT_H */
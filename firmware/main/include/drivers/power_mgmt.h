#ifndef POWER_MGMT_H
#define POWER_MGMT_H

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

// These pins are RTC enabled so CPU can wake up after sleep
#define LBO_GPIO_PIN GPIO_NUM_15
#define CHARGING_GPIO_PIN GPIO_NUM_16

#define BIT_LBO      (1 << 0)
#define BIT_CHARGING (1 << 1)

void configure_lbo_pin(TaskHandle_t task, void *args);
void configure_charging_pin(TaskHandle_t task, void *args);

#endif /* POWER_MGMT_H */
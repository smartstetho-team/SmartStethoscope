#ifndef BUTTON_H
#define BUTTON_H

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

void configure_push_button(TaskHandle_t task, void *args);

#endif /* BUTTON_H */
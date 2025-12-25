#include "cmn.h"
#include "dsp_ml_setup.h"

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include <cstring>

static const char *LCD_UI_TASK_TAG = "LCD_UI_TASK";

// void lcd_ui_task(void *arg)
// {
//     ESP_LOGI(LCD_UI_TASK_TAG, "Starting LVGL task");
//     uint32_t time_till_next_ms = 0;
//     while (1) 
//     {
//         _lock_acquire(&lvgl_api_lock);
//         // Note: This is a very important call. It figures out what components changed, updates 
//         // the DMA buffers with new pixels which then can be flushed out via the SPI driver
//         time_till_next_ms = lv_timer_handler();
//         _lock_release(&lvgl_api_lock);

//         vTaskDelay(pdMS_TO_TICKS(10));
//     }
// }
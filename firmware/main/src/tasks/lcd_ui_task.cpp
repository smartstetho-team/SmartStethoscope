#include "lcd_ui_setup.h"

#include "drivers/lcd_display.h"
#include "drivers/power_mgmt.h"
#include "cmn.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "esp_sleep.h"
#include "lvgl.h"

static const char *LCD_UI_TASK_TAG = "LCD_UI_TASK";

void bootup_screen_init(void * lcd_params)
{
    LCD_Display_Params* params = (LCD_Display_Params*)lcd_params;

    // Turn the LCD display ON
    ESP_ERROR_CHECK(esp_lcd_panel_disp_on_off(params->panel_handle, true));

    // Create components in the initial bootup screen
    _lock_acquire(&params->lvgl_api_lock);

    lv_obj_t *label = lv_label_create(lv_screen_active());
    lv_label_set_text(label, "Press button to start..");
    lv_obj_center(label);

    _lock_release(&params->lvgl_api_lock);
}

// This task keeps the LCD UI up to date
void lcd_ui_task(void *lcd_ui_parameters)
{
    ESP_LOGI(LCD_UI_TASK_TAG, "Starting LCD UI task");
    
    task_params* params = (task_params*)lcd_ui_parameters;
    LCD_Display_Params lcd_params = (LCD_Display_Params)params->lcd_params;
    uint32_t time_till_next_ms = 0;

    while (1) 
    {
        // If battery is low, put device into sleep mode
        // LBO pin fires an interrupt on a falling edge
        if (ulTaskNotifyTake(pdTRUE, 0)) 
        {
            _lock_acquire(&lcd_params.lvgl_api_lock);

            lv_obj_t * active_scr = lv_screen_active();
            lv_obj_clean(active_scr);

            lv_obj_t *err_icon = lv_label_create(active_scr);
            lv_label_set_text(err_icon, LV_SYMBOL_BATTERY_1);
            lv_obj_align(err_icon, LV_ALIGN_CENTER, 0, -50);

            lv_obj_t *end_label = lv_label_create(active_scr);
            lv_label_set_text(end_label, "Battery low");
            lv_obj_set_style_text_font(end_label, &lv_font_montserrat_30, 0);
            lv_obj_align(end_label, LV_ALIGN_CENTER, 0, 0);

            lv_obj_t *end_sub1 = lv_label_create(active_scr);
            lv_label_set_text(end_sub1, "Going to sleep..");
            lv_obj_align(end_sub1, LV_ALIGN_CENTER, 0, 45);

            _lock_release(&lcd_params.lvgl_api_lock);

            lv_timer_handler();

            vTaskDelay(pdMS_TO_TICKS(5000));
            
            // Go to sleep
            esp_sleep_enable_ext0_wakeup(LBO_GPIO_PIN, 1); 
            esp_deep_sleep_start();
        }

        _lock_acquire(&lcd_params.lvgl_api_lock);

        // Note: This is a very important call. It figures out what components changed, updates 
        // the DMA buffers with new pixels which then can be flushed out via the SPI driver.
        time_till_next_ms = lv_timer_handler();
        
        _lock_release(&lcd_params.lvgl_api_lock);
        
        if (time_till_next_ms < LVGL_TASK_MIN_DELAY_MS)
        {
            time_till_next_ms = LVGL_TASK_MIN_DELAY_MS;
        }
        
        if (LVGL_TASK_MAX_DELAY_MS < time_till_next_ms)
        {
            time_till_next_ms = LVGL_TASK_MAX_DELAY_MS;
        }
        vTaskDelay(pdMS_TO_TICKS(time_till_next_ms));
    }
}
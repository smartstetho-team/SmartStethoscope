#include "lcd_ui_setup.h"

#include "drivers/lcd_display.h"
#include "drivers/power_mgmt.h"
#include <driver/gpio.h>
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
    
    _lock_acquire(&params->lvgl_api_lock);
    lv_obj_t * active_scr = lv_screen_active();
    lv_obj_clean(active_scr);

    lv_obj_set_style_bg_color(active_scr, lv_color_hex(0x121212), 0);
    lv_obj_set_scrollbar_mode(active_scr, LV_SCROLLBAR_MODE_OFF);
    lv_obj_clear_flag(active_scr, LV_OBJ_FLAG_SCROLLABLE);

    // CardioScope Brand
    lv_obj_t *welcome = lv_label_create(active_scr);
    lv_label_set_text(welcome, "CardioScope"); 
    lv_obj_set_style_text_color(welcome, lv_color_white(), 0);
    lv_label_set_recolor(welcome, true);
    lv_obj_set_style_text_font(welcome, &lv_font_montserrat_30, 0); 
    lv_obj_set_style_text_align(welcome, LV_TEXT_ALIGN_CENTER, 0);
    lv_obj_align(welcome, LV_ALIGN_CENTER, 0, -85);

    // Logo
    lv_obj_t * logo_img = lv_img_create(active_scr);
    lv_img_set_src(logo_img, &logo);
    lv_img_set_zoom(logo_img, 64);
    lv_obj_align(logo_img, LV_ALIGN_CENTER, 0, 10);

    lv_obj_t *ready = lv_label_create(active_scr);
    lv_label_set_text(ready, "Press button to start..");
    lv_obj_set_style_text_font(ready, &lv_font_montserrat_20, 0);
    lv_obj_set_style_text_color(ready, lv_color_white(), 0);
    lv_obj_align(ready, LV_ALIGN_BOTTOM_MID, 0, -25);

    lv_anim_t a;
    lv_anim_init(&a);
    lv_anim_set_var(&a, ready);
    lv_anim_set_values(&a, 255, 80);
    lv_anim_set_time(&a, 1200);
    lv_anim_set_playback_time(&a, 800);
    lv_anim_set_repeat_count(&a, LV_ANIM_REPEAT_INFINITE);
    lv_anim_set_exec_cb(&a, [](void * var, int32_t v) {
        lv_obj_set_style_text_opa((lv_obj_t *)var, v, 0);
    });
    lv_anim_start(&a);

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
        uint32_t notified_value;

        // LBO pin fires an interrupt on a falling edge
        // Charging pin fires an interrupt on a rising edge
        if (xTaskNotifyWait(0, 0xFFFFFFFF, &notified_value, 0) == pdTRUE) 
        {
            ESP_LOGI(LCD_UI_TASK_TAG, "Notification received! Bits: 0x%08X", notified_value);

            // CASE 1: Low Battery (Priority 1)
            if (notified_value & BIT_LBO) 
            {
                ESP_LOGI(LCD_UI_TASK_TAG, "Low power detection interrupt!");
                _lock_acquire(&lcd_params.lvgl_api_lock);

                lv_obj_t * active_scr = lv_screen_active();
                lv_obj_clean(active_scr);

                lv_obj_t *err_icon = lv_label_create(active_scr);
                lv_label_set_text(err_icon, LV_SYMBOL_BATTERY_1);
                lv_obj_align(err_icon, LV_ALIGN_CENTER, 0, -50);
                lv_obj_set_style_text_color(err_icon, lv_palette_main(LV_PALETTE_AMBER), 0);

                lv_obj_t *end_label = lv_label_create(active_scr);
                lv_label_set_text(end_label, "Battery low");
                lv_obj_set_style_text_color(end_label, lv_color_white(), 0);
                
                lv_obj_set_style_text_font(end_label, &lv_font_montserrat_30, 0);
                lv_obj_align(end_label, LV_ALIGN_CENTER, 0, 0);

                lv_obj_t *end_sub1 = lv_label_create(active_scr);
                lv_label_set_text(end_sub1, "Going to sleep..");
                lv_obj_set_style_text_color(end_sub1, lv_color_white(), 0);
                lv_obj_align(end_sub1, LV_ALIGN_CENTER, 0, 45);

                _lock_release(&lcd_params.lvgl_api_lock);

                lv_timer_handler();

                vTaskDelay(pdMS_TO_TICKS(5000));
                
                // Go to sleep
                esp_sleep_enable_ext0_wakeup(LBO_GPIO_PIN, 1); 
                esp_deep_sleep_start();
            }

            // CASE 2: Charging (Priority 2)
            static lv_obj_t *chg_label = NULL;
            if (notified_value & BIT_CHARGING) 
            {
                ESP_LOGI(LCD_UI_TASK_TAG, "Charging dock interrupt!");
                bool is_docked = (gpio_get_level(CHARGING_GPIO_PIN) == 1);
                _lock_acquire(&lcd_params.lvgl_api_lock);
                
                if (is_docked) {
                    if (chg_label == NULL) {
                        chg_label = lv_label_create(lv_screen_active());
                        lv_label_set_text(chg_label, LV_SYMBOL_CHARGE " Docked");
                        lv_obj_align(chg_label, LV_ALIGN_TOP_RIGHT, -10, 10);
                        lv_obj_set_style_text_color(chg_label, lv_palette_main(LV_PALETTE_LIGHT_BLUE), 0);
                        ESP_LOGI(LCD_UI_TASK_TAG, "UI: Charging Icon Added");
                    }
                }
                else {
                    // If we just unplugged and the label exists, kill it
                    if (chg_label != NULL) {
                        lv_obj_del(chg_label); 
                        chg_label = NULL; // Reset to NULL so we can create it again later
                        ESP_LOGI(LCD_UI_TASK_TAG, "UI: Charging Icon Removed");
                    }
                }   
                
                _lock_release(&lcd_params.lvgl_api_lock);
            }
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
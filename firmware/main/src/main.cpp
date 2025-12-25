#include "cmn.h"
#include "mic_setup.h"
#include "ble_setup.h"
#include "dsp_ml_setup.h"
#include "drivers/button.h"
#include "drivers/lcd_display.h"

#include <stdio.h>
#include <string.h>
#include <esp_heap_caps.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_adc/adc_continuous.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "lvgl.h"

// Global variables
global_params parameters;

// Common parameters
adc_continuous_handle_t mic_adc_handle = NULL;
uint8_t* master_audio_buffer = NULL;
EventGroupHandle_t group_event_handle = NULL;
LCD_Display_Params lcd_params;

// Task related
TaskHandle_t audio_sampling_task_handle = NULL;
TaskHandle_t ble_streaming_task_handle = NULL;
TaskHandle_t ml_classification_task_handle = NULL;

static const char *MAIN_TAG = "MAIN";

#define LVGL_DRAW_BUF_LINES  LCD_DISPLAY_HEIGHT_PXLS/4
#define LVGL_TICK_PERIOD_MS     2
#define LVGL_TASK_STACK_SIZE 10240
#define LVGL_TASK_PRIORITY 3


static void lvgl_flush_cb(lv_display_t *disp, const lv_area_t *area, uint8_t *px_map)
{
    // Get the LCD handle and pixel offset
    esp_lcd_panel_handle_t panel_handle = (esp_lcd_panel_handle_t)lv_display_get_user_data(disp);

    int offsetx1 = area->x1;
    int offsetx2 = area->x2; 
    int offsety1 = area->y1;
    int offsety2 = area->y2;

    // because SPI LCD is big-endian, we need to swap the RGB bytes order
    lv_draw_sw_rgb565_swap(px_map, (offsetx2 + 1 - offsetx1) * (offsety2 + 1 - offsety1));

    printf("TRYING TO FLUSH DATA!\n");

    // copy a buffer's content to a specific area of the display
    esp_lcd_panel_draw_bitmap(panel_handle, offsetx1, offsety1, offsetx2 + 1, offsety2 + 1, px_map); // is what draws everything
}

static void increase_lvgl_tick(void *arg)
{
    /* Tell LVGL how many milliseconds has elapsed */
    lv_tick_inc(LVGL_TICK_PERIOD_MS);
}

static bool notify_lvgl_flush_ready(
    esp_lcd_panel_io_handle_t panel_io,
    esp_lcd_panel_io_event_data_t *edata,
    void *user_ctx)
{
    lv_display_t *disp = (lv_display_t *)user_ctx;
    lv_display_flush_ready(disp);
    return false;
}

static _lock_t lvgl_api_lock; // mutex for when a thread accesses a LVGL related resource

// Task that basically lets us render new lvgl components and tells the UI to constantly refresh
// Note: Use the mutex when you use lvgl related resources
static void lvgl_port_task(void *arg)
{
    ESP_LOGI(MAIN_TAG, "Starting LVGL task");
    uint32_t time_till_next_ms = 0;
    while (1) 
    {
        _lock_acquire(&lvgl_api_lock);
        // Note: This is a very important call. It figures out what components changed, updates 
        // the DMA buffers with new pixels which then can be flushed out via the SPI driver
        time_till_next_ms = lv_timer_handler();
        _lock_release(&lvgl_api_lock);

        vTaskDelay(pdMS_TO_TICKS(10));
    }
}

extern "C" void app_main(void) 
{
    setup_lcd_display(&lcd_params);

    lv_init();
    lv_display_t *display = lv_display_create(LCD_DISPLAY_HEIGHT_PXLS, LCD_DISPLAY_WIDTH_PXLS);

    size_t draw_buffer_sz = LCD_DISPLAY_HEIGHT_PXLS * LVGL_DRAW_BUF_LINES * sizeof(lv_color16_t);

    void *buf1 = spi_bus_dma_memory_alloc(LCD_HOST, draw_buffer_sz, 0);
    void *buf2 = spi_bus_dma_memory_alloc(LCD_HOST, draw_buffer_sz, 0);

    if ((buf1 == nullptr) || (buf2 == nullptr))
    {
        ESP_LOGE(MAIN_TAG, "SPI DMA Buffers Allocation Failed!");
    }

    lv_display_set_buffers(display, buf1, buf2, draw_buffer_sz, LV_DISPLAY_RENDER_MODE_PARTIAL);

    lv_display_set_user_data(display, lcd_params.panel_handle);

    lv_display_set_color_format(display, LV_COLOR_FORMAT_RGB565); // R - 5 bits, G - 6 bits, B - 5 bits

    lv_display_set_flush_cb(display, lvgl_flush_cb);

    const esp_timer_create_args_t lvgl_tick_timer_args = 
    {
        .callback = &increase_lvgl_tick,
        .name = "lvgl_tick"
    };
    esp_timer_handle_t lvgl_tick_timer = NULL;

    ESP_ERROR_CHECK(esp_timer_create(&lvgl_tick_timer_args, &lvgl_tick_timer));
    ESP_ERROR_CHECK(esp_timer_start_periodic(lvgl_tick_timer, LVGL_TICK_PERIOD_MS * 1000));
    const esp_lcd_panel_io_callbacks_t cbs = 
    {
        .on_color_trans_done = notify_lvgl_flush_ready,
    };
    ESP_ERROR_CHECK(esp_lcd_panel_io_register_event_callbacks(lcd_params.io_handle, &cbs, display));

    // Turn the panel ON here
    ESP_ERROR_CHECK(esp_lcd_panel_disp_on_off(lcd_params.panel_handle, true));

    _lock_init(&lvgl_api_lock);

    _lock_acquire(&lvgl_api_lock);

    // TEST PRINT - LOOK AT https://docs.lvgl.io/master/examples.html
    lv_obj_t *label = lv_label_create(lv_screen_active());
    lv_label_set_text(label, "READY TO LISTEN");
    lv_obj_set_style_text_color(label, lv_palette_main(LV_PALETTE_GREEN), 0);
    lv_obj_center(label);

    _lock_release(&lvgl_api_lock);

    xTaskCreate(lvgl_port_task, "LVGL", LVGL_TASK_STACK_SIZE, NULL, LVGL_TASK_PRIORITY, NULL);

    // group_event_handle = xEventGroupCreate();

    // if (group_event_handle == NULL)
    // {
    //     ESP_LOGE(MAIN_TAG, "Can't create group event handle\n");
    // }

    // // Initialize components (ex. mic)
    // init_mic_adc(&mic_adc_handle);

    // // Allocate space for audio buffer in external RAM
    // master_audio_buffer = (uint8_t*)heap_caps_malloc(MASTER_AUDIO_BUFFER_SIZE, MALLOC_CAP_SPIRAM);

    // if (master_audio_buffer == nullptr) {
    //     ESP_LOGE(MAIN_TAG, "PSRAM Allocation Failed! Critical Error. 
    //              Current Free PSRAM: %d bytes", heap_caps_get_free_size(MALLOC_CAP_SPIRAM));

    //     while(1) 
    //     { 
    //         vTaskDelay(pdMS_TO_TICKS(1000)); 
    //     }
    // }

    // // Set up all parameters
    // parameters.master_audio_buffer = master_audio_buffer;
    // parameters.mic_adc_handle = mic_adc_handle;
    // parameters.event_group_handle = group_event_handle;

    // // Create all tasks
    // xTaskCreate(audio_sampling_task, "audio_sampling_task", 8192, 
    //             (void*)&parameters, 10, &audio_sampling_task_handle);

    // xTaskCreate(ble_streaming_task, "ble_streaming_task", 8192, 
    //             (void*)&parameters, 5, &ble_streaming_task_handle);

    // xTaskCreate(ml_classification_task, "ml_classification_task", 8192, 
    //             (void*)&parameters, 5, &ml_classification_task_handle);

    // // Set up button for the audio task
    // setup_push_button(audio_sampling_task_handle, (void*)&parameters);
}
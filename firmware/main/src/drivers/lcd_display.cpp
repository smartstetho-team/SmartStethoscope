/*
 Reference Documentation:
- https://docs.espressif.com/projects/esp-idf/en/stable/esp32/api-reference/peripherals/lcd/index.html#application-example
- https://docs.lvgl.io/master/examples.html
- https://github.com/ryanfkeller/hello-lvgl-esp-idf
*/

#include "drivers/lcd_display.h"

#include "driver/gpio.h"
#include "driver/spi_master.h"
#include "esp_log.h"

static const char *LCD_DISPLAY_DRIVER_TAG = "LCD_DISPLAY_DRIVER";

void setup_lcd_display(LCD_Display_Params * lcd_params)
{
    // Turn on LCD backlight
    gpio_config_t bk_gpio_config = 
    {
        .pin_bit_mask = 1 << BK_LIGHT_GPIO_PIN,
        .mode = GPIO_MODE_OUTPUT,
    };
    ESP_ERROR_CHECK(gpio_config(&bk_gpio_config));
    ESP_ERROR_CHECK(gpio_set_level(BK_LIGHT_GPIO_PIN, 1));

    // Set up SPI Bus
    spi_bus_config_t bus = {0};
    bus.mosi_io_num = MOSI_GPIO_PIN;
    bus.miso_io_num = MISO_GPIO_PIN;
    bus.sclk_io_num = SCLK_GPIO_PIN;
    bus.quadwp_io_num = -1;
    bus.quadhd_io_num = -1;
    bus.max_transfer_sz = LCD_DISPLAY_HEIGHT_PXLS * LCD_DISPLAY_WIDTH_PXLS * sizeof(uint16_t);
    ESP_ERROR_CHECK(spi_bus_initialize(LCD_HOST, &bus, SPI_DMA_CH_AUTO));

    // Connect LCD panel to the SPI bus
    lcd_params->io_handle = NULL;
    lcd_params->io_config = 
    {
        .cs_gpio_num = LCD_CS_GPIO_PIN,
        .dc_gpio_num = LCD_DC_GPIO_PIN,
        .spi_mode = 0,
        .pclk_hz = 10000000, // 10 MHz SPI Clock Frequency 
        .trans_queue_depth = 10,
        .on_color_trans_done = NULL,
        .user_ctx = NULL,
        .lcd_cmd_bits = 8,
        .lcd_param_bits = 8,
        .flags = {0} 
    };
    ESP_ERROR_CHECK(esp_lcd_new_panel_io_spi((esp_lcd_spi_bus_handle_t)LCD_HOST, &lcd_params->io_config, &lcd_params->io_handle));

    // Install ST7789 Driver
    lcd_params->panel_handle = NULL; 
    lcd_params->panel_config = 
    {
        .reset_gpio_num = LCD_RESET_GPIO_PIN,
        .rgb_ele_order = LCD_RGB_ELEMENT_ORDER_RGB,
        .data_endian = LCD_RGB_DATA_ENDIAN_BIG,
        .bits_per_pixel = 16,
        .flags = { .reset_active_high = 0 },
        .vendor_config = NULL
    };
    ESP_ERROR_CHECK(esp_lcd_new_panel_st7789(lcd_params->io_handle, &lcd_params->panel_config, &lcd_params->panel_handle));

    // Initialize and Configure LCD Display
    ESP_ERROR_CHECK(esp_lcd_panel_reset(lcd_params->panel_handle));
    vTaskDelay(pdMS_TO_TICKS(150));
    ESP_ERROR_CHECK(esp_lcd_panel_init(lcd_params->panel_handle));
    vTaskDelay(pdMS_TO_TICKS(150));
    ESP_ERROR_CHECK(esp_lcd_panel_set_gap(lcd_params->panel_handle, 0, 0));
    ESP_ERROR_CHECK(esp_lcd_panel_swap_xy(lcd_params->panel_handle, true));
    ESP_ERROR_CHECK(esp_lcd_panel_mirror(lcd_params->panel_handle, false, true));
    ESP_ERROR_CHECK(esp_lcd_panel_invert_color(lcd_params->panel_handle, false));
    ESP_ERROR_CHECK(esp_lcd_panel_disp_on_off(lcd_params->panel_handle, true)); // Display turned off by default

    ESP_LOGI(LCD_DISPLAY_DRIVER_TAG, "LCD Display Setup Complete.\n");
}

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_lcd_panel_io.h"
#include "esp_lcd_panel_vendor.h"
#include "esp_lcd_panel_ops.h"

#define LCD_HOST           SPI2_HOST
#define SCLK_GPIO_PIN      GPIO_NUM_12
#define MOSI_GPIO_PIN      GPIO_NUM_11
#define MISO_GPIO_PIN      GPIO_NUM_NC  
#define LCD_DC_GPIO_PIN    GPIO_NUM_13
#define LCD_CS_GPIO_PIN    GPIO_NUM_10
#define LCD_RESET_GPIO_PIN GPIO_NUM_9
#define BK_LIGHT_GPIO_PIN  GPIO_NUM_5

#define LCD_DISPLAY_WIDTH_PXLS 240
#define LCD_DISPLAY_HEIGHT_PXLS 320

typedef struct
{
    esp_lcd_panel_io_handle_t io_handle;
    esp_lcd_panel_io_spi_config_t io_config;
    esp_lcd_panel_handle_t panel_handle;
    esp_lcd_panel_dev_config_t panel_config;
} LCD_Display_Params;

void setup_lcd_display(LCD_Display_Params * lcd_params);

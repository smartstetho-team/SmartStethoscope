#ifndef LCD_UI_SETUP_H
#define LCD_UI_SETUP_H

#include "lvgl.h"

#ifdef __cplusplus
extern "C" {
#endif
    LV_IMG_DECLARE(logo);
    LV_IMG_DECLARE(heart);
#ifdef __cplusplus
}
#endif

/* LCD UI Configuration */

void bootup_screen_init(void * lcd_params);
void lcd_ui_task(void *lcd_ui_parameters);

#endif /* LCD_UI_SETUP_H */

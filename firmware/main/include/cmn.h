#ifndef CMN_H
#define CMN_H

#include "esp_adc/adc_continuous.h"
#include "freertos/FreeRTOS.h"
#include "drivers/lcd_display.h"

// Event group bits
#define AUDIO_RECORDING_START_BIT      (1 << 0)
#define AUDIO_RECORDING_DONE_BIT       (1 << 1)

#define BLE_STREAMING_START_BIT        (1 << 2)
#define BLE_STREAMING_END_BIT          (1 << 3)

#define ML_CLASSIFICATION_START_BIT    (1 << 4)
#define ML_CLASSIFICATION_END_BIT      (1 << 5)

// Parameters to be used between tasks
typedef struct
{
    uint8_t* master_audio_buffer;           // Buffer containing raw audio data
    float* filtered_audio_buffer;           // Buffer containing filtered audio data
    float audio_dc_offset;                  // Calculated DC offset from raw audio data
    adc_continuous_handle_t mic_adc_handle; // Reference Handle to our ADC Mic
    EventGroupHandle_t event_group_handle;  // Reference Handle to our event groups
    LCD_Display_Params lcd_params;          // Parameters for the LCD Dispkay

    float* inference_buffer_a;              // Buffer 1 for heart inference
    float* inference_buffer_b;              // Buffer 2 for heart inference
} task_params;

#endif /* CMN_H */
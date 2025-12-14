#ifndef SETUP_H_
#define SETUP_H_

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"

// Mic configuration
#define MIC_ADC_CHANNEL         ADC1_CHANNEL_0
#define MIC_ADC_WIDTH_BIT       ADC_WIDTH_BIT_12
#define MIC_ADC_ATTEN           ADC_ATTEN_DB_11
#define SAMPLE_RATE         8000
#define RECORD_SECONDS      10
#define NUM_SAMPLES         (SAMPLE_RATE * RECORD_SECONDS)

// QueueHandle_t raw_audio_buffer;
// QueueHandle_t dsp_audio_buffer;

void ble_streaming_task(void *ble_parameters);
void lcd_ui_task(void *lcd_display_parameters);
void audio_sampling_task(void *audio_parameters);
void dsp_ml_processing_task(void *dsp_parameters);

#endif /* SETUP_H_ */
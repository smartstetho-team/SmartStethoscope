#include <stdio.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/adc.h"
#include "esp_timer.h"

#define SAMPLE_RATE         8000
#define RECORD_SECONDS      10
#define NUM_SAMPLES         (SAMPLE_RATE * RECORD_SECONDS)

#define ADC_CHANNEL         ADC1_CHANNEL_0
#define ADC_WIDTH_BIT       ADC_WIDTH_BIT_12
#define ADC_ATTEN           ADC_ATTEN_DB_11

void recorder_task(void *arg)
{
    int interval_us = 1000000 / SAMPLE_RATE;
    uint8_t buf[512];
    int idx = 0;

    while (1) {
        int c = getchar();
        if (c == 'r') {

            printf("Recording...\n");

            int64_t next = esp_timer_get_time();

            for (int i = 0; i < NUM_SAMPLES; i++) {

                next += interval_us;
                while (esp_timer_get_time() < next) {}

                int raw = adc1_get_raw(ADC_CHANNEL);
                uint16_t s = (uint16_t)raw;

                buf[idx++] = (s >> 8) & 0xFF;
                buf[idx++] = s & 0xFF;

                if (idx >= sizeof(buf)) {
                    fwrite(buf, 1, idx, stdout);
                    fflush(stdout);
                    idx = 0;
                }
            }
            if (idx > 0) {
                fwrite(buf, 1, idx, stdout);
                fflush(stdout);
                idx = 0;
            }

            printf("\nDONE\n");
            printf("READY\n");
            fflush(stdout);
        }
    }
}

void app_main(void)
{
    adc1_config_width(ADC_WIDTH_BIT);
    adc1_config_channel_atten(ADC_CHANNEL, ADC_ATTEN);

    printf("READY\n");
    fflush(stdout);

    xTaskCreate(recorder_task, "recorder_task", 4096, NULL, 5, NULL);
}

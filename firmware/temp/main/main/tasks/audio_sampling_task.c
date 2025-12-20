#include "setup.h"

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/adc.h"
#include "esp_timer.h"

void audio_sampling_task(void *audio_parameters)
{
    int interval_us = 1000000 / SAMPLE_RATE;
    uint8_t buf[512];
    int idx = 0;

    while (1) 
    {
        printf("Start of recording...\n");

        int64_t next = esp_timer_get_time();

        for (int i = 0; i < NUM_SAMPLES; i++) {

            next += interval_us;
            while (esp_timer_get_time() < next) {}

            int raw_mic_voltage = adc1_get_raw(MIC_ADC_CHANNEL);
            uint16_t s = (uint16_t)raw_mic_voltage;

            buf[idx++] = (s >> 8) & 0xFF;
            buf[idx++] = s & 0xFF;

            if (idx >= sizeof(buf)) {
                fwrite(buf, 1, idx, stdout);
                fflush(stdout);
                idx = 0;
            }
        }

        // Write remaining data in the buffer
        if (idx > 0) {
            fwrite(buf, 1, idx, stdout);
            fflush(stdout);
            idx = 0;
        }

        printf("\nDONE\n");
        printf("READY\n");
        fflush(stdout);
        vTaskDelay(pdMS_TO_TICKS(100));
    }
}
#ifndef MIC_SETUP_H
#define MIC_SETUP_H

#include "esp_adc/adc_continuous.h"

/* MIC ADC Channel Configuration */
#define ADC_UNIT            ADC_UNIT_1
#define ADC_CHANNEL         ADC_CHANNEL_0    // GPIO1 on S3
#define ADC_ATTENUATION     ADC_ATTEN_DB_12   // 0-3.3V range
#define ADC_BITWIDTH        ADC_BITWIDTH_12
#define ADC_OUTPUT_LEN      4                // Continuous mode Type 2 format provides 4 bytes per sample
#define ADC_READ_TIMEOUT_MS    200           

/* VALUES SUBJECT TO CHANGE */
// BEST RIGHT NOW. 8000 Hz and 3 secs.
// 5000 hz and 5 secs
// 3000 hz and 8 secs -> good for bpm and somewhat murmur accuracy if there is noise
// 4000 hz and 6 secs -> ok for bpm, but good for murmur accuracy
#define SAMPLE_FREQ_HZ      4000             // 4kHz sampling rate (250us between each sample)
#define AUDIO_LENGTH        6                // Recorded audio length in seconds
#define READ_LEN            1024             // Bytes to read per DMA block
#define NUM_OF_SAMPLES      (SAMPLE_FREQ_HZ * AUDIO_LENGTH) // Number of samples
#define MASTER_AUDIO_BUFFER_SIZE (SAMPLE_FREQ_HZ * AUDIO_LENGTH * ADC_OUTPUT_LEN) // Size of master audio buffer (may need offset to prevent overflow)

void configure_mic_adc(adc_continuous_handle_t *handle);
void audio_sampling_task(void *audio_parameters);


/*
Notes: If audio length is 3 seconds, it works well. But the moment
it is increased, it stops working. Size of buffer a and b are 1 700 000.
I think its too much memory to copy at once.

*/

#endif /* MIC_SETUP_H */

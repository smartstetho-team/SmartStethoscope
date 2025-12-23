
#include "esp_adc/adc_continuous.h"

/* MIC ADC Channel Configuration */
#define ADC_UNIT            ADC_UNIT_1
#define ADC_CHANNEL         ADC_CHANNEL_0    // GPIO1 on S3
#define ADC_ATTEN           ADC_ATTEN_DB_12  // 0-3.3V range
#define ADC_BITWIDTH        ADC_BITWIDTH_12
#define ADC_OUTPUT_LEN      4                // Continuous mode Type 2 format provides 4 bytes per sample

/* VALUES SUBJECT TO CHANGE */
#define SAMPLE_FREQ_HZ      8000             // 8kHz sampling rate (125us between each sample)
#define AUDIO_LENGTH        10               // Recorded audio length in seconds
#define READ_LEN            1024             // Bytes to read per DMA block
#define MASTER_AUDIO_BUFFER_SIZE (SAMPLE_FREQ_HZ*AUDIO_LENGTH*ADC_OUTPUT_LEN) // Size of master audio buffer (may need offset to prevent overflow)

void init_mic_adc(adc_continuous_handle_t *handle);
void audio_sampling_task(void *audio_parameters);

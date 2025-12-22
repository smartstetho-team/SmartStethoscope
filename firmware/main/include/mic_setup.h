
#include "esp_adc/adc_continuous.h"

/* MIC ADC Channel Configuration */
#define ADC_UNIT            ADC_UNIT_1
#define ADC_CHANNEL         ADC_CHANNEL_0    // GPIO1 on S3
#define ADC_ATTEN           ADC_ATTEN_DB_12  // 0-3.3V range
#define ADC_BITWIDTH        ADC_BITWIDTH_12  

/* VALUES SUBJECT TO CHANGE */
#define SAMPLE_FREQ_HZ      8000             // 8kHz sampling rate (125us between each sample)
#define READ_LEN            1024             // Bytes to read per DMA block

// typedef struct
// {
    
// } audio_packet;

void init_mic_adc(adc_continuous_handle_t *handle);
void audio_sampling_task(void *audio_parameters);

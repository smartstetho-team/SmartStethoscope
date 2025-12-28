#include "dsp_ml_setup.h"
#include "mic_setup.h"

#include "cmn.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"

static const char *ML_CLASSIFICATION_TASK_TAG = "ML_CLASSIFICATION_TASK";

void ml_classification_task(void *dsp_ml_parameters)
{
    ESP_LOGI(ML_CLASSIFICATION_TASK_TAG, "Starting ML classification task");
    
    task_params* params = (task_params*)dsp_ml_parameters;

    uint8_t * master_audio_buffer = params->master_audio_buffer;
    EventGroupHandle_t event_group_handle = params->event_group_handle;

    while (1)
    {
        xEventGroupWaitBits(event_group_handle,
                            AUDIO_RECORDING_DONE_BIT | ML_CLASSIFICATION_START_BIT, 
                            pdFALSE, pdTRUE, portMAX_DELAY);

        // TODO: Do filtering here, remove noise, etc.
        ESP_LOGI(ML_CLASSIFICATION_TASK_TAG, "Filtering audio..");

        uint32_t sum = 0;
        uint32_t num_of_samples = MASTER_AUDIO_BUFFER_SIZE/4;
        
        // Output ADC value for every captured ADC sample
        for (int i = 0; i <= MASTER_AUDIO_BUFFER_SIZE; i += ADC_OUTPUT_LEN)
        {
            adc_digi_output_data_t *sample = (adc_digi_output_data_t*)&master_audio_buffer[i];
            sum += sample->type2.data;
        }
        
        uint32_t dc_offset = sum/num_of_samples;

        ESP_LOGI(ML_CLASSIFICATION_TASK_TAG, "DC Offset: %d", dc_offset);

        ESP_LOGI(ML_CLASSIFICATION_TASK_TAG, "Audio filtering complete.");

        // TODO: Classify here via MFCC model

        xEventGroupSetBits(event_group_handle, ML_CLASSIFICATION_END_BIT);
        xEventGroupClearBits(event_group_handle, ML_CLASSIFICATION_START_BIT);
        vTaskDelay(pdMS_TO_TICKS(100));
    }
}
#include "cmn.h"
#include "dsp_ml_setup.h"

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include <cstring>

static const char *ML_CLASSIFICATION_TASK_TAG = "ML_CLASSIFICATION_TASK";

void ml_classification_task(void *dsp_ml_parameters)
{
    global_params* params = (global_params*)dsp_ml_parameters;

    uint8_t * master_audio_buffer = params->master_audio_buffer;
    EventGroupHandle_t event_group_handle = params->event_group_handle;

    while (1)
    {
        ESP_LOGI(ML_CLASSIFICATION_TASK_TAG, "Ready for ML classification.\n");

        xEventGroupWaitBits(event_group_handle,
                            AUDIO_RECORDING_DONE_BIT | ML_CLASSIFICATION_START_BIT, 
                            pdFALSE, pdTRUE, portMAX_DELAY);

        // TODO: Do filtering here, remove noise, etc.
        ESP_LOGI(ML_CLASSIFICATION_TASK_TAG, "Hello from ML Classification task! \n");

        // TODO: Classify here via MFCC model

        xEventGroupSetBits(event_group_handle, ML_CLASSIFICATION_END_BIT);
        vTaskDelay(pdMS_TO_TICKS(100));
    }
}
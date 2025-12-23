#include "cmn.h"
#include "ble_setup.h"

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include <cstring>

static const char *BLE_TASK_TAG = "BLE_TASK";

void ble_streaming_task(void *ble_parameters)
{
    global_params* params = (global_params*)ble_parameters;
    uint8_t * master_audio_buffer = params->master_audio_buffer; // AUDIO DATA WILL COME HERE
    EventGroupHandle_t event_group_handle = params->event_group_handle;

    while (1)
    {
        ESP_LOGI(BLE_TASK_TAG, "Ready for BLE streaming.\n");

        xEventGroupWaitBits(event_group_handle, 
                            AUDIO_RECORDING_DONE_BIT | BLE_STREAMING_START_BIT, 
                            pdFALSE, pdTRUE, portMAX_DELAY);
    
        // TODO: PUT CODE HERE
        ESP_LOGI(BLE_TASK_TAG, "Hello from BLE task! \n");

        xEventGroupSetBits(event_group_handle, BLE_STREAMING_END_BIT);
        vTaskDelay(pdMS_TO_TICKS(100));
    }
}
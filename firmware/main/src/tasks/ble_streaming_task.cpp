#include "ble_setup.h"
#include "NimBLEDevice.h"
#include "esp_log.h"
#include "cmn.h"
#include "mic_setup.h"

static const char* BLE_STREAMING_TASK_TAG = "BLE_STREAMING_TASK";

void ble_streaming_task(void *pvParameters) {
    task_params* params = (task_params*)pvParameters;
    EventGroupHandle_t event_group_handle = params->event_group_handle;
    uint8_t* master_audio_buffer = params->master_audio_buffer;
    NimBLECharacteristic* pAudioDataChar = params->pAudioDataChar;

    ESP_LOGI(BLE_STREAMING_TASK_TAG, "Starting BLE streaming task");

    while (1) {
        xEventGroupWaitBits(event_group_handle, 
                            ML_CLASSIFICATION_END_BIT | BLE_STREAMING_START_BIT, 
                            pdFALSE, // Don't clear bits yet
                            pdTRUE,  // Wait for both
                            portMAX_DELAY);

        ESP_LOGI(BLE_STREAMING_TASK_TAG, "Bursting audio to iPhone...");

        size_t total_size = MASTER_AUDIO_BUFFER_SIZE;
        size_t sent_bytes = 0;
        const size_t CHUNK_SIZE = 480; 

        while (sent_bytes < total_size) {
            size_t to_send = std::min(CHUNK_SIZE, total_size - sent_bytes);
            
            pAudioDataChar->setValue(&master_audio_buffer[sent_bytes], to_send);
            
            // Only increment sent_bytes if notify succeeded
            if(pAudioDataChar->notify()) {
                sent_bytes += to_send;
            } else {
                vTaskDelay(pdMS_TO_TICKS(20)); // Congestion handling
                continue; 
            }

            vTaskDelay(pdMS_TO_TICKS(12)); 
        }

        ESP_LOGI(BLE_STREAMING_TASK_TAG, "Transfer finished successfully.");

        xEventGroupSetBits(event_group_handle, BLE_STREAMING_END_BIT);
        xEventGroupClearBits(event_group_handle, BLE_STREAMING_START_BIT);
        vTaskDelay(pdMS_TO_TICKS(100));
    }
}
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
        // Wait for BOTH the recording to be done AND the streaming request to be active
        xEventGroupWaitBits(event_group_handle, 
                            AUDIO_RECORDING_DONE_BIT | BLE_STREAMING_START_BIT, 
                            pdFALSE, // Don't clear bits yet
                            pdTRUE,  // Wait for both
                            portMAX_DELAY);

        ESP_LOGI(BLE_STREAMING_TASK_TAG, "Bursting audio to iPhone...");

        size_t total_size = MASTER_AUDIO_BUFFER_SIZE;
        size_t sent_bytes = 0;
        const size_t CHUNK_SIZE = 480; 

        while (sent_bytes < total_size) {
            size_t to_send = std::min(CHUNK_SIZE, total_size - sent_bytes);
            
            // Set data and notify
            pAudioDataChar->setValue(&master_audio_buffer[sent_bytes], to_send);
            
            if(!pAudioDataChar->notify()) {
                // If congestion occurs, wait longer
                vTaskDelay(pdMS_TO_TICKS(20)); 
                continue; 
            }

            sent_bytes += to_send;
            
            // Stability delay: keeps the iOS BLE stack from crashing
            vTaskDelay(pdMS_TO_TICKS(12)); 
        }

        ESP_LOGI(BLE_STREAMING_TASK_TAG, "Transfer finished successfully.");

        // --- CLEANUP ---
        // Signal that we are done so the record task can take over again
        xEventGroupSetBits(event_group_handle, BLE_STREAMING_END_BIT);
        
        // Clear the start bits so we don't loop infinitely
        xEventGroupClearBits(event_group_handle, BLE_STREAMING_START_BIT);
        
        vTaskDelay(pdMS_TO_TICKS(100));
    }
}
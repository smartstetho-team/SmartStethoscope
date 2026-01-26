#include "ble_setup.h"
#include "NimBLEDevice.h"
#include "esp_log.h"
#include "cmn.h"
#include "mic_setup.h"

static const char* TAG = "BLE_STREAMING";

// Characteristics need to be accessible inside the loop
static NimBLECharacteristic* pHeartChar = nullptr;
static NimBLECharacteristic* pAudioDataChar = nullptr; 
static NimBLECharacteristic* pBatteryChar = nullptr;

#define SERVICE_UUID_STETHO      "0000abcd-0000-1000-8000-00805f9b34fb"
#define CHAR_UUID_AUDIO_DATA     "00001234-0000-1000-8000-00805f9b34fb"

class MyServerCallbacks : public NimBLEServerCallbacks {
    void onConnect(NimBLEServer* pServer, NimBLEConnInfo& connInfo) override {
        ESP_LOGI(TAG, "iPhone Connected!");
    };
    void onDisconnect(NimBLEServer* pServer, NimBLEConnInfo& connInfo, int reason) override {
        ESP_LOGI(TAG, "Disconnected. Restarting Advertising...");
        NimBLEDevice::startAdvertising();
    }
};

void ble_streaming_task(void *pvParameters) {
    task_params* params = (task_params*)pvParameters;
    EventGroupHandle_t event_group_handle = params->event_group_handle;
    uint8_t* master_audio_buffer = params->master_audio_buffer;

    // --- ONE-TIME SETUP (Outside the loop) ---
    NimBLEDevice::init("SmartStetho-S3");
    NimBLEDevice::setMTU(512); 

    NimBLEServer* pServer = NimBLEDevice::createServer();
    pServer->setCallbacks(new MyServerCallbacks());

    // Services
    NimBLEService* pBatteryService = pServer->createService("180f");
    pBatteryChar = pBatteryService->createCharacteristic("2a19", NIMBLE_PROPERTY::READ | NIMBLE_PROPERTY::NOTIFY);
    pBatteryService->start();

    NimBLEService* pHeartService = pServer->createService("180d");
    pHeartChar = pHeartService->createCharacteristic("2a37", NIMBLE_PROPERTY::NOTIFY);
    pHeartService->start();

    NimBLEService* pStethoService = pServer->createService(SERVICE_UUID_STETHO);
    pAudioDataChar = pStethoService->createCharacteristic(CHAR_UUID_AUDIO_DATA, NIMBLE_PROPERTY::NOTIFY);
    pStethoService->start();

    // Advertising
    NimBLEAdvertising* pAdvertising = NimBLEDevice::getAdvertising();
    pAdvertising->addServiceUUID("180d");
    pAdvertising->addServiceUUID(SERVICE_UUID_STETHO); 
    pAdvertising->setName("SmartStetho-S3");
    pAdvertising->start();

    ESP_LOGI(TAG, "BLE initialized. Waiting for recording bits...");

    while (1) {
        // Wait for BOTH the recording to be done AND the streaming request to be active
        xEventGroupWaitBits(event_group_handle, 
                            AUDIO_RECORDING_DONE_BIT | BLE_STREAMING_START_BIT, 
                            pdFALSE, // Don't clear bits yet
                            pdTRUE,  // Wait for both
                            portMAX_DELAY);

        // --- TODO: PUT CODE HERE ---
        ESP_LOGI(TAG, "Bursting audio to iPhone...");

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

        ESP_LOGI(TAG, "Transfer finished successfully.");

        // --- CLEANUP ---
        // Signal that we are done so the record task can take over again
        xEventGroupSetBits(event_group_handle, BLE_STREAMING_END_BIT);
        
        // Clear the start bits so we don't loop infinitely
        xEventGroupClearBits(event_group_handle, BLE_STREAMING_START_BIT);
        
        vTaskDelay(pdMS_TO_TICKS(100));
    }
}
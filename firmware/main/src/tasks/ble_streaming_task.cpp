#include "ble_setup.h"

#include "cmn.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "NimBLEDevice.h"

static const char *BLE_TASK_TAG = "BLE_TASK";
static const char *SERVICE_UUID = "a99a483c-c54e-4ec7-970f-01603a73ca20";
static const char *CHARACTERISTIC_UUID_TX = "9ae10c51-adcc-4171-9ec7-19e565e13fa3";

NimBLEServer* p_server = NULL;
NimBLEService* p_service = NULL;
NimBLEAdvertising* p_advertising = NULL;
NimBLECharacteristic* p_tx_characteristic = NULL;

void ble_init()
{
    NimBLEDevice::init("SmartStethoscope");
    NimBLEDevice::setPower(ESP_PWR_LVL_P9);
    
    p_server = NimBLEDevice::createServer();
    p_service = p_server->createService(SERVICE_UUID);

    p_tx_characteristic = p_service->createCharacteristic(
        CHARACTERISTIC_UUID_TX,
        NIMBLE_PROPERTY::NOTIFY
    );

    p_service->start();

    p_advertising = NimBLEDevice::getAdvertising();
    p_advertising->addServiceUUID(SERVICE_UUID);

    p_advertising->start();

    ESP_LOGI(BLE_TASK_TAG, "BLE Streaming Service Started!");
}

void ble_streaming_task(void *ble_parameters)
{
    ESP_LOGI(BLE_TASK_TAG, "Starting BLE streaming task");
    
    task_params* params = (task_params*)ble_parameters;
    uint8_t * master_audio_buffer = params->master_audio_buffer;
    EventGroupHandle_t event_group_handle = params->event_group_handle;

    while (1)
    {
        xEventGroupWaitBits(event_group_handle, 
                            AUDIO_RECORDING_DONE_BIT | BLE_STREAMING_START_BIT, 
                            pdFALSE, pdTRUE, portMAX_DELAY);
    
        // TODO: PUT CODE HERE
        ESP_LOGI(BLE_TASK_TAG, "Hello from BLE task!");

        xEventGroupSetBits(event_group_handle, BLE_STREAMING_END_BIT);
        xEventGroupClearBits(event_group_handle, BLE_STREAMING_START_BIT);
        vTaskDelay(pdMS_TO_TICKS(100));
    }
}
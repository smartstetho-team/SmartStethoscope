#include "drivers/button.h"
#include "drivers/lcd_display.h"
#include "cmn.h"
#include "mic_setup.h"
#include "ble_setup.h"
#include "dsp_ml_setup.h"
#include "lcd_ui_setup.h"
#include "debug.h"

#include <stdio.h>
#include <string.h>
#include <esp_heap_caps.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_adc/adc_continuous.h"
#include "esp_log.h"
#include "lvgl.h"
#include "NimBLEDevice.h"

// Global variables
static const char *MAIN_TAG = "MAIN";

task_params task_parameters = 
{
    .master_audio_buffer = NULL,
    .filtered_audio_buffer = NULL,
    .audio_dc_offset = 0,
    .mic_adc_handle = NULL,
    .event_group_handle = NULL,
    .inference_buffer_a = NULL,
    .inference_buffer_b = NULL,
    .pHeartChar = NULL,
    .pAudioDataChar = NULL,
    .pBatteryChar = NULL,
};

TaskHandle_t audio_sampling_task_handle = NULL;
TaskHandle_t ble_streaming_task_handle = NULL;
TaskHandle_t ml_classification_task_handle = NULL;
TaskHandle_t lcd_ui_task_handle = NULL;
TaskHandle_t debug_task_handle = NULL;

// Characteristics need to be accessible inside the loop
NimBLECharacteristic* pHeartChar = NULL;
NimBLECharacteristic* pAudioDataChar = NULL; 
NimBLECharacteristic* pBatteryChar = NULL;

class MyServerCallbacks : public NimBLEServerCallbacks {
    void onConnect(NimBLEServer* pServer, NimBLEConnInfo& connInfo) override {
        ESP_LOGI(MAIN_TAG, "iPhone Connected!");
    };
    void onDisconnect(NimBLEServer* pServer, NimBLEConnInfo& connInfo, int reason) override {
        ESP_LOGI(MAIN_TAG, "Disconnected. Restarting Advertising...");
        NimBLEDevice::startAdvertising();
    }
};

extern "C" void app_main(void) 
{
    // Set up driver for serial debugging
    debug_init();

    // Set up mutex for LVGL resources
    _lock_init(&task_parameters.lcd_params.lvgl_api_lock);

    // Configure LCD display
    configure_lcd_display(&task_parameters.lcd_params);

    // Configure LVGL library and timer
    configure_lcd_lvgl(&task_parameters.lcd_params);

    // Initialize bootup screen
    bootup_screen_init((void*)&task_parameters.lcd_params);

    // Create event group so tasks can talk with each other
    task_parameters.event_group_handle = xEventGroupCreate();

    if (task_parameters.event_group_handle == NULL)
    {
        ESP_LOGE(MAIN_TAG, "Can't create group event handle!");
    }

    // Configure Mic ADC
    configure_mic_adc(&task_parameters.mic_adc_handle);
    
    ESP_LOGI(MAIN_TAG, "PSRAM Size Before: %d", heap_caps_get_free_size(MALLOC_CAP_SPIRAM));

    // Allocate space for audio buffer in external RAM
    task_parameters.master_audio_buffer = (uint8_t*)heap_caps_malloc
                                          (MASTER_AUDIO_BUFFER_SIZE, MALLOC_CAP_SPIRAM);

    // Allocate space for filtered audio buffer in external RAM
    // Note: Size is halved since we only need ADC value instead of the whole packet
    task_parameters.filtered_audio_buffer = (float*)heap_caps_malloc
                                            (NUM_OF_SAMPLES * sizeof(float), MALLOC_CAP_SPIRAM);

    task_parameters.inference_buffer_a = (float*)heap_caps_malloc(1700000, MALLOC_CAP_SPIRAM);
    task_parameters.inference_buffer_b = (float*)heap_caps_malloc(1700000, MALLOC_CAP_SPIRAM);

    if (task_parameters.master_audio_buffer == NULL || 
        task_parameters.filtered_audio_buffer == NULL ||
        task_parameters.inference_buffer_a == NULL ||
        task_parameters.inference_buffer_b == NULL)
    {
        ESP_LOGE(MAIN_TAG, "PSRAM Allocation Failed! Critical Error. \\
                 Current Free PSRAM: %d bytes", heap_caps_get_free_size(MALLOC_CAP_SPIRAM));

        while(1) 
        { 
            vTaskDelay(pdMS_TO_TICKS(1000)); 
        }
    }

    ESP_LOGI(MAIN_TAG, "PSRAM Size After: %d", heap_caps_get_free_size(MALLOC_CAP_SPIRAM));

    // Set up BLE stack
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

    task_parameters.pAudioDataChar = pAudioDataChar;
    task_parameters.pBatteryChar = pBatteryChar;
    task_parameters.pHeartChar = pHeartChar;

    // Create all tasks
    xTaskCreatePinnedToCore(audio_sampling_task, "audio_sampling_task", 8192, 
                            (void*)&task_parameters, 5, &audio_sampling_task_handle, 0);

    BaseType_t rslt = xTaskCreatePinnedToCore(ml_classification_task, "ml_classification_task", 10240, 
                                            (void*)&task_parameters, 5, &ml_classification_task_handle, 1);
    if (rslt != pdPASS) 
    {
        ESP_LOGE(MAIN_TAG, "Failed to create ML Task! Error code: %d", rslt);
    }

    xTaskCreatePinnedToCore(ble_streaming_task, "ble_streaming_task", 8192, 
                        (void*)&task_parameters, 3, &ble_streaming_task_handle, 0);

    xTaskCreatePinnedToCore(lcd_ui_task, "lcd_ui_task", 10240, 
                            (void*)&task_parameters, 2, &lcd_ui_task_handle, 0);

    xTaskCreatePinnedToCore(debug_task, "debug_task", 8192, 
                            (void*)&task_parameters, 2, &debug_task_handle, 0);

    // Set up button for the audio task
    configure_push_button(audio_sampling_task_handle, (void*)&task_parameters);
}
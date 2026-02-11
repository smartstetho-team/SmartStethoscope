#ifndef BLE_SETUP_H
#define BLE_SETUP_H

#ifdef __cplusplus
extern "C" {
#endif

// BLE Identifiers
#define SERVICE_UUID_STETHO      "0000abcd-0000-1000-8000-00805f9b34fb"
#define CHAR_UUID_AUDIO_DATA     "00001234-0000-1000-8000-00805f9b34fb"

void ble_streaming_task(void *pvParameters);

#ifdef __cplusplus
}
#endif

#endif
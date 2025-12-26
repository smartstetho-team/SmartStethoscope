
#ifndef BLE_SETUP_H
#define BLE_SETUP_H

/* BLE Configuration */
#include <cstdint>

#define SEND_LEN 512

typedef struct 
{
    uint8_t data[SEND_LEN];
} ble_packet;

void ble_streaming_task(void *ble_parameters);

#endif /* BLE_SETUP_H */

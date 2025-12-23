

/* BLE Configuration */

#define SEND_LEN 512

// TODO: PUT CODE HERE

typedef struct 
{
    uint8_t data[SEND_LEN];
} ble_packet;

void ble_streaming_task(void *ble_parameters);

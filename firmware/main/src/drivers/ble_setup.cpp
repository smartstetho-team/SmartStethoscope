#include "drivers/ble_setup.h"
#include "nimble/nimble_port.h"
#include "nimble/nimble_port_esp32.h"
#include "host/ble_hs.h"
#include "services/gap/ble_svc_gap.h"
#include "services/gatt/ble_svc_gatt.h"
#include "esp_log.h"

static const char *BLE_TAG = "BLE_SETUP";
uint16_t heart_rate_handle;

// 1. Define the Service and Characteristic
static int gatt_svr_chr_access_heart_rate(uint16_t conn_handle, uint16_t attr_handle, struct ble_gatt_access_ctxt *ctxt, void *arg) {
    return 0; // The app primarily listens for NOTIFY, so we leave this empty
}

static const struct ble_gatt_svc_def gatt_svr_svcs[] = {
    {.type = BLE_GATT_SVC_TYPE_PRIMARY,
     .uuid = BLE_UUID16_DECLARE(0x180D), // Heart Rate Service
     .characteristics = (struct ble_gatt_chr_def[]){
         {.uuid = BLE_UUID16_DECLARE(0x2A37), // Heart Rate Measurement
          .access_cb = gatt_svr_chr_access_heart_rate,
          .flags = BLE_GATT_CHR_F_NOTIFY,
          .val_handle = &heart_rate_handle},
         {0}}},
    {0}};

// 2. Advertising Logic
void ble_app_advertise(void) {
    struct ble_gap_adv_params adv_params;
    struct ble_hs_adv_fields fields;
    memset(&fields, 0, sizeof(fields));

    fields.flags = BLE_HS_ADV_F_DISC_GEN | BLE_HS_ADV_F_BREDR_UNSUP;
    fields.name = (uint8_t *)"CardioScope";
    fields.name_len = strlen("CardioScope");
    fields.name_is_complete = 1;
    fields.uuids16 = (ble_uuid16_t[]){BLE_UUID16_INIT(0x180D)};
    fields.num_uuids16 = 1;
    fields.uuids16_is_complete = 1;

    ble_gap_adv_set_fields(&fields);
    memset(&adv_params, 0, sizeof(adv_params));
    adv_params.conn_mode = BLE_GAP_CONN_MODE_UND;
    adv_params.disc_mode = BLE_GAP_DISC_MODE_GEN;
    ble_gap_adv_start(0, NULL, BLE_HS_FOREVER, &adv_params, NULL, NULL);
}
#ifndef DSP_ML_SETUP_H
#define DSP_ML_SETUP_H

/* DSP+MFCC+ML Configuration */

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "lvgl.h"

#define MURMUR_THRESHOLD 0.5   // threshold used to classify sound as "Abnormal" or "Normal"
#define DIGITAL_GAIN 1.0f       // digital gain to amplify sound captured

// Global struct for LCD Updates
typedef struct {
    lv_obj_t *progress_bar;
    lv_obj_t *status_label;
    _lock_t *lvgl_lock;
} ui_update_handle_t;

void ml_classification_task(void *dsp_ml_parameters);

#endif /* DSP_ML_SETUP_H */
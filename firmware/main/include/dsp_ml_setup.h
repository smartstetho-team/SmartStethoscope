#ifndef DSP_ML_SETUP_H
#define DSP_ML_SETUP_H

/* DSP+MFCC+ML Configuration */

#define MURMUR_THRESHOLD 0.76   // threshold used to classify sound as "Abnormal" or "Normal"
#define DIGITAL_GAIN 1.0f       // digital gain to amplify sound captured

void ml_classification_task(void *dsp_ml_parameters);

#endif /* DSP_ML_SETUP_H */
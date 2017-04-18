//
// Created by 神经元002 on 2017/3/23.
//

#ifndef MCLDROID_BITMAPPROCESS_H
#define MCLDROID_BITMAPPROCESS_H

#include <jni.h>
#include <android/bitmap.h>
#include "MultiDimensionData.h"

void logBitmapInfo(AndroidBitmapInfo info);

void pixels2MDData(AndroidBitmapInfo * info, void *pixels, MultiDimensionData<float> *data);

void pixels2MDDataWithPreproccess(AndroidBitmapInfo * info, void *pixels, MultiDimensionData<float> *data);

void pixels2MultiDimensionData(AndroidBitmapInfo * info, void *pixels, MultiDimensionData<float> *data);

void pixels2MultiDimensionDataNeon(AndroidBitmapInfo * info, void *pixels, MultiDimensionData<float> *data);

void multiDimensionData2Pixels(AndroidBitmapInfo * info, void *pixels, MultiDimensionData<float> *data);

#endif //MCLDROID_BITMAPPROCESS_H

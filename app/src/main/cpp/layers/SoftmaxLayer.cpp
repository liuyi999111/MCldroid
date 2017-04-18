//
// Created by 神经元002 on 2017/3/20.
//

#include <utils.h>
#include "SoftmaxLayer.h"
#include "ComputeTool.h"

void SoftmaxLayer::compute(MultiDimensionData<float> *inputData, MultiDimensionData<float> *outputData){
    computeOutputShape(inputData, outputData);
    size_t batch = inputData->get_n();
    size_t channel = inputData->get_c();
//    softmaxNeon(inputData->data_ptr, outputData->data_ptr, batch, channel);
//    softmaxNnpack(inputData->data_ptr, outputData->data_ptr, batch, channel);
    softmaxC(inputData->data_ptr, outputData->data_ptr, batch, channel);
}
//输入与输出都是二维数据。
void SoftmaxLayer::computeOutputShape(MultiDimensionData<float> *inputData, MultiDimensionData<float> *outputData){
    if ( outputData->data_ptr != NULL){
        LOGE(" outputData->data_ptr != NULL, file:%s, line:%i", __FILE__, __LINE__);
    }
    float *outDataPtr = new float[inputData->totalSize()];
    outputData->setData(outDataPtr, inputData->get_n(), inputData->get_c(), inputData->get_h(), inputData->get_w());
}

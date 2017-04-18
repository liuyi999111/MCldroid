//
// Created by 神经元002 on 2017/4/12.
//

#include "LocalResponseNormalizationLayer.h"
#include <math.h>

// input1 / (((a/n) * input2 + 1 ) ^ beta)
void calculate(float * output, float * input1, float * input2,
               size_t size, int localSize, float alpha, float beta){
//    output = new float[size];
    const float p1 = alpha / localSize;
    for (int i = 0; i < size; ++i){
        output[i] = input1[i] / powf(p1 * input2[i] + 1, beta);
    }
}

float * sumChannel(const float * input, const size_t c, const size_t h, const size_t w ){
    float * output = new float[h * w];
    const size_t cSize = h * w;

    for (int indexH = 0; indexH < h; ++indexH) {
        for (int indexW = 0; indexW < w; ++indexW) {
            size_t indexBase = indexH * w + indexW;
            float temp = 0;
            for (int indexC = 0; indexC < c; ++indexC) {
                temp += input[indexC * cSize + indexBase];
            }

            output[indexBase] = temp;
        }
    }

    return output;
}

float * meanChannel(const float * input, const size_t c, const size_t h, const size_t w ){
    float * output = new float[h * w];
    const size_t cSize = h * w;

    for (int indexH = 0; indexH < h; ++indexH) {
        for (int indexW = 0; indexW < w; ++indexW) {
            size_t indexBase = indexH * w + indexW;
            float temp = 0;
            for (int indexC = 0; indexC < c; ++indexC) {
                temp += input[indexC * cSize + indexBase];
            }

            output[indexBase] = temp/(float)c;
        }
    }

    return output;
}

float *  power(const float * inputDataPtr,
           size_t n, size_t c, size_t c_l, size_t c_h, size_t h, size_t w, float p){

    float * outputDataPtr = new float[(c_h - c_l) * h * w];
    const size_t cSize = h * w;
    const size_t nSize = c * cSize;

    for (size_t indexC = c_l; indexC < c_h; ++indexC) {
        for (size_t indexH = 0; indexH < h; ++indexH){
            for (size_t indexW = 0; indexW < w; ++indexW) {
                outputDataPtr[(indexC - c_l) * cSize + indexH * w + indexW] =
                        powf(inputDataPtr[n * nSize + indexC * cSize + indexH * w + indexW], p);
            }
        }
    }

    return outputDataPtr;
}




void LocalResponseNormalizationLayer::compute(MultiDimensionData<float> *input,
                                              MultiDimensionData<float> *output) {
    computeOutputShape(input, output);

    const size_t n_i = input->get_n();
    const size_t c_i = input->get_c();
    const size_t h_i = input->get_h();
    const size_t w_i = input->get_w();

    const size_t cSize = h_i * w_i;
    const size_t nSize = c_i * cSize;

    float * outputDataPtr = output->data_ptr;
    float * inputDataPtr = input->data_ptr;

    const int localSize = params.localSize;
    const float alpha = params.alpha;
    const float beta = params.beta;

    for (size_t n = 0; n < n_i; ++n) {
        for (size_t c = 0; c < c_i; ++c) {
            const size_t baseIndex = n * nSize + c * cSize;

            bool zeroPadding = true;//do zero padding

            size_t c_l = c - (localSize - 1) / 2;
            size_t c_h = c + (localSize - 1) / 2 + 1;

            if (c < (localSize - 1)/2){
                c_l = 0;
            } else if (c > c_i - (localSize - 1)/2 - 1){
                c_h = c_i;
            } else {
                zeroPadding = false;
            }

            // xi^2
            float * tempPowResult = NULL;
            tempPowResult = power(inputDataPtr, n, c_i, c_l, c_h, h_i, w_i, 2);

            // sum(Xi^2)
            float * tempSumResult = NULL;
            if (zeroPadding){
                tempSumResult = sumChannel(tempPowResult, (c_h - c_l), h_i, w_i);
            } else{
                tempSumResult = meanChannel(tempPowResult, (c_h - c_l), h_i, w_i);
            }

            // Xi / (((a/n)*sum(Xi^2) + 1 ) ^ beta)
            calculate(&outputDataPtr[baseIndex], &inputDataPtr[baseIndex], tempSumResult,
                      cSize, localSize, alpha, beta);

            delete[] tempPowResult;
            delete[] tempSumResult;
        }
    }

}

void LocalResponseNormalizationLayer::computeOutputShape(MultiDimensionData<float> *input,
                                                         MultiDimensionData<float> *output) {

    // Calculate sizes.
    size_t n_i = input->get_n();
    size_t c_i = input->get_c();
    size_t h_i = input->get_h();
    size_t w_i = input->get_w();

    float * dataptr = new float[n_i * c_i * h_i *w_i];
    output->setData(dataptr, input->shape);
}

extern "C"{
JNIEXPORT jlong JNICALL
Java_com_compilesense_liuyi_mcldroid_mcldroid_LRNLayer_createLRNLayer(JNIEnv *env, jclass type,
                                                                      jstring name_, jint localSize,
                                                                      jfloat alpha, jfloat beta
) {
    const char *name = env->GetStringUTFChars(name_, 0);
    LocalResponseNormalizationLayer::Param params(localSize, alpha, beta);
    LocalResponseNormalizationLayer * layer =
            new LocalResponseNormalizationLayer(name, params);

    env->ReleaseStringUTFChars(name_, name);

    return (jlong) layer;
}
}


//
// Created by 神经元002 on 2017/3/21.
//

#ifndef MCLDROID_COMPUTETOOL_H
#define MCLDROID_COMPUTETOOL_H
void copyData(float * dst, float * src, size_t size);
void copyDataNeon(float * dst, float * src, size_t size);

bool initNnpack();

void releaseNnpack();

void softmaxNnpack(float * input_pointer,
                   size_t input_n, size_t input_c);

void softmaxNnpack(const float * input_pointer, float * output_pointer,
                   size_t input_n, size_t input_c);

void softmaxNeon(const float * input_pointer, float * output_pointer,
                 size_t input_n, size_t input_c);
void softmaxC(const float * input_pointer, float * output_pointer,
                  size_t input_n, size_t input_c);
int meanPoolingC(float *output, float *input,
                 size_t input_n, size_t input_c,
                 size_t input_h, size_t input_w,
                 size_t output_n, size_t output_c,
                 size_t output_h, size_t output_w,
                 size_t stride_h, size_t stride_w,
                 size_t pad_h, size_t pad_w,
                 size_t kernel_h, size_t kernel_w
);

void maxPoolingNnpack(
        float *output, float *input,
        size_t input_n, size_t input_c,
        size_t input_h, size_t input_w,
        size_t stride_h, size_t stride_w,
        size_t pad_h, size_t pad_w,
        size_t kernel_h, size_t kernel_w
);

int convNnpack(
        float * output, float * input, float * kernel, float * bias,
        size_t batch_size, size_t input_channels,
        size_t inputSize_h, size_t inputSize_w,
        size_t output_batch_size, size_t output_channels,
        size_t outputSize_h, size_t outputSize_w,
        size_t pad, size_t stride,
        size_t kernelSize_h, size_t kernelSize_w
);

int convNnpackWithReLu(
        float * output, float * input, float * kernel, float * bias,
        size_t batch_size, size_t input_channels,
        size_t inputSize_h, size_t inputSize_w,
        size_t output_batch_size, size_t output_channels,
        size_t outputSize_h, size_t outputSize_w,
        size_t pad, size_t stride,
        size_t kernelSize_h, size_t kernelSize_w, int nonlinearType
);

void relu(float *input, size_t totalSize);
//ai = 0.25
void prelu(float *input, size_t n_i, size_t c_i, size_t h_i, size_t  w_i, float *param);
void tanh(float *input, size_t totalSize);
void abs(float *input, size_t totalSize);

int fullyConnectC(float * output, float *input, float * kernel, float * bias,
                  int w_w, int c_o, int n_i, int c_i , int h_i , int w_i);

int fullyConnectNeon(float * output, float *input, float * kernel, float * bias,
                  int w_w, int c_o, int n_i, int c_i , int h_i , int w_i);

int fullyConnectNnpack(float * output, float * input, float * kernel,
                       size_t batch_size, size_t input_channels, size_t output_channels );


void elementSumNeon(float * * inputArray, float *output, float *coefficients,
                    size_t inputArrayLength, size_t outSize);
void elementProductNeon(float * * inputArray, float *output, float *coefficients,
                        size_t inputArrayLength, size_t outSize);
void elementMaxNeon(float * * inputArray, float *output, float *coefficients,
                    size_t inputArrayLength, size_t outSize);

#endif //MCLDROID_COMPUTETOOL_H

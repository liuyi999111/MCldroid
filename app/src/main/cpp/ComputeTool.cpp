//
// Created by 神经元002 on 2017/3/21.
//

#include <utils.h>
#include <ComputeTool.h>
#include <nnpack.h>
#include <arm_neon.h>
#include <math.h>

pthreadpool_t threadpool = NULL;
bool hadInitNnpack = false;//是否已经启动NNPACK

bool initNnpack(){
    //初始化 nnpack;
    nnp_status s = nnp_initialize();
    if (s != nnp_status_success){
        LOGD("nnpack init failed:%i", s);
        return false;
    } else{
        hadInitNnpack = true;
        return true;
    }
}

void releaseNnpack(){
    nnp_status s = nnp_deinitialize();
    if (s != nnp_status_success){
        LOGE("nnpack release failed:%i", s);
    } else{
        hadInitNnpack = false;
    }
}

void checkAndInitNnpack(unsigned int line){
    if (!hadInitNnpack){
        if (!initNnpack()){
            LOGE("!initNnpack(), file:%s, line:%i", __FILE__, line);
        }
    }
}

void softmaxNnpack(float * input_pointer,
                   size_t input_n, size_t input_c){
    checkAndInitNnpack(__LINE__);
    //pthreadpool_create(4); //nnpack运行线程数量

    nnp_softmax_output(
            input_n,
            input_c,
            input_pointer,
            input_pointer,
            threadpool);

}

void softmaxNnpack(const float * input_pointer, float * output_pointer,
                   size_t input_n, size_t input_c){
    checkAndInitNnpack(__LINE__);
     //pthreadpool_create(4); //nnpack运行线程数量

    nnp_softmax_output(
            input_n,
            input_c,
            input_pointer,
            output_pointer,
            threadpool);

}

void softmaxNeonDiff(
        const float * input_pointer, float * output_pointer,
        int input_c, const float sumN1
){
    const int numVector = input_c / 4;
    int left = input_c % 4;
    int batchIndex = numVector;
    int index = 0;
    while(batchIndex > 0){
        batchIndex--;
        float32x4_t vector1 = vld1q_f32(input_pointer + index);
        float32x4_t result = vmulq_n_f32(vector1, sumN1);
        vst1q_f32(output_pointer + index, result);
        index += 4;
    }
    while(left > 0){
        left--;
        output_pointer[index] = input_pointer[index] * sumN1;
        index++;
    }
}

void softmaxNeon(const float * input_pointer, float * output_pointer,
                 size_t input_n, size_t input_c){

    for (int i = 0; i < input_n; ++i){
        size_t indexN = i * input_c;
        float sum = 0;
        for (int j = 0; j < input_c; ++ j){
            float temp = powf(2.71828, input_pointer[indexN + j]);
            output_pointer[indexN + j] = temp;
            sum += temp;
        }
        if (sum == 0){
            LOGE("sum == 0, file:%s, line:%i", __FILE__, __LINE__);
            continue;
        }
        softmaxNeonDiff(&input_pointer[indexN], &output_pointer[indexN],
                        (int)input_c, (1.0f/sum));
    }
}

void softmaxC(const float * input_pointer, float * output_pointer,
                 size_t input_n, size_t input_c){

    for (int i = 0; i < input_n; ++i){
        size_t indexN = i * input_c;
        float sum = 0;
        for (int j = 0; j < input_c; ++ j){
            float temp = powf(2.71828, input_pointer[indexN + j]);
            output_pointer[indexN + j] = temp;
            sum += temp;
        }
        for (int j = 0; j < input_c; ++ j){
            if(isinf(output_pointer[indexN + j])){
                output_pointer[indexN + j] = 1;
                continue;
            }
            output_pointer[indexN + j] /= sum;
        }
    }
}


void meanPooling(
        const float * input, float * output,
        size_t input_height, size_t input_width,
        size_t output_height, size_t output_width,
        size_t padding_top, size_t padding_left,
        size_t stride_height, size_t stride_width,
        size_t pooling_height, size_t pooling_width)
{


    for (size_t y = 0; y < output_height; y++) {
        for (size_t x = 0; x < output_width; x++) {


            float sum = 0;
            for (size_t i = 0; i < pooling_height; i++) {
                const size_t s = y * stride_height + i - padding_top;

                if (s < input_height) {
                    for (size_t j = 0; j < pooling_width; j++) {
                        const size_t t = x * stride_width + j - padding_left;
                        if (t < input_width) {
                            sum += input[s * input_width +t];
                        }
                    }
                }
            }
            output[y * output_width + x] = sum/(pooling_width * pooling_height);


        }
    }
}

int meanPoolingC(float *output, float *input,
                 size_t input_n, size_t input_c,
                 size_t input_h, size_t input_w,
                 size_t output_n, size_t output_c,
                 size_t output_h, size_t output_w,
                 size_t stride_h, size_t stride_w,
                 size_t pad_h, size_t pad_w,
                 size_t kernel_h, size_t kernel_w
){
    const size_t cSize_i = input_h * input_w;
    const size_t nSize_i = input_c * cSize_i;
    const size_t cSize_o = output_h * output_w;
    const size_t nSize_o = output_c * cSize_o;

    for (int n = 0; n < output_n; ++n) {
        for (int c = 0; c < output_c; ++c) {
            int indexInput = (int) (n * nSize_i + c * cSize_i);
            int indexOutput = (int) (n * nSize_o + c * cSize_o);
            meanPooling(&input[indexInput], &output[indexOutput],
                        input_h, input_w,
                        output_h, output_w,
                        pad_h, pad_w,
                        stride_h, stride_w,
                        kernel_h, kernel_w
            );
        }
    }

    return 0;
}

void maxPoolingNnpack(
        float *output, float *input,
        size_t input_n, size_t input_c,
        size_t input_h, size_t input_w,
        size_t stride_h, size_t stride_w,
        size_t pad_h, size_t pad_w,
        size_t kernel_h, size_t kernel_w
){
    checkAndInitNnpack(__LINE__);
    nnp_size input_size;
    input_size.height = input_h;
    input_size.width = input_w;
    nnp_padding input_padding;
    input_padding.left = pad_w;
    input_padding.right = pad_w;
    input_padding.top = pad_h;
    input_padding.bottom = pad_h;
    nnp_size pooling_size;
    pooling_size.width = kernel_w;
    pooling_size.height = kernel_h;
    nnp_size pooling_stride;
    pooling_stride.height = stride_h;
    pooling_stride.width = stride_w;
    nnp_max_pooling_output(
            input_n,
            input_c,
            input_size,
            input_padding,
            pooling_size,
            pooling_stride,
            input,
            output,
            threadpool);
}

//用于计算有 stride 的情况, 一次只能执行一个batch.
int convNnpackWithStride(
        float * output, float * input, float * kernel, float * bias,
        size_t input_channels, size_t output_channels,
        size_t pad,size_t stride, size_t inputSize_h, size_t inputSize_w,
        size_t kernelSize_h, size_t kernelSize_w
){
    checkAndInitNnpack(__LINE__);

    enum nnp_convolution_algorithm algorithm = nnp_convolution_algorithm_auto;//卷积算法
    enum nnp_convolution_transform_strategy transform_strategy = nnp_convolution_transform_strategy_block_based;
    struct nnp_size input_size = {inputSize_h, inputSize_w };
    struct nnp_padding input_padding = {pad,pad,pad,pad};
    struct nnp_size kernel_size = {kernelSize_h, kernelSize_w};
    struct nnp_size output_subsampling = {stride, stride};

    nnp_convolution_inference(
            algorithm,
            transform_strategy,
            input_channels,
            output_channels,
            input_size,
            input_padding,
            kernel_size,
            output_subsampling,
            input,
            kernel,
            bias,
            output,
            nnp_activation_identity,
            NULL,
            threadpool,
            NULL);
}

int convNnpack(
        float * output, float * input, float * kernel, float * bias,
        size_t batch_size, size_t input_channels,
        size_t inputSize_h, size_t inputSize_w,
        size_t output_batch_size, size_t output_channels,
        size_t outputSize_h, size_t outputSize_w,
        size_t pad, size_t stride,
        size_t kernelSize_h, size_t kernelSize_w
){
    checkAndInitNnpack(__LINE__);

    enum nnp_convolution_algorithm algorithm = nnp_convolution_algorithm_auto;//卷积算法
    enum nnp_convolution_transform_strategy transform_strategy = nnp_convolution_transform_strategy_tuple_based;
    struct nnp_size input_size = {inputSize_h, inputSize_w };
    struct nnp_padding input_padding = {pad,pad,pad,pad};
    struct nnp_size kernel_size = {kernelSize_h, kernelSize_w};
    struct nnp_size output_subsampling = {stride, stride};

    nnp_convolution_inference(
            algorithm,
            transform_strategy,
            input_channels,
            output_channels,
            input_size,
            input_padding,
            kernel_size,
            output_subsampling,
            input,
            kernel,
            bias,
            output,
            nnp_activation_identity,
            NULL,
            threadpool,
            NULL);
    return 0;
}

int convNnpackWithReLu(
        float * output, float * input, float * kernel, float * bias,
        size_t batch_size, size_t input_channels,
        size_t inputSize_h, size_t inputSize_w,
        size_t output_batch_size, size_t output_channels,
        size_t outputSize_h, size_t outputSize_w,
        size_t pad, size_t stride,
        size_t kernelSize_h, size_t kernelSize_w, int nonlinearType
){
    checkAndInitNnpack(__LINE__);
    enum nnp_convolution_algorithm algorithm = nnp_convolution_algorithm_auto;//卷积算法
    enum nnp_convolution_transform_strategy transform_strategy = nnp_convolution_transform_strategy_tuple_based;
    struct nnp_size input_size = {inputSize_h, inputSize_w };
    struct nnp_padding input_padding = {pad,pad,pad,pad};
    struct nnp_size kernel_size = {kernelSize_h, kernelSize_w};
    struct nnp_size output_subsampling = {stride, stride};

    nnp_convolution_inference(
            algorithm,
            transform_strategy,
            input_channels,
            output_channels,
            input_size,
            input_padding,
            kernel_size,
            output_subsampling,
            input,
            kernel,
            bias,
            output,
            nnp_activation_relu,
            NULL,
            threadpool,
            NULL);
    return 0;
}

void relu(float *input, size_t totalSize){
    if(input == NULL ){
        return;
    }
    int num = (int) totalSize;
    //一个向量寄存器可128位装4个32位 float。共有16个
    const int numVector = num / 4;
    int left = num % 4;
    int batchIndex = numVector;
    int index = 0;

    float * data = input;
    while(batchIndex > 0){
        batchIndex--;
        float32x4_t vector1 = vld1q_f32(data + index);
        float32x4_t vector1Abs = vabsq_f32(vector1);
        float32x4_t vector1Sum= vaddq_f32(vector1 , vector1Abs);
        float32x4_t result = vmulq_n_f32(vector1Sum, 0.5);
        vst1q_f32(data + index, result);
        index += 4;
    }

    while(left > 0){
        left--;
        float temp = data[index];
        data[index] = (temp > 0) ? temp : 0;
        index++;
    }
}

void prelu(float *input, size_t n_i, size_t c_i, size_t h_i, size_t  w_i, float *param){


    if(input == NULL ){
        return;
    }

    const size_t cSize = h_i * w_i;
    const size_t nSize = c_i * cSize;

    for (int n = 0; n < n_i; ++n){
        for (int c = 0; c < c_i; ++c) {
            const float P = param[c];
            //一个向量寄存器可128位装4个32位 float。共有16个
            const int numVector = (int) cSize / 4;
            int left = (int) cSize % 4;
            int batchIndex = numVector;
            int index = 0;

            float * data = &input[n * nSize + c * cSize];
            while(batchIndex > 0){
                batchIndex--;
                float32x4_t vector = vld1q_f32(data + index);//v
                float32x4_t vectorAbs = vabsq_f32(vector);//|v|
                float32x4_t vectorSub = vsubq_f32(vector, vectorAbs);// v - |v|
                float32x4_t vectorTemp1 = vmulq_n_f32(vectorSub, P * 0.5f);//(v - |v|) * 0.5 * ai
                float32x4_t vectorAdd = vaddq_f32(vector, vectorAbs);// v + |v|
                float32x4_t vectorTemp2 = vmulq_n_f32(vectorAdd, 0.5f);
                float32x4_t result = vaddq_f32(vectorTemp1, vectorTemp2);
                vst1q_f32(data + index, result);
                index += 4;
            }

            while(left > 0){
                left--;
                float temp = data[index];
                data[index] = (temp > 0) ? temp : temp * P;
                index++;
            }
        }
    }

}

void tanh(float *input, size_t totalSize){
    if(input == NULL){
        return;
    }
    int num = (int) totalSize;
    float * data = input;
    for (int i = 0; i < num; ++i) {
        data[i] = tanhf(data[i]);
    }

}

void abs(float *input, size_t totalSize){
    if(input == NULL){
        return;
    }
    int num = (int) totalSize;
    //一个向量寄存器可128位装4个32位 float。共有16个
    const int numVector = num / 4;
    int left = num % 4;
    int batchIndex = numVector;
    int index = 0;

    float * data = input;
    while(batchIndex > 0){
        batchIndex--;
        float32x4_t vector1 = vld1q_f32(data + index);
        float32x4_t result = vabsq_f32(vector1);
        vst1q_f32(data + index, result);
        index += 4;
    }

    while(left > 0){
        left--;
        float temp = data[index];
        data[index] = fabsf(temp);
        index++;
    }
}

int fullyConnectC(float * output, float *input, float * kernel, float * bias,
                   int w_w, int c_o, int n_i, int c_i , int h_i , int w_i){
    const int cSize_i = h_i * w_i;
    const int nSize_o = c_o * n_i;

    for (int n = 0; n < n_i; n++){
        for (int c = 0; c < c_o; c++){
            float sum = 0;
            int wIter = c;
            for (int i = 0 ; i < c_i ; ++i){
                for (int j = 0 ; j < h_i ; ++j){
                    for (int k = 0 ; k < w_i ; ++k){
                        sum += input[i*cSize_i + j*w_i + k] * kernel[wIter * w_w + i * h_i * w_i + j * w_i + k];
                    }
                }
            }
            output[n * nSize_o + c] = sum + bias[c];
        }
    }

    return 0;
}

// w_w 指每一贞的大小
int fullyConnectNeon(float * output, float *input, float * kernel, float * bias,
                  int w_w, int c_o, int n_i, int c_i , int h_i , int w_i){
    const int hSize_i = w_i;
    const int cSize_i = h_i * w_i;
    const int nSize_i = cSize_i * c_i;
    const int nSize_o = c_o * n_i;

    const int VECTOR_SIZE = 4;
    const int BATCH = nSize_i / VECTOR_SIZE;
    const int LEFT = nSize_i % VECTOR_SIZE;

//    if (BATCH <= 0){
//        for (int n = 0; n < n_i; n++){
//            for (int c = 0; c < c_o; c++){
//                float sum = 0;
//                int wIter = c * w_w;//kernel 中 src Chanel 的 index.
//                int baseIndex = n * nSize_i;
//                for (int i = 0; i < nSize_i; ++i) {
//                    float inputData = input[baseIndex + i];
//                    float kernelData = kernel[baseIndex + wIter + i];
//                    sum += inputData * kernelData;
//                }
//                output[n * nSize_o + c] = sum + bias[c];
//            }
//        }
//        return 0;
//    }

    for (int n = 0; n < n_i; n++){
        int baseIndex = n * nSize_i;
        for (int c = 0; c < c_o; c++){
            float sum = 0;
            int wIter = c * w_w;//kernel 中 src Chanel 的 index.

            for (int i = 0; i < BATCH * VECTOR_SIZE; i += VECTOR_SIZE){
                float32x4_t vInput = vld1q_f32(input + baseIndex + i);
                float32x4_t vKernel = vld1q_f32(kernel + wIter + baseIndex + i);
                float32x4_t  vTemp = vmulq_f32(vInput, vKernel);
                float r = vaddvq_f32(vTemp);
                sum += r;
            }

            if (LEFT > 0){
                int baseIndex_ = baseIndex + BATCH * VECTOR_SIZE;
                for (int i = 0; i < nSize_i; ++i) {
                    float inputData = input[baseIndex_ + i];
                    float kernelData = kernel[baseIndex_ + wIter + i];
                    sum += inputData * kernelData;
                }
            }
            output[n * nSize_o + c] = sum + bias[c];
        }
    }

    return 0;
}

int fullyConnectNnpack(float * output, float * input, float * kernel,
                       size_t batch_size, size_t input_channels, size_t output_channels ){
    checkAndInitNnpack(__LINE__);

    if (batch_size == 1){
        nnp_fully_connected_inference(
                input_channels,
                output_channels,
                input,
                kernel,
                output,
                NULL
        );

    } else{
        nnp_fully_connected_output(
                batch_size,
                input_channels,
                output_channels,
                input,
                kernel,
                output,
                threadpool,
                NULL
        );
    }
    return 0;
}

void _2elementSumNeon(float *inputA, float *inputB, float *output,
                      const float cfA, const float cfB, size_t size){
    size_t numVector = size / 4;
    size_t left = size % 4;
    size_t index = 0;
//    LOGD("_2elementSumNeon numVector:%lu, left:%lu",numVector, left);
//    LOGD("_2elementSumNeon cfA:%f, cfB:%f",cfA,cfB);

    while(numVector > 0){
        numVector--;
        float32x4_t vectorA = vld1q_f32(inputA + index);
        float32x4_t vectorB = vld1q_f32(inputB + index);
        vectorA = vmulq_n_f32(vectorA, cfA);
        vectorB = vmulq_n_f32(vectorB, cfB);
        float32x4_t result = vaddq_f32(vectorA, vectorB);
        vst1q_f32(output + index, result);
        index += 4;
    }
    while(left > 0){
        left--;
        inputA[index] = inputA[index] * cfA + inputB[index] * cfB;
        index++;
    }
}
//输出到 inputA
void _2elementSumNeon(float *inputA, float *inputB,
                      const float cfA, const float cfB, size_t size){
    size_t numVector = size / 4;
    size_t left = size % 4;
    size_t index = 0;
    while(numVector > 0){
        numVector--;
        float32x4_t vectorA = vld1q_f32(inputA + index);
        float32x4_t vectorB = vld1q_f32(inputB + index);
        vectorA = vmulq_n_f32(vectorA, cfA);
        vectorB = vmulq_n_f32(vectorB, cfB);
        float32x4_t result = vaddq_f32(vectorA, vectorB);
        vst1q_f32(inputA + index, result);
        index += 4;
    }
    while(left > 0){

        inputA[index] = inputA[index] * cfA + inputB[index] * cfB;
        left--;
    }
}

void _2elementSumNeon(float *inputA, float *inputB,
                      const float cfB, size_t size){
    size_t numVector = size / 4;
    size_t left = size % 4;
    size_t index = 0;
    while(numVector > 0){
        numVector--;
        float32x4_t vectorA = vld1q_f32(inputA + index);
        float32x4_t vectorB = vld1q_f32(inputB + index);
        vectorB = vmulq_n_f32(vectorB, cfB);
        float32x4_t result = vaddq_f32(vectorA, vectorB);
        vst1q_f32(inputA + index, result);
        index += 4;
    }
    while(left > 0){
        inputA[index] = inputA[index] + inputB[index] * cfB;
        index++;
        left--;
    }
}

void elementSumNeon(float * * inputArray, float *output, float *coefficients,
                    const size_t inputArrayLength, const size_t outSize){
    if (inputArrayLength < 2){
        LOGE("inputArrayLength < 2  file:%s, line:%i", __FILE__, __LINE__);
        return;
    }
    size_t index = 0;
    _2elementSumNeon(inputArray[index],
                     inputArray[index+1],
                     output,
                     coefficients[index],
                     coefficients[index + 1],
                     outSize);
    index += 2;
    while (index < inputArrayLength){
        _2elementSumNeon(output, inputArray[index], coefficients[index], outSize);
        index++;
    }
}


void _2elementProductNeon(float *inputA, float *inputB, float *output,
                      const float cfA, const float cfB, size_t size){
    size_t numVector = size / 4;
    size_t left = size % 4;
    size_t index = 0;
    while(numVector > 0){
        numVector--;
        float32x4_t vectorA = vld1q_f32(inputA + index);
        float32x4_t vectorB = vld1q_f32(inputB + index);
        vectorA = vmulq_n_f32(vectorA, cfA);
        vectorB = vmulq_n_f32(vectorB, cfB);
        float32x4_t result = vmulq_f32(vectorA, vectorB);
        vst1q_f32(output + index, result);
        index += 4;
    }
    while(left > 0){
        left--;
        inputA[index] = inputA[index] * cfA + inputB[index] * cfB;
        index++;
    }
}

void _2elementProductNeon(float *inputA, float *inputB,
                          const float cfB, size_t size){
    size_t numVector = size / 4;
    size_t left = size % 4;
    size_t index = 0;
    while(numVector > 0){
        numVector--;
        float32x4_t vectorA = vld1q_f32(inputA + index);
        float32x4_t vectorB = vld1q_f32(inputB + index);
        vectorB = vmulq_n_f32(vectorB, cfB);
        float32x4_t result = vmulq_f32(vectorA, vectorB);
        vst1q_f32(inputA + index, result);
        index += 4;
    }
    while(left > 0){
        left--;
        inputA[index] = inputA[index] + inputB[index] * cfB;
        index++;
    }
}

void elementProductNeon(float * * inputArray, float *output, float *coefficients,
                        size_t inputArrayLength, size_t outSize){
    if (inputArrayLength < 2){
        LOGE("inputArrayLength < 2  file:%s, line:%i", __FILE__, __LINE__);
        return;
    }
    size_t index = 0;
    _2elementProductNeon(inputArray[index],
                     inputArray[index+1],
                     output,
                     coefficients[index],
                     coefficients[index + 1],
                     outSize);
    index += 2;
    while (index < inputArrayLength){
        _2elementProductNeon(output, inputArray[index], coefficients[index], outSize);
        index++;
    }
}

void _2elementMaxNeon(float *inputA, float *inputB, float *output,
                          const float cfA, const float cfB, size_t size){
    size_t numVector = size / 4;
    size_t left = size % 4;
    size_t index = 0;
    while(numVector > 0){
        numVector--;
        float32x4_t vectorA = vld1q_f32(inputA + index);
        float32x4_t vectorB = vld1q_f32(inputB + index);
        vectorA = vmulq_n_f32(vectorA, cfA);
        vectorB = vmulq_n_f32(vectorB, cfB);
        float32x4_t result = vmaxq_f32(vectorA, vectorB);
        vst1q_f32(output + index, result);
        index += 4;
    }
    while(left > 0){
        left--;
        inputA[index] = inputA[index] * cfA + inputB[index] * cfB;
        index++;
    }
}
void _2elementMaxNeon(float *inputA, float *inputB,
                      const float cfB, size_t size){
    size_t numVector = size / 4;
    size_t left = size % 4;
    size_t index = 0;
    while(numVector > 0){
        numVector--;
        float32x4_t vectorA = vld1q_f32(inputA + index);
        float32x4_t vectorB = vld1q_f32(inputB + index);
        vectorB = vmulq_n_f32(vectorB, cfB);
        float32x4_t result = vmaxq_f32(vectorA, vectorB);
        vst1q_f32(inputA + index, result);
        index += 4;
    }
    while(left > 0){
        left--;
        inputA[index] = inputA[index] + inputB[index] * cfB;
        index++;
    }
}
void elementMaxNeon(float * * inputArray, float *output, float *coefficients,
                    size_t inputArrayLength, size_t outSize){
    if (inputArrayLength < 2){
        LOGE("inputArrayLength < 2  file:%s, line:%i", __FILE__, __LINE__);
        return;
    }
    size_t index = 0;
    _2elementMaxNeon(inputArray[index],
                         inputArray[index+1],
                         output,
                         coefficients[index],
                         coefficients[index + 1],
                         outSize);
    index += 2;
    while (index < inputArrayLength){
        _2elementMaxNeon(output, inputArray[index], coefficients[index], outSize);
        index++;
    }
}
//
// Created by 神经元002 on 2017/3/3.
//

#ifndef CAFFEMODELINPUT_CONVOLUTIONLAYER_H
#define CAFFEMODELINPUT_CONVOLUTIONLAYER_H

#include <string>
#include <vector>
#include <MultiDimensionData.h>
#include <BaseLayer.h>


class ConvolutionLayer : public BaseLayer{
public:
    class Params {
    public:
        size_t group;
        size_t pad,pad_h,pad_w;
        size_t stride,stride_h,stride_w;
        size_t dilation,dilation_h,dilation_w; //暂时输入没有

        Params(size_t pad = 0, size_t stride = 1, size_t dilation = 1,size_t group = 1) {
            this->pad = pad;
            pad_h = pad;
            pad_w = pad;
            this->stride = stride;
            stride_h = stride;
            stride_w = stride;
            this->dilation = dilation;
            dilation_h = dilation;
            dilation_w = dilation;
        }

        void setParams(size_t pad_, size_t stride_, size_t dilation_, size_t group_ ){
            pad_h = pad_w = pad = pad_;
            stride_h = stride_w = stride = stride_;
            dilation_h = dilation_w = dilation = dilation_;
            group = group_;
        }
    };

    ConvolutionLayer(const std::string &name, size_t stride, size_t pad, size_t group, bool nonLinear)
            : BaseLayer(name), nonLinear(nonLinear), paramHadLoad(false){
        params.setParams(pad, stride, 1, 1);
        kernelSholdRelease = true;
    }

    ConvolutionLayer(const std::string &name, ConvolutionLayer::Params params, bool nonLinear)
            : BaseLayer(name), params(params), nonLinear(nonLinear){
        kernelSholdRelease = true;
    };

    ~ConvolutionLayer() {
        LOGD("ConvolutionLayer delete!");
        releaseKernel();
    }

    void setKernel(float *weight, std::vector<size_t> weightShape,
                   float *bias, int biasSize);
    void setKernel(bool shouldRelease,
                   float *weight, std::vector<size_t> weightShape,
                   float *bias, int biasSize);
    void releaseKernel();

    /**
     * 本地直接读取数据
     * 之前的 java 层加载 model 数据在通过 JNI 传递数据的方式显然是不优雅的,
     * 将 model 数据用 msgpack 序列化后,用这个方法读取文件中的数据。
     */
    void loadKernelNative(std::string filePath);

    void compute(MultiDimensionData<float> *input, MultiDimensionData<float> *output);

protected:
    void computeOutputShape(MultiDimensionData<float> *input, MultiDimensionData<float> * output);

private:
    bool kernelSholdRelease;
    bool nonLinear; //是否有非线性部分
    ConvolutionLayer::Params params;
    bool paramHadLoad;
    MultiDimensionData<float> weight;
    MultiDimensionData<float> bias;
};

#endif //CAFFEMODELINPUT_CONVOLUTIONLAYER_H

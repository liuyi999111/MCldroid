//
// Created by 神经元002 on 2017/4/12.
//

#ifndef MCLDROID_LOCALRESPONSENORMALIZATIONLAYER_H
#define MCLDROID_LOCALRESPONSENORMALIZATIONLAYER_H


#include "BaseLayer.h"

class LocalResponseNormalizationLayer : BaseLayer{
public:
    class Param {
    public:
        int localSize;
        float alpha;
        float beta;

        Param(int localSize, float alpha, float beta) : localSize(localSize), alpha(alpha),
                                                        beta(beta) { }

        void setLocalSize(int localSize_) {
            localSize = localSize_;
        }

        void setAlpha(float alpha_) {
            alpha = alpha_;
        }

        void setBeta(float beta_) {
            beta = beta_;
        }
    };


    LocalResponseNormalizationLayer(const std::string &name, const Param &params) : BaseLayer(name),
                                                                                    params(params) { }

    void compute(MultiDimensionData<float> *input, MultiDimensionData<float> *output);
protected:
    /**
     * 计算每一次输出的Shape, 并申请输出需站的内存。
     */
    void computeOutputShape(MultiDimensionData<float> *input, MultiDimensionData<float> * output);

private:
    LocalResponseNormalizationLayer::Param params;

};


#endif //MCLDROID_LOCALRESPONSENORMALIZATIONLAYER_H

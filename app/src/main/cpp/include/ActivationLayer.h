//
// Created by 神经元002 on 2017/3/10.
//

#ifndef CAFFEMODELINPUT_ACTIVATIONLAYER_H
#define CAFFEMODELINPUT_ACTIVATIONLAYER_H

#include <MultiDimensionData.h>
#include <string>
#include <BaseLayer.h>

class ActivationLayer : public BaseLayer{
public:
    enum Type {ReLU,PReLU,TanH,Abs};
    ActivationLayer(const std::string name, Type type_)
            : BaseLayer(name),type(type_) , paramHadLoad(false){ };
    void loadParamsNative(std::string filePath);
    void compute(MultiDimensionData<float> *input);
    void compute(MultiDimensionData<float> *input, MultiDimensionData<float> *output){
        compute(input);
        output->setData(input->data_ptr,input->shape);//确保 output 为输出数据,且与 input 保有不同的地址数据
        input->setData(NULL, 1,1,1,1);
    }
protected:
    void computeOutputShape(MultiDimensionData<float> *input, MultiDimensionData<float> * output){
        //空
    }
private:
    Type type;
    bool paramHadLoad;
    MultiDimensionData<float> params;
};


#endif //CAFFEMODELINPUT_ACTIVATIONLAYER_H

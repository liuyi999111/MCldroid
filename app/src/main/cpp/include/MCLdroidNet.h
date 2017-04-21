//
// Created by 神经元002 on 2017/3/23.
//

#ifndef MCLDROID_NET_H
#define MCLDROID_NET_H


#include <vector>
#include <BaseLayer.h>
//适用于类似队列的网络
class Net {
public:
    Net():layers(NULL),layerIndex(-1),layerSize(0){};

    void setUpNet( long long *layers, size_t layerSize){
        if (layers == NULL || layerSize <= 0){
            return;
        }
        this->layers = layers;
        this->layerSize = layerSize;
        this->layerIndex = 0;
    };

    void forward(MultiDimensionData<float> *input, MultiDimensionData<float> *output);

    void reStart(){
        layerIndex = 0;
    }

    BaseLayer* getLayerInOrder(){
        if (layerIndex >= layerSize){
            return NULL;
        }
        BaseLayer* r = (BaseLayer*) layers[layerIndex];
        layerIndex++;
        return r;
    }

    BaseLayer* getLastLayer(){
        BaseLayer * r = (BaseLayer *) layers[layerSize - 1];
        return r;
    }

private:
    long long *layers;
    int layerIndex;
    size_t layerSize;
};

//网络成树状
class Net_Tree {
public:
    Net_Tree():netHeader(NULL),layerSize(0),numNetOutput(0),indexOutPut(0),outputLayers(NULL){};
    ~Net_Tree(){
        if (outputLayers != NULL){
            delete[] outputLayers;
        }
    }

    void setUp(BaseLayer * netHeader, size_t layerSize, size_t numNetOutput){
        this->netHeader = netHeader;
        this->layerSize = layerSize;
        this->numNetOutput = numNetOutput;
        indexOutPut = 0;
        outputLayers = new BaseLayer* [numNetOutput];
    }


    void forward(MultiDimensionData<float> *input, MultiDimensionData<float> **output, size_t num_netOutput);

    size_t getNumOutput(){
        return numNetOutput;
    }

    size_t indexOutPut;
    BaseLayer ** outputLayers;
private:
    BaseLayer * netHeader;
    size_t layerSize;
    size_t numNetOutput;

};


#endif //MCLDROID_NET_H

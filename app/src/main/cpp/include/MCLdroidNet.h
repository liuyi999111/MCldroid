//
// Created by 神经元002 on 2017/3/23.
//

#ifndef MCLDROID_NET_H
#define MCLDROID_NET_H


#include <vector>
#include <BaseLayer.h>
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


#endif //MCLDROID_NET_H

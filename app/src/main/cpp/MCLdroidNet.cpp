//
// Created by 神经元002 on 2017/3/23.
//

#include <BitmapProcess.h>
#include "MCLdroidNet.h"
#include <logUtil.h>
#include <ComputeTool.h>
#include <stdlib.h>

MultiDimensionData<float> inputBitmapMap;
Net net;
Net_Tree * net_tree = NULL;

#define DEBUG__
#define DEBUG__MEAN
#define DEBUG__RESULT
#define DEBUG__LOG_TIME

#undef DEBUG__
#undef DEBUG__MEAN
#undef DEBUG__RESULT
#undef DEBUG__LOG_TIME

void copyMDData(MultiDimensionData<float> *dst, MultiDimensionData<float> *src){
    size_t size = src->totalSize();
    float * temp = new float[size];
    copyData(temp, src->data_ptr, size);
    dst->setData(temp, src->shape);
}

void deriveLayer(Net_Tree *net,BaseLayer * layer, MultiDimensionData<float> * input, MultiDimensionData<float> ** outputArray ){
#ifdef DEBUG__
    LOGD("deriveLayer: %s",layer->getName().data());
    std::string title = "det2_";
    title.append(layer->getName()).append("_in");
    logMDData(input, title.data());
#endif

#ifdef DEBUG__LOG_TIME
    double st = now_ms();
#endif
    MultiDimensionData<float> * output = outputArray[net->indexOutPut];
    layer->compute(input, output);

#ifdef DEBUG__LOG_TIME
    LOGD("deriveLayer: %s compute time: %f", layer->getName().data(), now_ms() - st );
#endif

#ifdef DEBUG__
    std::string title2 = "det2_";
    title2.append(layer->getName()).append("_out");
    logMDData(output, title2.data());
#endif

    size_t num_nextLayer = layer->num_nextLayer();

    if (num_nextLayer == 0){//网络的一个终点
        net->outputLayers[ net->indexOutPut ] = layer;
        net->indexOutPut ++;

    } else if (num_nextLayer == 1){//没有分支向前

        if (output->data_ptr != NULL){
            input->releaseData();
            input->setData(output->data_ptr, output->shape);
            output->setData(NULL,1,1,1,1);
        }

        deriveLayer(net, layer->getNextLayer(0), input, outputArray);

    } else if ( num_nextLayer > 1){//有分支向前

        if (output->data_ptr != NULL){
            input->releaseData();
            input->setData(output->data_ptr, output->shape);
            output->setData(NULL,1,1,1,1);
        }

        for (int i = 0; i < layer->num_nextLayer(); ++i){
            MultiDimensionData<float> temp;
            if (i > 0){
                copyMDData(&temp, input);
            } else {
                temp.setData(input->data_ptr, input->shape);
            }

            deriveLayer(net, layer->getNextLayer(i), &temp, outputArray);
        }

    } else {
        LOGE("Error: num_nextLayer=%lu",num_nextLayer);
    }

}

void Net_Tree::forward(MultiDimensionData<float> *input, MultiDimensionData<float> **output,
                       size_t num_netOutput) {
    if (netHeader == NULL || num_netOutput <= 0){
        return;
    }
    BaseLayer * currentLayer = netHeader;
    deriveLayer(this, currentLayer, input, output);
}

void Net::forward(MultiDimensionData<float> *input, MultiDimensionData<float> *output){
    if (layers == NULL || layerSize <= 0){
        return;
    }
    if (input == NULL || output == NULL){
        return;
    }

    double st;


    BaseLayer* layerPtr = getLayerInOrder();
    while (layerPtr != NULL){
#ifdef DEBUG__

        LOGD("---------------------compute :%s------------------------",layerPtr->getName().data());

        st = now_ms();
#endif

#ifdef DEBUG__LOG_TIME
        st = now_ms();
#endif
        if (output->data_ptr != NULL){
            input->releaseData();
            input->setData(output->data_ptr, output->shape);
            output->setData(NULL,1,1,1,1);//不再保存上一次输出的信息
        }

#ifdef DEBUG__
        std::string title  = "det1_";
        title.append(layerPtr->getName());
        title.append("_input");
        logMDData(input, title.data());
#endif

        layerPtr->compute(input, output);//计算

#ifdef DEBUG__
        title = "det1_";
        title.append(layerPtr->getName());
        title.append("_output");
        logMDData(output, title.data());

        LOGD("outputSize:%lu, n*c*h*w : %lu*%lu*%lu*%lu",
             output->totalSize(),output->get_n(),output->get_c(),output->get_h(),output->get_w());
        LOGD("convTime:%f", now_ms() - st);
        LOGD("---------------------compute end------------------------");
#endif

#ifdef DEBUG__LOG_TIME
        LOGD("layer: %s, time:%f", layerPtr->getName().data(), now_ms() - st);
#endif

        layerPtr = getLayerInOrder();//取下一层
    }

#ifdef DEBUG__RESULT
    std::string title = "det1_net_output";
    logMDData(output, title.data());
#endif
}

void meanInput(MultiDimensionData<float> *input, MultiDimensionData<float> *mean){
    //input && mean :1*3*32*32

    size_t cSize = input->get_h()*input->get_w();
    size_t nSize = input->get_c()*cSize;
    size_t h = input->get_h();
    size_t w = input->get_w();

    float *dataPtr = input->data_ptr;
    float *meanPtr = mean->data_ptr;
    for (int i = 0; i < input->get_n(); i++){
        for (int chanel = 0; chanel<input->get_c(); chanel++){

            for (int row = 0; row < h; row++){
                for (int column = 0; column < w; column++){
                    //! mean 的存储顺序是 bgr ,{0,1,2} -> {2,1,0}
                    int c_mean = abs(chanel - 2);
                    float meanData = meanPtr[ c_mean*cSize + column*w + row];
                    dataPtr[i*nSize + chanel*cSize + row*w + column] -= meanData;
                }
            }

        }
    }
};

extern "C" {

JNIEXPORT void JNICALL
Java_com_compilesense_liuyi_mcldroid_mcldroid_MCLdroidNet_setInputMean(JNIEnv *env,
                                                                       jobject instance,
                                                                       jfloatArray mean_,
                                                                       jintArray meanShape_) {


    int weightArrayLength = 0;
    float * mean = jFloatArray2prtFast(env, mean_, &weightArrayLength);

    int meanShapeLength = 0;
    int* meanShape = jIntArray2prt(env, meanShape_, &meanShapeLength);
    std::vector<size_t> meanShapeV(4);

    //TODO meanShape 不是4维的时候出错,这样整个 MultiDimensionData 的维度设计是不好的!

    meanShapeV[0] = 1;
    meanShapeV[1] = (size_t) meanShape[0];
    meanShapeV[2] = (size_t) meanShape[1];
    meanShapeV[3] = (size_t) meanShape[2];

    MultiDimensionData<float> meanMDD(mean, meanShapeV);

#ifdef DEBUG__MEAN
    logMDData(&meanMDD, "MCLdroid-mean");
#endif

    meanInput(&inputBitmapMap, &meanMDD);

#ifdef DEBUG__MEAN
    logMDData(&inputBitmapMap, "MCLdroid-mean-finish");
#endif

    jFloatArrayRelease(env, mean_, mean);
    meanMDD.releaseData();
}


JNIEXPORT void JNICALL
Java_com_compilesense_liuyi_mcldroid_mcldroid_MCLdroidNet_compute(JNIEnv *env, jobject instance) {
    if (inputBitmapMap.data_ptr == NULL){
        LOGE("inputBitmapMap.data_ptr == NULL");
        return;
    }
    MultiDimensionData<float> outPut;
    net.forward(&inputBitmapMap, &outPut);
}



JNIEXPORT void JNICALL
Java_com_compilesense_liuyi_mcldroid_mcldroid_MCLdroidNet_setupNet(JNIEnv *env, jobject instance,
                                                                   jlongArray nativeObjectArray_) {
    int layerArrayLength = 0;
    long long * layers = jLongArray2prt(env, nativeObjectArray_, &layerArrayLength);
    net.setUpNet(layers, (size_t) layerArrayLength);
}


JNIEXPORT void JNICALL
Java_com_compilesense_liuyi_mcldroid_mcldroid_MCLdroidNet_bitmapProcess(JNIEnv *env, jclass type, jobject bitmap) {
    AndroidBitmapInfo  info;
    void*              pixels;//数据
    int                ret;

    if ((ret = AndroidBitmap_getInfo(env, bitmap, &info)) < 0) {
        LOGE("AndroidBitmap_getInfo() failed ! error=%d", ret);
        return;
    }

    if ((ret = AndroidBitmap_lockPixels(env, bitmap, &pixels)) < 0) {
        LOGE("AndroidBitmap_lockPixels() failed ! error=%d", ret);
    }

    pixels2MDData(&info ,pixels, &inputBitmapMap);
//    pixels2MDDataWithPreproccess(&info, pixels, &inputBitmapMap);
    AndroidBitmap_unlockPixels(env, bitmap);
}


JNIEXPORT void JNICALL
Java_com_compilesense_liuyi_mcldroid_mcldroid_MCLdroidNet_setupNetTree(JNIEnv *env, jobject instance,
                                                                       jlong netHeader_,
                                                                       jint numNetOutput,
                                                                       jint numLayer) {
    if (net_tree == NULL){
        net_tree = new Net_Tree;
    }
    net_tree->setUp((BaseLayer *) netHeader_, (size_t) numLayer, (size_t) numNetOutput);
}

JNIEXPORT void JNICALL
Java_com_compilesense_liuyi_mcldroid_mcldroid_MCLdroidNet_computeTree(JNIEnv *env, jobject instance) {
    if (inputBitmapMap.data_ptr == NULL){
        LOGE("inputBitmapMap.data_ptr == NULL");
        return;
    }
    size_t numOutput = net_tree->getNumOutput();
    MultiDimensionData<float> * outPut[numOutput];
    for (int i = 0; i < numOutput; ++i) {
        outPut[i] = new MultiDimensionData<float>();
    }
    net_tree->forward(&inputBitmapMap, outPut, numOutput);
    for (int i = 0; i < numOutput; ++i) {
        delete  outPut[i];
    }
}

JNIEXPORT void JNICALL
Java_com_compilesense_liuyi_mcldroid_mcldroid_BaseLayer_addNextLayerNative(JNIEnv *env,
                                                                           jobject instance,
                                                                           jlong nativeObject,
                                                                           jlong nextLayer) {

    BaseLayer * baseLayer = (BaseLayer *) nativeObject;
    baseLayer->addNextLayer((BaseLayer *) nextLayer);
}


}
//
// Created by 神经元002 on 2017/3/10.
//

#include "ActivationLayer.h"
#include <ComputeTool.h>
#include <ModelInput.h>

void ActivationLayer::compute(MultiDimensionData<float> *input) {
    switch (type){
        case ReLU:
            relu(input->data_ptr, input->totalSize());
            break;
        case PReLU:
            prelu(input->data_ptr, input->get_n(),input->get_c(),input->get_h(),input->get_w(),
                  params.data_ptr);
            break;
        case TanH:
            tanh(input->data_ptr, input->totalSize());
            break;
        case Abs:
            abs(input->data_ptr, input->totalSize());
            break;
    }
}

void ActivationLayer::loadParamsNative(std::string filePath) {
    switch (type){
        case ReLU:
            break;
        case PReLU:
            loadPReLuParams(filePath, &params);
            paramHadLoad = true;
            break;
        case TanH:
            break;
        case Abs:
            break;
    }

}

#ifdef __cplusplus
extern "C" {
#endif

JNIEXPORT jlong JNICALL
Java_com_compilesense_liuyi_mcldroid_mcldroid_ActivationLayer_createActivationLayer(JNIEnv *env,
                                                                                    jclass type_,
                                                                                    jstring name_,
                                                                                    jint type) {
    const char *name = env->GetStringUTFChars(name_, 0);

    ActivationLayer::Type acType;
    switch (type) {
        case 1:
            acType = ActivationLayer::ReLU;
            break;
        case 2:
            acType = ActivationLayer::PReLU;
            break;
        case 3:
            acType = ActivationLayer::TanH;
            break;
        case 4:
            acType = ActivationLayer::Abs;
            break;
        default:
            acType = ActivationLayer::ReLU;
    }

    ActivationLayer *ptr = new ActivationLayer(name, acType);
    env->ReleaseStringUTFChars(name_, name);
    return (jlong) ptr;
}

JNIEXPORT void JNICALL
Java_com_compilesense_liuyi_mcldroid_mcldroid_ActivationLayer_loadParamsNative(JNIEnv *env,
                                                                               jclass type_,
                                                                               jlong objectPrt_,
                                                                               jstring filePath) {
    const char * path = env->GetStringUTFChars(filePath,NULL);
    ActivationLayer *objectPrt = (ActivationLayer *) objectPrt_;
    objectPrt->loadParamsNative(path);
    env->ReleaseStringUTFChars(filePath, path);
}


#ifdef __cplusplus
}
#endif
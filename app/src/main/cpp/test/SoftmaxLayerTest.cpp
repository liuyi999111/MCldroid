//
// Created by 神经元002 on 2017/3/21.
//

#include <jni.h>
#include <SoftmaxLayer.h>
#include <stdlib.h>

extern "C" {

JNIEXPORT void JNICALL
Java_com_compilesense_liuyi_mcldroid_NativeTest_sofymaxLayerTest(JNIEnv *env, jclass type) {
    //二位数据
    SoftmaxLayer softmaxLayer("testSoftmaxLayer");
    float * dataPtr1 = new float[100];
    for (int i = 0; i<100; i++){
        dataPtr1[i] = 2.0f *rand()/RAND_MAX;
    }
    std::vector<size_t> shape(4);
    shape[0] = 1;
    shape[1] = 100;
    shape[2] = 1;
    shape[3] = 1;
    MultiDimensionData<float> input(dataPtr1, shape);
    MultiDimensionData<float> outputPtr;
    softmaxLayer.compute(&input, &outputPtr);
    for (int i = 0; i<100; i++){
        LOGD("output data[%i]:%f",i,outputPtr.data_ptr[i]);
    }
}

JNIEXPORT jlong JNICALL
Java_com_compilesense_liuyi_mcldroid_mcldroid_SoftmaxLayer_createSoftmaxLayer(JNIEnv *env,
                                                                              jclass type,
                                                                              jstring name_) {
    const char *name = env->GetStringUTFChars(name_, 0);

    SoftmaxLayer *ptr = new SoftmaxLayer(name);

    env->ReleaseStringUTFChars(name_, name);
    return (jlong) ptr;
}

}
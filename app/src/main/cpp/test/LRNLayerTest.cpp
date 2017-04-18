//
// Created by 神经元002 on 2017/4/17.
//

#include <jni.h>
#include <LocalResponseNormalizationLayer.h>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <ModelInput.h>
#include <logUtil.h>

using std::ifstream;
using std::ios;


void getData(MultiDimensionData<float> *data){
    loadTestTempData( "/storage/emulated/0/MCLdroidTestLog/temp_data", data);
}

extern "C"{


JNIEXPORT void JNICALL
Java_com_compilesense_liuyi_mcldroid_NativeTest_lrnLayerTest(JNIEnv *env, jclass type) {

    LocalResponseNormalizationLayer::Param params(5, 0.0001f, 0.75f);
    LocalResponseNormalizationLayer layer("testLRN", params);
    MultiDimensionData<float> input;
    getData(&input);
    std::vector<size_t > shape(4);
    shape[0]=1;
    shape[1]=96;
    shape[2]=27;
    shape[3]=27;
    input.reShape(shape);

    MultiDimensionData<float> result;
    layer.compute(&input, &result);

    logMDData(&result,"testLRN");
}

}
//
// Created by 神经元002 on 2017/4/6.
//
#include "ModelInput.h"
#include <cstdlib>
#include <fstream>
#include <msgpack.hpp>
#include <sstream>

using std::ifstream;
using std::ios;

#define LOG_CONV_WEIGHT

#undef LOG_CONV_WEIGHT

#ifdef LOG_CONV_WEIGHT
std::stringstream ss;
#endif

//对多维数组的序列化,直接转换时很难的,采用 visitor 的方式灵活好用。
struct convolution_kernel_weight_msgpack_visitor : msgpack::null_visitor{
    MultiDimensionData<float> * MDDataptr;
    std::vector<size_t > shape;
    float * dataptr;
    size_t index_data;
    size_t dataSize;
    bool shaping;

    convolution_kernel_weight_msgpack_visitor( MultiDimensionData<float> * dataPtr_)
            : MDDataptr(dataPtr_),shape() {
        dataptr = NULL;
        index_data = 0;
        dataSize = 1;
        shaping = true;
    }

    bool visit_float(double v){

#ifdef LOG_CONV_WEIGHT
        ss<<v;
#endif
        if (index_data >= dataSize){
            return true;
        }

        dataptr[index_data] = (float) v;
        index_data++;

        return true;
    }

    bool start_array(uint32_t num_elements){
#ifdef LOG_CONV_WEIGHT
        ss<<"[";
#endif
        if (shaping){
            shape.push_back(num_elements);
            dataSize *= num_elements;
            if (shape.size() == 4){
                dataptr = new float[dataSize];
                shaping = false;
            }
        }
        return true;
    }

    bool end_array_item(){
//        m_s += ",";
#ifdef LOG_CONV_WEIGHT
        ss<<",";
#endif
        return true;
    }

    bool end_array(){
#ifdef LOG_CONV_WEIGHT
        ss<<"]";
#endif

        if (index_data == dataSize ){
            MDDataptr->setData(dataptr, shape);
        }

        return true;
    }


};

struct one_dimension_array_msgpack_visitor : msgpack::null_visitor{
    MultiDimensionData<float> * MDDataptr;
    std::vector<size_t > * shape;
    float * dataptr;
    size_t index_data;
    size_t dataSize;

    one_dimension_array_msgpack_visitor( MultiDimensionData<float> * dataPtr_)
            : MDDataptr(dataPtr_) {

        shape = new std::vector<size_t>(4);
        dataptr = NULL;
        index_data = 0;
        dataSize = 1;
    }

    ~one_dimension_array_msgpack_visitor(){
        delete shape;
    }

    bool visit_float(double v){
#ifdef LOG_CONV_WEIGHT
        ss<<v;
#endif
        if (index_data >= dataSize){
            return true;
        }

        dataptr[index_data] = (float) v;
        index_data++;

        return true;
    }

    bool start_array(uint32_t num_elements){
#ifdef LOG_CONV_WEIGHT
        ss<<"[";
#endif
        dataSize = num_elements;
        dataptr = new float[dataSize];
        (*shape)[0] = 1;
        (*shape)[1] = 1;
        (*shape)[2] = 1;
        (*shape)[3] = num_elements;
        return true;
    }

    bool end_array(){
#ifdef LOG_CONV_WEIGHT
        ss<<"]";
#endif

        if (index_data == dataSize ){
            MDDataptr->setData(dataptr, *shape);
        }

        return true;
    }


};

//全连接层 kernel 数据 weight[], bias[];
void loadFullyConnectKernel(std::string filePath, MultiDimensionData<float> *MDDweight, MultiDimensionData<float> *MDDbias){
    ifstream binaryFstream(filePath, ios::in|ios::binary|ios::ate);
    size_t size = (size_t) binaryFstream.tellg();
    char* buff = new char[size];
    binaryFstream.seekg(0, ios::beg);
    binaryFstream.read(buff, size);
    binaryFstream.close();

    std::size_t off = 0;
    msgpack::unpacked resultWeight;
    unpack(resultWeight, buff, size, off);
    msgpack::object objW(resultWeight.get());
    std::vector<float> *vWeight = new std::vector<float>();
    objW.convert(*vWeight);
    float * weightPtr = vWeight->data();
    MDDweight->setData(weightPtr, 1, 1, 1, vWeight->size());
    LOGE("weightSize: %lu", MDDweight->totalSize());
    msgpack::unpacked resultBias;
    unpack(resultBias, buff, size, off);
    msgpack::object objB(resultBias.get());
    std::vector<float> *vBias = new std::vector<float>();
    objB.convert(*vBias);
    float * biasPtr = vBias->data();
    MDDbias->setData(biasPtr, 1, 1, 1, vBias->size());
    delete[] buff;
}

void loadFullyConnectKernel_v(std::string filePath, MultiDimensionData<float> *MDDweight, MultiDimensionData<float> *MDDbias){
    ifstream binaryFstream(filePath, ios::in|ios::binary|ios::ate);
    size_t size = (size_t) binaryFstream.tellg();
    char* buff = new char[size];
    binaryFstream.seekg(0, ios::beg);
    binaryFstream.read(buff, size);
    binaryFstream.close();

    one_dimension_array_msgpack_visitor visitor( MDDweight );
    std::size_t off = 0;
    bool ret1 = msgpack::v2::parse(buff, size, off, visitor);
    one_dimension_array_msgpack_visitor visitor2( MDDbias );
    bool ret2 = msgpack::v2::parse(buff, size, off, visitor2);

    delete[] buff;
}

//卷积层 kernel 数据 weight[][][][], bias[];
void loadConvolutionKernel(std::string filePath, MultiDimensionData<float> *weightMDD, MultiDimensionData<float> *biasMDD){
    ifstream binaryFstream(filePath, ios::in|ios::binary|ios::ate);
    size_t size = (size_t) binaryFstream.tellg();
    char* buff = new char[size];
    binaryFstream.seekg(0, ios::beg);
    binaryFstream.read(buff, size);
    binaryFstream.close();

    convolution_kernel_weight_msgpack_visitor visitor( weightMDD );
    std::size_t off = 0;
    bool ret1 = msgpack::v2::parse(buff, size, off, visitor);
    one_dimension_array_msgpack_visitor visitor2(biasMDD);
    bool ret2 = msgpack::v2::parse(buff, size, off, visitor2);

    delete[] buff;
}

void loadPReLuParams(std::string filePath, MultiDimensionData<float> *params){
    ifstream binaryFstream(filePath, ios::in|ios::binary|ios::ate);
    size_t size = (size_t) binaryFstream.tellg();
    char* buff = new char[size];
    binaryFstream.seekg(0, ios::beg);
    binaryFstream.read(buff, size);
    binaryFstream.close();
    std::size_t off = 0;
    one_dimension_array_msgpack_visitor visitor(params);
    bool ret1 = msgpack::v2::parse(buff, size, off, visitor);
    delete[] buff;
}

void loadTestTempData(std::string filePath, MultiDimensionData<float> *data){
    ifstream binaryFstream(filePath, ios::in|ios::binary|ios::ate);
    size_t size = (size_t) binaryFstream.tellg();
    char* buff = new char[size];
    binaryFstream.seekg(0, ios::beg);
    binaryFstream.read(buff, size);
    binaryFstream.close();

    one_dimension_array_msgpack_visitor visitor(data );
    std::size_t off = 0;
    bool ret1 = msgpack::v2::parse(buff, size, off, visitor);

    delete[] buff;
}



extern "C" {

JNIEXPORT void JNICALL
Java_com_compilesense_liuyi_mcldroid_NativeTest_testModelInput(JNIEnv *env, jclass type,
                                                             jstring filePath_) {
    const char *filePath = env->GetStringUTFChars(filePath_, 0);
    MultiDimensionData<float> weight;
    MultiDimensionData<float> bias;
    loadFullyConnectKernel_v("/sdcard/Data_CaffeNet/model_param_fc6.msg", &weight, &bias);
    env->ReleaseStringUTFChars(filePath_, filePath);
}

}//extern "C"
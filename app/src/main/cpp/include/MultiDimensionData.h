//
// Created by 神经元002 on 2017/3/21.
//

#ifndef MCLDROID_MULTIDIMENSIONDATA_H
#define MCLDROID_MULTIDIMENSIONDATA_H

#include <vector>
#include <cwchar>
#include <utils.h>

/**
 * 在网络中的数据大多是多维的,如果用一维数组表示,那么 Shape 这个数据就是形影不离的。
 * 所以将其封装为一个类型。
 *
 * 注意是否需要在析构时回收数据内存。
 * 如果数据在栈上,则不需要回收,在赋值时 shouldRelease = false;
 *
 * TODO 可以不止4位有更多的维度, 但是在这样数据将会是连续的占用大片区域, 不知道这么做是否合适。
 */
template<typename dataType>
class MultiDimensionData {
public:
    dataType * data_ptr;
    std::vector<size_t> shape;//{n,c,h,w}
    size_t num_dimensions;

    MultiDimensionData():data_ptr(NULL),shape(4),num_dimensions(1),shouldRelease(true){
    }
    MultiDimensionData(dataType * data_ptr, std::vector<size_t> shape , bool shouldRelease = true)
            : data_ptr(data_ptr), shape(shape),shouldRelease(shouldRelease){
        num_dimensions = shape.size();
    }
    MultiDimensionData(dataType * data_ptr, size_t n, size_t c ,size_t h, size_t w, bool shouldRelease = true)
            : data_ptr(data_ptr), shape(4){
        shape[0] = n;
        shape[1] = c;
        shape[2] = h;
        shape[3] = w;
        num_dimensions = shape.size();
    }
    ~MultiDimensionData(){
        LOGD("~MultiDimensionData shape.w:%lu",get_w());
        if (data_ptr!=NULL && shouldRelease){
            delete[] data_ptr;
        } else{
            LOGD("~MultiDimensionData data_ptr don't release");
        }
    }
    void setData(dataType * data_ptr, size_t n, size_t c ,size_t h, size_t w, bool shouldRelease = true){
        this->data_ptr = data_ptr;
        if (shape.capacity() >= 4){
            shape[0] = n;
            shape[1] = c;
            shape[2] = h;
            shape[3] = w;
        } else{
            shape.reserve(4);
            shape[0] = n;
            shape[1] = c;
            shape[2] = h;
            shape[3] = w;
        }
        num_dimensions = 4;
        this->shouldRelease = shouldRelease;
    }

    void setData(dataType * data_ptr, std::vector<size_t> shape_, bool shouldRelease = true){
        this->data_ptr = data_ptr;
        if (shape.capacity() >= shape_.size()){
            shape = shape_;
        } else{
            shape.reserve(shape_.size());
            shape = shape_;
        }
        num_dimensions = shape_.size();
        this->shouldRelease = shouldRelease;
    }

    void releaseData(){
        if (data_ptr != NULL && shouldRelease){
            delete[] data_ptr;
        }
        setData(NULL, 1,1,1,1);
        num_dimensions = 0;
    }

    size_t totalSize(){
        int num = 1;
        for (int i = 0; i < num_dimensions; ++i){
            num *= shape[i];
        }
        return (size_t)num;
    }

    size_t get_n(){
        return  (num_dimensions < 4) ? 1 : shape[0];
    }
    size_t get_c(){
        return  (num_dimensions < 3) ? 1 : shape[1];
    }
    size_t get_h(){
        return  (num_dimensions < 2) ? 1 : shape[2];
    }
    size_t get_w(){
        return  (num_dimensions < 1) ? 1 : shape[3];
    }

    void reShape(std::vector<size_t> shape_){
        if (shape.capacity() >= shape_.size()){
            shape = shape_;
        } else{
            shape.reserve(shape_.size());
            shape = shape_;
        }
        num_dimensions = shape_.size();
    }
private:
    bool shouldRelease;
};

#endif //MCLDROID_MULTIDIMENSIONDATA_H

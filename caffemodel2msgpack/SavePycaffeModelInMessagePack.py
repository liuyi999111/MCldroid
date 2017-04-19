# .caffemodel file path:
MODEL_FILE = '/Users/shenjingyuan002/work/app-repo-android/MCldroid/caffemodel2msgpack/model/det1.caffemodel'
# .prototxt file path:
MODEL_NET = '/Users/shenjingyuan002/work/app-repo-android/MCldroid/caffemodel2msgpack/model/det1.prototxt'
# Saving path:
SAVE_TO = '/Users/shenjingyuan002/work/app-repo-android/MCldroid/caffemodel2msgpack/msgpack/'

# Set True if you want to get parameters:
GET_PARAMS = True

import sys
import os
import numpy as np
import msgpack

# Make sure that Caffe is on the python path:
# Caffe installation path:
caffe_root = '/Users/shenjingyuan002/work/caffe/caffe-master/'
sys.path.insert(0, caffe_root + 'python')
sys.path.append(os.path.abspath('.'))
import caffe


def get_params(net):
    net_params = {'info': 'net.params data'}
    for key in net.params.iterkeys():
        print('Getting parameters: ' + key)

        if type(net.params[key]) is not caffe._caffe.BlobVec:
            net_params[key] = net.params[key].data
        elif len(net.params[key]) == 2: 
            net_params[key] = [net.params[key][0].data, net.params[key][1].data]
        elif len(net.params[key]) == 1:
            net_params[key] = net.params[key][0].data
        else:
            pass
            print "params length > 2"

    return net_params


# Open a model:
caffe.set_mode_cpu()
net = caffe.Net(MODEL_NET, MODEL_FILE, caffe.TEST)
# snet.forward()
# print layer info
for layer_name, param in net.params.items():
    if len(param) == 2:
        pass
        print layer_name + '\t' + str(param[0].data.shape), str(param[1].data.shape)
    elif len(param) == 1:
        pass
        print layer_name + '\t' + str(param[0].data.shape)
    else:
        pass
        print "params length > 2"
    

# Extract the model:
if GET_PARAMS:
    params = get_params(net)

save_list = []


# Write parameters:
if GET_PARAMS:
    save_list.append('***PARAMS***')
    for b in params.iterkeys():
        print('Saving parameters: ' + b)
        if type(params[b]) is np.ndarray:
            save_list.append(b)
            buf = msgpack.packb(params[b].tolist(), use_single_float = True)
            with open(SAVE_TO + 'model_param_'+ b + '.msg', 'wb') as f:
                f.write(buf)
        elif type(params[b]) is list:
            save_list.append(b)
            if len(params[b][0].shape) == 4:			# for the convolution layers
               buf1 = msgpack.packb(params[b][0].tolist(), use_single_float = True)
            elif len(params[b][0].shape) == 2:			# for the fully-connected layers
               buf1 = msgpack.packb(params[b][0].ravel().tolist(), use_single_float = True)
            buf2 = msgpack.packb(params[b][1].tolist(), use_single_float = True)
            with open(SAVE_TO + 'model_param_'+ b + '.msg', 'wb') as f:
                f.write(buf1)
                f.write(buf2)
        else:
            print('Parameters ' + b + ' not saved.')

print('Saved data:')
print(save_list)

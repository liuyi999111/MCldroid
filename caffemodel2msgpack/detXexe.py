# .caffemodel file path:
MODEL_FILE = '/Users/shenjingyuan002/work/app-repo-android/MCldroid/caffemodel2msgpack/model/det2.caffemodel'
# .prototxt file path:
MODEL_NET = '/Users/shenjingyuan002/work/app-repo-android/MCldroid/caffemodel2msgpack/model/det2.prototxt'
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

# Open a model:
caffe.set_mode_cpu()
net = caffe.Net(MODEL_NET, MODEL_FILE, caffe.TEST)

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_raw_scale('data', 255)
transformer.set_transpose('data', (2,0,1)) 

im = caffe.io.load_image("/Users/shenjingyuan002/work/app-repo-android/MCldroid/caffemodel2msgpack/test_24x24.jpg")
net.blobs['data'].data[...] = transformer.preprocess('data', im)
# net.blobs['data'].reshape(1, 3, 12, 12)

out = net.forward()

prob= net.blobs['prob1'].data[0].flatten() 
print "result:\n"
print prob

# print "input img data"
# print net.blobs['data'].data[0].shape
# print net.blobs['data'].data[0]

print "conv5-2 output"
print net.blobs['conv5-2'].data[0].shape
print net.blobs['conv5-2'].data[0]



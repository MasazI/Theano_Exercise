# coding: utf-8

import theano
import theano.tensor as T

import numpy as np

## max pooling
from theano.tensor.signal import downsample

# input データ
input = T.dtensor4('input')

# max poolingのウィンドウサイズ
maxpool_shape = (2, 2)

# max poolingのシンボル
pool_out = downsample.max_pool_2d(input, maxpool_shape, ignore_border=True)

# max poolingのファンクション化
f = theano.function([input], pool_out)

invals = np.random.RandomState(1).rand(3,2,5,5)

print 'invals[0,0,:,:] = \n' , invals[0,0,:,:]
print 'output[0,0,:,:] = \n' , f(invals)[0,0,:,:]

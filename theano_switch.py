# coding: utf-8
import theano.tensor as T
from theano.ifelse import ifelse
import theano, time
import numpy as np

a, b = T.scalars('a', 'b')
x, y = T.matrices('x', 'y')

z_switch = T.switch(T.lt(a,b), T.mean(x), T.mean(y))
z_lazy = ifelse(T.lt(a,b), T.mean(x), T.mean(y))

f_switch = theano.function([a,b,x,y], z_switch, mode=theano.Mode(linker='vm'))
f_lazyifelse = theano.function([a,b,x,y], z_lazy, mode=theano.Mode(linker='vm'))

# you can test change val1 and val2
val1 = 1.
val2 = 0.

big_mat1 = np.ones((10, 10))
big_mat2 = np.zeros((10, 10))


print 'init big_mat1'
print big_mat1

print 'init big_mat2'
print big_mat2

n_times = 10

tic = time.clock()
for i in xrange(n_times):
    val = f_switch(val1, val2, big_mat1, big_mat2)
    print 'iteration %i'%i
    print val
print 'time spent evaluating both values %f sec'%(time.clock()-tic)

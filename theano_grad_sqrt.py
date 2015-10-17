# coding: utf-8
import numpy
import theano
import theano.tensor as T

x = T.dscalar('x')

y = (T.sqrt(x) + 1) ** 3

dy = T.grad(cost=y, wrt=x)

f = theano.function(inputs=[x], outputs=dy)

print theano.pp(f.maker.fgraph.outputs[0])

print f(2)
print f(3)

# coding: utf-8
import numpy as np
import theano
import theano.tensor as T

x = T.dscalar('x')

y = T.sin(x)

## cost関数、微分する変数の順で渡す
gy = T.grad(cost=y, wrt=x)

## 値、関数の順で渡す
f = theano.function(inputs=[x], outputs=gy)
print theano.pp(f.maker.fgraph.outputs[0])

print f(0)
# numpy の pi が利用可能
print f(np.pi / 2)
print f(np.pi)

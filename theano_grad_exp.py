# coding: utf-8
import numpy as np
import theano
import theano.tensor as T

x = T.dscalar('x')

## 微分される元の数式
y = T.exp(x)

## 数式をxについて微分
gy = T.grad(cost=y, wrt=x)

## function化
f = theano.function(inputs=[x], outputs=gy)

## 関数の表示
print theano.pp(f.maker.fgraph.outputs[0])

print f(2)
print f(3)
print f(4)

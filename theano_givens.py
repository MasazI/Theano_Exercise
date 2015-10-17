#coding: utf-8
import theano
import theano.tensor as T

x = T.dscalar()
y = T.dscalar()
c = T.dscalar()

## givens によってx、yとcの関係を与える
ff = theano.function([c], x*2+y, givens=[(x, c*10.), (y, 5.)])

print ff(2)

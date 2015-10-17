#coding: utf-8
import numpy as np
import theano
import theano.tensor as T

x = T.dscalar("x")

## functionの中で参照可能な共有型のデータ
b = theano.shared(np.array([1,2,3,4,5]))

f = theano.function([x], b * x)

print f(2)

## functionの実行後、共有型のデータを変更
b.set_value([4,5,6])

## function中の共有型データも変更
print f(2)

## 共有型の要素型指定(gpuを使う場合はfloat)
data = np.array([[1,2,3], [4,5,6]], dtype=theano.config.floatX)

## borrowはデフォルトでFalse、TrueではPython空間で共有した変数への変更が共有の変数データに反映される(Python空間上で定義されたデータの実体を共有変数でも共有するかどうか)
X = theano.shared(data, name='X', borrow=True)

print type(X)
print X.get_value()

data[0][0] = 3

## 共有空間の変数Xにdataへの変更が反映される
print X.get_value()

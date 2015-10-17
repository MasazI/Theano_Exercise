#coding: utf-8
import numpy
import theano
import theano.tensor as T

c = theano.shared(0)

## c = c + 1 という値更新をupdatesにより行う
f = theano.function([], c, updates=[(c, c+1)])

print f()
print f()
print f()

## updatesを利用した勾配法の実装
# 入力
x = T.dvector("x")

# 求めたい共有型のデータ：関数を実行しても共有され続けるので繰り返し関数を実行して更新する際に利用
c = theano.shared(0.)

# 最小化したい目的関数
y = T.sum((x-c)**2)

# yをcについて微分
gc = T.grad(y, c)

# 実行する度に0を更新して、現時点でのyをかえす
d2 = theano.function([x], y, updates=[(c, c - 0.05*gc)])

d2([1,2,3,4,5])
print c.get_value()

d2([1,2,3,4,5])
print c.get_value()

d2([1,2,3,4,5])
print c.get_value()

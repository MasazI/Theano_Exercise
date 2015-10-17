#coding: utf-8
import theano
import theano.tensor as T

## まとめて宣言
x, y = T.dscalars("x", "y")

z = (x+2*y)**2

## zをxについて微分
gx = T.grad(z, x)

## zをyについて微分
gy = T.grad(z, y)

## まとめて

v1 = [x, y]
v2 = [x, y]

## 配列を足してもできる
v = v1+v2

# vの中の変数について順番に微分
grads = T.grad(z, v)

print grads

fgy = theano.function([x, y], grads[3])
fgx = theano.function([x, y], grads[2])


## 自動微分の結果を表示
print theano.pp(fgx.maker.fgraph.outputs[0])
print theano.pp(fgy.maker.fgraph.outputs[0])

print fgx(1, 2)
print fgy(2, 3)

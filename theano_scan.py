# coding: utf-8
import theano
import theano.tensor as T
from theano import config
import numpy as np

## scan.Loop for int
n = T.iscalar('n')
a = T.iscalar('a')
b = T.iscalar('b')
# fnには式をいれることが可能。lambdaでもdefでも良い
def prior(a, b):
    return a * 2 + b
    
# fn: loop/iterationで適用される式、sequences: シーケンスとしてIterationされる引数
# outputs_info: Loopの場合の初期値、non_sequences: シーケンスでない場合のfunctionへの引数
# n_steps: 繰り返しの回数
# この場合、sequencesがNoneなので、Loop処理であり、繰り返しの関数はfn、繰り返し回数がn_steps
#a = 0
# lambdaで書く場合
#result, updates = theano.scan(fn=lambda prior, nonseq: prior * 2, sequences=None, outputs_info=a, non_sequences=a, n_steps=n)
# defで書く場合
# outputs_infoはfnの第１引数=priorとしてわたる、つまり繰り返しごとに前回の関数の結果を引き継いで渡されるのがa、non_sequencesはfnの第２引数=nonseqとして渡る、つまり繰り返しの結果を引き継がない定数がb
result, updates = theano.scan(fn=prior, sequences=None, outputs_info=a, non_sequences=b, n_steps=n)
sf1 = theano.function(inputs=[a, b, n], outputs=result, updates=updates)
print sf1(5, 7, 3)

## scan.Loop for vector
v = T.ivector('v')
v2 = T.ivector('v2')
# scalarのときと同じ、vが繰り返し結果を引き継ぐ、v2は毎回繰り返しのたびに渡される定数
result, updates = theano.scan(fn=lambda prior, nonseq: prior * 2 + nonseq, sequences=None, outputs_info=v, non_sequences=v2, n_steps=n)
sf2 = theano.function(inputs=[v, v2, n], outputs=result, updates=updates)
#vec = [1,2,3]
print sf2([1,2,3], [10, 10, 10], 3)

## scan.Iteration
outputs = T.as_tensor_variable(np.asarray(0))
# sequencesはシーケンスとしてIterationされる引数であり、ここでは配列のシンボルを指定。配列の各要素に処理がIterationされる
# outputs_infoはLoop処理の初期値
c = T.scalar('c', dtype='int64')
[result, updates] = theano.scan(fn=lambda seq, prior: seq + 2, sequences = T.arange(c), outputs_info=outputs, non_sequences=None)
sf3 = theano.function(inputs=[c], outputs=result, updates=updates)
# 0,1,2,3,4の配列の各要素に2ずつ加算して結果をえる
print sf3(5)

## scan.Iteration using prior
# シーケンスの場合の第２引数priorを使うと、前回の結果を引き継いで計算できる
[result, updates] = theano.scan(fn=lambda seq, prior: seq + prior, sequences = T.arange(c), outputs_info=outputs, non_sequences=None)
# n_stepsを指定して強制的に処理を終了させることが可能、その場合戻り値の要素はn_stepsで実行した分になる
# result, updates = theano.scan(fn=lambda seq, prior: seq + prior, sequences = T.arange(c), outputs_info=outputs, non_sequences=None, n_steps=3)

sf4 = theano.function(inputs=[c], outputs=result, updates=updates)
# 0,1,2,3,4の配列の各要素に、前回までの要素の和を加算して結果をえる
print sf4(5)


## scan.Iteration for vector using prior and return
outputs = T.as_tensor_variable(np.asarray(0))
x = T.ivector('x')
def recurrence(x_t, h_tm1):
    print x_t.eval
    print h_tm1.eval
    h_t = x_t + h_tm1
    s_t = x_t * 10
    return [h_t, s_t]

[h, s] = theano.scan(fn=recurrence, sequences=x, outputs_info=[outputs, None])
sf5 = theano.function(inputs=[x], outputs=h, updates=s)
print sf5([1,2,3,4,5])


outputs = T.as_tensor_variable(np.asarray(0))
def recurrence_lstm(m_, x_, h_, c_):
    print m_.eval
    print x_.eval
    print h_.eval
    print c_.eval
    h = m_ + h_
    c = x_ + c_
    return h, c

m = T.lvector('m')
x = T.lvector('x')
a = T.as_tensor_variable(np.asarray(0))
b = T.as_tensor_variable(np.asarray(0))

rval, updates = theano.scan(fn=recurrence_lstm, sequences=[m, x], outputs_info=[a, b], n_steps=5)
sf6 = theano.function(inputs=[m, x], outputs=rval, updates=updates)
print sf6([0,1,2,3,4], [10,11,12,13,14])


## scan be used as repeat-until
def power_of_2(previous_power, max_value):
    #return previous_power*2, theano.scan_module.until(previous_power*2 > max_value)
    return previous_power*2
max_value = T.scalar()
values,_= theano.scan(fn=power_of_2,
                        outputs_info = T.constant(1.),
                        non_sequences = max_value,
                        n_steps = 10)

f = theano.function([max_value], values)

#print _

print f(45)

## scan taps in outputs_info loop
def addf(a1, a2):
    print a1.eval
    print a2.eval
    
    return a1+a2

i = T.iscalar('i')
x0 = T.ivector('x0')
step = T.iscalar('step')

# 常に入力はx0である。
# taps -n は入力に x0[t-n]を利用する
results, updates = theano.scan(fn=addf, outputs_info=[{'initial':x0, 'taps':[-3]}], non_sequences=step, n_steps=i)
# この記述でもLoopの場合はx0[t-1]が使われる、Iterationの場合はx0[t]が使われる、つまりtapsは[0]がデフォルトなので注意
results2, _ = theano.scan(fn=addf, outputs_info=x0, non_sequences=step, n_steps=i)

f = theano.function([x0, i, step], results)
f2 = theano.function([x0, i, step], results2)

print f([1],10,2)
print f2([1],10,2)

## scan taps in sequence iteration
outputs = T.as_tensor_variable(np.asarray(0))
def recurrence_lstm(m_, h_, c_):
    print m_.eval
    print h_.eval
    print c_.eval
    h = m_ + h_
    c = m_ + c_
    return h, c

m = T.lvector('m')
a = T.as_tensor_variable(np.asarray(0))
b = T.as_tensor_variable(np.asarray(1))

rval, updates = theano.scan(fn=recurrence_lstm, sequences=dict(input=m, taps=[0]), outputs_info=[a, b], n_steps=5)
sf6 = theano.function(inputs=[m], outputs=rval, updates=updates)
print sf6([0,1,2,3,4])



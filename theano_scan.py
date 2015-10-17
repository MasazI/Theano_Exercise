# coding: utf-8
import theano
import theano.tensor as T

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
result, updates = theano.scan(fn=lambda seq, prior: seq + 2, sequences = T.arange(c), outputs_info=outputs, non_sequences=None)
sf3 = theano.function(inputs=[c], outputs=result, updates=updates)
# 0,1,2,3,4の配列の各要素に2ずつ加算して結果をえる
print sf3(5)

## scan.Iteration using prior
# シーケンスの場合の第２引数priorを使うと、前回の結果を引き継いで計算できる
result, updates = theano.scan(fn=lambda seq, prior: seq + prior, sequences = T.arange(c), outputs_info=outputs, non_sequences=None)
# n_stepsを指定して強制的に処理を終了させることが可能、その場合戻り値の要素はn_stepsで実行した分になる
# result, updates = theano.scan(fn=lambda seq, prior: seq + prior, sequences = T.arange(c), outputs_info=outputs, non_sequences=None, n_steps=3)

sf4 = theano.function(inputs=[c], outputs=result, updates=updates)
# 0,1,2,3,4の配列の各要素に、前回までの要素の和を加算して結果をえる
print sf4(5)

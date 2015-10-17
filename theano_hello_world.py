#coding: utf-8
import theano
import theano.tensor as T

## Aのmatrix宣言(i: int, l: float, d: doubl)をmatrixの前におくと、要素の型を指定できる:
A = T.imatrix()

## 行列Aの各要素を倍にする演算を定義
B = A*2

## 関数の宣言(C++のコードが生成され、コンパイルが実行される)第一引数が変数、第二引数が演算
f = theano.function([A], B)

## AとBに具体的な値を与えて計算する
print f([[1,24,5], [3,33,4]])

# coding: utf-8
import numpy as np

# 例1 lambdaは式を変数に格納できる仕組み
func = lambda x, y, z: x + y + z

print func(2,3,4)


# 例2 引数にキーワードやデフォルトを設定できるのはdefと同じ
func2 = lambda x, y=10, z=20: x + y + z

print func2(5)


# 例3 配列に格納して、loopで関数処理を回す
list = np.array([lambda x: x + 1, lambda x: x + 2, lambda x: x+3])

for f in list:
    print f(10)

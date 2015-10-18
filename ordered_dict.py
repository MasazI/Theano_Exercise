# coding: utf-8
from collections import OrderedDict

# 普通のdict
d = {}
d["a"] = 0
d["b"] = 1
d["c"] = 2

# 順序が保持されない
for key, val in d.items():
    print key, val

print '----------'

# 順序を保持するdict ordered dict
od = OrderedDict()

od["a"] = 0
od["b"] = 1
od["c"] = 2

# 挿入の順序が保持される
for key, val in od.items():
    print key, val

print '----------'

# 注意としては、順序が保証されない挿入の仕方をすれば、
# 当然、Ordered dictの順序も保証されない

# コンストラクタの**可変長引数で渡しても、挿入の順序は保証されない
odn = OrderedDict(one=1, two=2, three=3)

for key, val in odn.items():
    print key, val

print '----------'

# tips tupleで追加すると順序が保証される
odt = OrderedDict((('one', 1), ('two', 2), ('three', 3)))

for key, val in odt.items():
    print key, val


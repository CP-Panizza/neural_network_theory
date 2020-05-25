'''通过bp算法拟合y = 2x这个函数'''


import numpy as np

x = np.array([1,2,3]).T
y = np.array([2,4,6]).T
w = 10

for i in range(1000):
	p_y = x * w
	dy = (p_y - y)/3
	dw = x.T.dot(dy)
	w += -0.01 * dw

print(w)




	
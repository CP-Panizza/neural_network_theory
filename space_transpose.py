'''
此文件研究的是二维平面内的二分类问题：
二维
'''


import numpy as np
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import Axes3D  # 空间三维画图

def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T 

    x = x - np.max(x) # 溢出对策
    return np.exp(x) / np.sum(np.exp(x))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))    


def Norm(x, y):
    """求一个二维向量模长"""
    return math.pow(math.pow(x, 2) + math.pow(y, 2), 0.5)

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        if Norm(x, y) > 5:
            self.kind = [1, 0]
        else:
            self.kind = [0, 1]
            
    def __str__(self):
        return "Point({}, {}, {})".format(self.x, self.y, self.kind)

def CreateData(n):
    data = []
    x = np.random.randint(20,size=n)
    y = np.random.randint(20,size=n)
    for i in range(n):
        if i % 2 == 0:
            t = np.random.randint(2, size=1)[0]
            if t % 2 == 0:
                data.append(Point(int(-x[i]), int(-y[i])))
            else:
                data.append(Point(int(-x[i]), int(y[i])))
        else:
            t = np.random.randint(2, size=1)[0]
            if t % 2 == 0:
                data.append(Point(int(x[i]), int(-y[i])))
            else:
                data.append(Point(int(x[i]), int(y[i])))
    return data





params = {'W1': np.array([[ 0.72907018, -0.11628131, -0.59031236],
                        [ 0.31832248, -0.76271294,  0.41404479]]), 
                  'b1': np.array([-3.7256935 , -3.3605026 , -3.56923247]), 
                  'W2': np.array([[ 3.03088151, -3.01463272],
                                [ 2.90246557, -2.91045341],
                               [ 2.9320408 , -2.96879877]]), 
                  'b2': np.array([-1.03164879,  1.03164879])}



def predict(x):
    return softmax(sigmoid((np.array(x).dot(params['W1']) + params['b1'])).dot(params['W2']) + params['b2'])

test = CreateData(500)
t = []
for i in test:
    t.append([i.x, i.y])


# #### 第一层隐含层把二维数据升维到三维


three = np.array(t).dot(params['W1']) + params['b1']
sigmoid_three = sigmoid(np.array(t).dot(params['W1']) + params['b1'])
tow = sigmoid_three.dot(params['W2']) + params['b2']
softmax_tow = softmax(sigmoid_three.dot(params['W2']) + params['b2'])

fig = plt.figure()
ax = Axes3D(fig)
index = 0
for i in three:
    if test[index].kind[0] == 1:
        ax.scatter(i[0], i[1], i[2], c='g')
    else:
        ax.scatter(i[0], i[1], i[2], c='r')
    index += 1

# 绘制图例
ax.legend(loc='best')
ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'red'})
ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'red'})
ax.set_xlabel('X', fontdict={'size': 15, 'color': 'red'})
plt.show()


fig = plt.figure()
ax = Axes3D(fig)
index = 0
for i in sigmoid_three:
    if test[index].kind[0] == 1:
        ax.scatter(i[0], i[1], i[2], c='g')
    else:
        ax.scatter(i[0], i[1], i[2], c='r')
    index += 1

# 绘制图例
ax.legend(loc='best')
ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'red'})
ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'red'})
ax.set_xlabel('X', fontdict={'size': 15, 'color': 'red'})
plt.show()


index = 0
for i in tow:
    if test[index].kind[0] == 1:
        plt.plot(i[0], i[1], 'o-g')
    else:
        plt.plot(i[0], i[1], 'o-r')
    index += 1

plt.show()


index = 0
for i in softmax_tow:
    if test[index].kind[0] == 1:
        plt.plot(i[0], i[1], 'o-g')
    else:
        plt.plot(i[0], i[1], 'o-r')
    index += 1

plt.show()








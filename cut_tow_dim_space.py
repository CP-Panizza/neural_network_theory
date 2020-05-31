#分类问题核心思想：物以类聚，人以群分
#数学中用事物的属性组成的向量来表示一个物体，
#相同的事物的向量在向量空间中总是聚集在一起的
#我们可以通过分割向量空间来区分不同的事物


#权重矩阵的意义：
#权重矩阵的每一个列向量和训练数据集中的每一个数据向量是同一个维度下的向量




#训练的过程就是寻找这些分类向量的方向的过程！！！

import numpy as np
import matplotlib.pyplot as plt
import math
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        if x >= 0 and y >= 0:
            self.kind = [1, 0]
        else:
            self.kind = [0, 1]
def random_mat(data):
    cut = np.random.random_integers(len(data))
    l = []
    start = 0
    end = 0
    if cut - 50 > 0:
        start = cut - 50
        end = cut
        l = data[start:end]
    elif cut + 50 < len(data):
        end = cut + 50
        start = cut
        l = data[start:end]

    mat = []
    for i in range(len(l)):
        mat.append([l[i].x,l[i].y])

    return np.array(mat), start, end

def get_kind(data, start, end):
    t = data[start:end]
    kind = []
    for i in range(50):
        kind.append(t[i].kind)

    return np.array(kind)

def Norm(v):
    return math.pow((math.pow(v[0], 2) + math.pow(v[1], 2)), 0.5)

def GetMaxIndex(v):
    max = v[0]
    index = 0
    for i in range(len(v)):
        if v[i] > max:
            max = v[i]
            index = i
    return index

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

if __name__ == '__main__':
    data = CreateData(1000)
    w = np.random.rand(2, 2)
    for i in range(2000):
        mat, start, end = random_mat(data)
        kind = get_kind(data, start, end)
        out = mat.dot(w)


        dout = (out - kind) / 50
        dw = mat.T.dot(dout)

        w += -0.01 * dw

    test = CreateData(100)
    for i in test:
        r = np.array([i.x, i.y]).dot(w)
        color = ''
        index = GetMaxIndex(r)
        if index == 0:
            color = 'o-g'
        elif index == 1:
            color = 'o-r'
        plt.plot(i.x, i.y, color)

    # 设置坐标轴范围
    plt.xlim((-100, 100))
    plt.ylim((-100, 100))
    ax = plt.gca()  # gca:get current axis得到当前轴
    # 设置图片的右边框和上边框为不显示
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    # 挪动x，y轴的位置，也就是图片下边框和左边框的位置
    ax.spines['bottom'].set_position(('data', 0))  # data表示通过值来设置x轴的位置，将x轴绑定在y=0的位置
    ax.spines['left'].set_position(('axes', 0.5))  # axes表示以百分比的形式设置轴的位置，即将y轴绑定在x轴50%的位置，也就是x轴的中点

    plt.show()






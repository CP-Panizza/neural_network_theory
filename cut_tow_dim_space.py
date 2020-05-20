#分类问题核心思想：物以类聚，人以群分
#数学中用事物的属性组成的向量来表示一个物体，
#相同的事物的向量在向量空间中总是聚集在一起的
#我们可以通过分割向量空间来区分不同的事物


#权重矩阵的意义：
#权重矩阵的每一个列向量和训练数据集中的每一个数据向量是同一个维度下的向量
#权重矩阵中的每一个列向量在向量空间中指向不同的方向，这个方向就代表数据的分类



#训练的过程就是寻找这些分类向量的方向的过程！！！

import numpy as np
import matplotlib.pyplot as plt
import math
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        if x >= 0 and y >= 0:
            self.kind = [1, 0, 0, 0]
        elif x <= 0 and y >= 0:
            self.kind = [0, 1, 0, 0]
        elif x <= 0 and y <= 0:
            self.kind = [0, 0, 1, 0]
        elif x >= 0 and y <= 0:
            self.kind = [0, 0, 0, 1]

def random_mat(data):
    cut = np.random.random_integers(1000)
    l = []
    start = 0
    end = 0
    if cut - 50 > 0:
        start = cut - 50
        end = cut
        l = data[start:end]
    elif cut + 50 < 1000:
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

if __name__ == '__main__':
    data = []
    x = np.random.randint(20,size=10000)
    y = np.random.randint(20,size=10000)
    for i in range(1000):
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
    w = np.random.rand(2, 4)
    for i in range(2000):
        mat, start, end = random_mat(data)
        kind = get_kind(data, start, end)
        out = mat.dot(w)


        dout = (out - kind) / 50
        dw = mat.T.dot(dout)

        w += -0.01 * dw

    v1 = [w[0][0], w[1][0]]
    v2 = [w[0][1], w[1][1]]
    v3 = [w[0][2], w[1][2]]
    v4 = [w[0][3], w[1][3]]

    print("w:", w)
    print("norm v1:",Norm(v1))
    print("norm v2:",Norm(v2))
    print("norm v3:", Norm(v3))
    print("norm v4:", Norm(v4))

    k1 = v1[0] / v1[1]
    k2 = v2[0] / v2[1]
    k3 = v3[0] / v3[1]
    k4 = v4[0] / v4[1]

    print("k1:", (k1))
    print("k2:", (k2))
    print("k3:", (k3))
    print("k4:", (k4))

    fx1 = np.arange(0, 20, 0.5)
    fx2 = np.arange(-20, 0, 0.5)
    fy1 = k1 * fx1
    fy2 = k2 * fx2
    fy3 = k3 * fx2
    fy4 = k4 * fx1

    plt.plot(fx1, fy1, 'o-g')
    plt.plot(fx2, fy2, 'o-r')
    plt.plot(fx2, fy3, 'o-b')
    plt.plot(fx1, fy4, 'o-c')

    for x in range(100):
        dx = 0
        dy = 0
        if x % 2 == 0:
            t = np.random.randint(2, size=1)[0]
            if t % 2 == 0:
                dx = -np.random.randint(100, size=1)
                dy = np.random.randint(100, size=1)
            else:
                dx = -np.random.randint(100, size=1)
                dy = -np.random.randint(100, size=1)

        else:
            t = np.random.randint(2, size=1)[0]
            if t % 2 == 0:
                dx = np.random.randint(100, size=1)
                dy = -np.random.randint(100, size=1)
            else:
                dx = np.random.randint(100, size=1)
                dy = np.random.randint(100, size=1)


        r = np.array([dx[0], dy[0]]).dot(w)
        color = ''
        index = GetMaxIndex(r)
        if index == 0:
            color = 'o-g'
        elif index == 1:
            color = 'o-r'
        elif index == 2:
            color = 'o-b'
        elif index == 3:
            color = 'o-c'
        plt.plot(dx, dy, color)

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






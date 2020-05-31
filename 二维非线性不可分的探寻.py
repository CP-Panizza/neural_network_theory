import numpy as np
import matplotlib.pyplot as plt
import math
from two_layer_net import TwoLayerNet

#尝试对二维点进行非线性三分类
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        if x >= 0 and y >= 0:
            self.kind = [1, 0, 0]
        elif x >= 0 and y <= 0:
            self.kind = [0, 1, 0]
        else:
            self.kind = [0, 0, 1]

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
    net = TwoLayerNet(2, 3, 3)
    data = CreateData(2000)
    for i in data:
        if GetMaxIndex(i.kind) == 0:
            plt.plot(i.x, i.y, 'o-g')
        elif GetMaxIndex(i.kind) == 1:
            plt.plot(i.x, i.y, 'o-b')
        else:
            plt.plot(i.x, i.y, 'o-r')
    plt.show()

    for i in range(5000):
        x_batch, start, end = random_mat(data)
        t_batch = get_kind(data, start, end)
        grad = net.gradient(x_batch, t_batch)

        # 更新参数
        for key in ('W1', 'b1', 'W2', 'b2'):
            net.params[key] -= 0.01 * grad[key]

    testData = CreateData(800)
    for i in testData:
        if GetMaxIndex(net.predict(np.array([i.x, i.y]))) == 0:
            plt.plot(i.x, i.y, 'o-g')
        elif GetMaxIndex(net.predict(np.array([i.x, i.y]))) == 1:
            plt.plot(i.x, i.y, 'o-b')
        elif GetMaxIndex(net.predict(np.array([i.x, i.y]))) == 2:
            plt.plot(i.x, i.y, 'o-r')
    plt.show()
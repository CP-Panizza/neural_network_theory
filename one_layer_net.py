# coding: utf-8
import sys, os

sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
from common.functions import *
from common.gradient import numerical_gradient

from dataset.mnist import load_mnist
from PIL import Image

def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

class OneLayerNet:

    def __init__(self, input_size, output_size, weight_init_std=0.01):
        # 初始化权重
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, output_size)
        self.params['b1'] = np.zeros(output_size)


    def predict(self, x):
        W1 = self.params['W1']
        b1 = self.params['b1']

        a1 = np.dot(x, W1) + b1

        y = softmax(a1)

        return y

    # x:输入数据, t:监督数据
    def loss(self, x, t):
        y = self.predict(x)

        return cross_entropy_error(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    # x:输入数据, t:监督数据
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)

        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])

        return grads

    def gradient(self, x, t):
        W1 = self.params['W1']
        b1 = self.params['b1']
        grads = {}

        batch_num = x.shape[0]

        # forward
        a1 = np.dot(x, W1) + b1
        y = softmax(a1)

        # backward
        dy = (y - t) / batch_num
        grads['W1'] = np.dot(x.T, dy)
        grads['b1'] = np.sum(dy, axis=0)

        return grads


def GetMaxIndex(v):
    max = v[0]
    index = 0
    for i in range(len(v)):
        if v[i] > max:
            max = v[i]
            index = i
    return index

if __name__ == '__main__':
    net = OneLayerNet(input_size=784, output_size=10)
    iters_num = 10000  # 适当设定循环的次数
    train_size = x_train.shape[0]  # 60000

    batch_size = 100
    learning_rate = 0.1


    for i in range(iters_num):
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]
        grad = net.gradient(x_batch, t_batch)
        # 更新参数
        for key in ('W1', 'b1'):
            net.params[key] -= learning_rate * grad[key]

        loss = net.loss(x_batch, t_batch)
        print("loss:", loss)


    result = GetMaxIndex(net.predict(x_test[0]))
    print("result:", result)
    print("t_test[0]",GetMaxIndex(t_test[0]))


    col = np.array([i[1] for i in net.params["W1"]])

    #权重矩阵中的每一个列向量都是这个权重矩阵列向量构成的向量空间中的一组向量
    #神经网络输入数据X是一组向量，X与W两个矩阵进行点积，W把X转化成高维空间或者低维空间中的向量
    #神经网络分类的思想是通过向量空间的高低维度相互转化，并通过分割向量空间的原理进行目标分类
    print(net.params["W1"].T.dot(net.params["W1"]))


    #W权重矩阵中的每一个列向量可以理解成训练样本某一个类别的特征向量
    col = col + np.abs(np.min(col))
    print("np.sum(col):", np.sum(col))
    col = col / np.sum(col) * 25500
    print(np.max(col), " ", np.min(col))
    img = col.reshape(28, 28)
    img_show(img)

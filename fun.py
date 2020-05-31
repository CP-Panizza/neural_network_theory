import numpy as np
import matplotlib.pyplot as plt

def relu(x):
    return np.maximum(0, x)

def f1(x):
    return x*4 + 8

def f2(x):
    return 0.5 *x + 3.5


x = np.arange(-10, 10, 0.5)
y1 = f1(x)
y = f2(relu(y1))


plt.plot(x, y)
plt.show()
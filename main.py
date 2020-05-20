learnning_rate = 0.01
class One:
    def __init__(self, price, count):
        self.price = price
        self.count = count
        self.dPrice = None
        self.dCount = None

    def forword(self):
        return self.price * self.count



    def back(self, d):
        self.dPrice = d * self.count
        self.dCount = d * self.price
        return d * self.count, d * self.price



if __name__ == '__main__':
    net = One(100, 2)

    target = 600

    for i in range(1000):
        z = net.forword()
        print("z:", z)
        dz = z - target
        print("dz:", dz)
        dPrice, dCount =  net.back(dz)
        print("dPrice:", dPrice)
        net.price += -learnning_rate * dPrice

    print("net.count:",net.count)
    print("net.price:", net.price)
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
    apple_count = 100
    totle_money = 600
    apple_price = 2
    net = One(apple_price, apple_count)



    for i in range(10000):
        z = net.forword()
        print("z:", z)
        dz = z - totle_money
        print("dz:", dz)
        dPrice, dCount =  net.back(dz)
        print("dCount:", dCount)
        net.count += -learnning_rate * dCount

    print("net.count:",net.count)
    print("net.price:", net.price)
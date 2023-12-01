# N点序列FFT计算，输入顺序x序列，输出顺序X序列
import math
from math import sin, cos, pi, ceil, log2, pow
from numpy import conj
import matplotlib.pyplot as plt


class FFT_algorithm:
    def __init__(self, _list=[], N=0):
        self.list = _list  # 待处理原序列
        self.N = N
        # 以下各变量补零后再进行计算
        self.reverse_list = []  # 倒位序时域序列
        self.output = []
        self.total = 0  # 计算深度
        self.W = []  # Wnk系数

    '''
        for p in range(len(self.list)):
            self.reverse_list.append(self.list[self.reverse_pos(p)])
        self.output = self.reverse_list.copy()
    '''

    def zero_fill(self):
        n = int(pow(2, ceil(log2(self.N))))
        if n == self.N:
            return
        for i in range(self.N, n):
            self.list.append(0)
        self.N = n
        return

    def reverse_init(self):
        for p in range(len(self.list)):
            self.reverse_list.append(self.list[self.reverse_pos(p)])
        self.output = self.reverse_list.copy()
        N = self.N
        for k in range(self.N):
            self.W.append((cos(2 * pi / N) - sin(2 * pi / N) * 1j) ** k)

    def reverse_pos(self, num):
        out = 0  # 倒位序结果
        bits = 0
        i = self.N
        while i != 0:
            i = i // 2
            bits = bits + 1
        data = num
        for i in range(bits - 1):
            out = out << 1
            out |= (data >> i) & 1
        self.total = bits - 1
        return out

    def fft(self, _abs=True):  # _abs决定结果是否取模值
        for m in range(self.total):
            split = self.N // (2 ** (m + 1))
            num_each = self.N // split
            for i in range(split):
                for j in range(num_each // 2):
                    temp = self.output[i * num_each + j]
                    temp2 = self.output[i * num_each + j + num_each // 2] * self.W[j * 2 ** (self.total - m - 1)]
                    self.output[i * num_each + j] = temp + temp2
                    self.output[i * num_each + j + num_each // 2] = temp - temp2
        if _abs:
            for k in range(len(self.output)):
                self.output[k] = abs(self.output[k])
        return self.output

    def fft_normalize(self, _abs=True):
        self.fft(_abs=_abs)
        max = 0
        if _abs:
            for i in range(self.N):
                if max < self.output[i]:
                    max = self.output[i]
        else:
            for i in range(self.N):
                if max < abs(self.output[i]):
                    max = abs(self.output[i])
        for k in range(self.N):
            self.output[k] = self.output[k] / max
        return self.output

    def ifft(self, _abs=True):
        for i in range(self.N):
            self.reverse_list[i] = conj(self.reverse_list[i])
            self.output[i] = conj(self.output[i])
        'print(self.output)'
        self.fft(_abs)
        for j in range(self.N):
            self.output[j] = self.output[j] / self.N
        return self.output

    def dft(self, _abs=True):
        list = self.list.copy()
        N = self.N
        for k in range(self.N):
            temp = 0
            for n in range(self.N):
                temp += list[n] * ((cos(2*pi/N)-sin(2*pi/N)*1j)**(n*k))
            if _abs:
                self.output[k] = abs(temp)
            else:
                self.output[k] = temp
        return self.output

    def frequency_plot(self):  # 输出序列是DTFT 2pi/N 的等间隔采样，以0-2pi(k/N)为坐标轴绘图
        x = []
        for k in range(self.N):
            x.append(2*k*math.pi/self.N)
        y = self.output.copy()
        plt.xlabel("w/rad")
        plt.ylabel("|X(K)|")
        plt.title("|X(K)|--Frequency")
        plt.stem(x, y)  # 绘制火柴图
        plt.show()

if __name__ == '__main__':
    T = 0.75
    list_t = []
    for n in range(10000):
        list_t.append(pow(math.e, -0.1*n*T))
    'list_t = [0, 1, 2, 3, 4, 5, 6, 7]'
    a = FFT_algorithm(list_t, len(list_t))
    a.zero_fill()
    a.reverse_init()
    print(a.fft())


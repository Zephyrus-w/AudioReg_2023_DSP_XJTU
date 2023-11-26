import soundfile as sf
import matplotlib.pyplot as plt

def average(list, start, length):#注意这里算的是绝对值
    sum = 0
    for i in range(length):
        sum += abs(list[start + i])
    return sum / length
'''
以下请修改文件名
'''
# 读取音频文件
wavsignal, rt = sf.read('dataset/original/original_9.wav')
# rt是采样频率，wavsignal.shape[0]存储了音频的长度

# 单通道音频
length = wavsignal.shape[0]
print("sampling rate = {} Hz, length = {} samples, channels = 1".format(rt, length))
plt.plot(wavsignal)


plt.show()
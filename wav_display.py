import soundfile as sf
import matplotlib.pyplot as plt

# 读取音频文件
wavsignal, rt = sf.read('dataset/original/original_1.wav')


# 单通道音频
length = wavsignal.shape[0]
print("sampling rate = {} Hz, length = {} samples, channels = 1".format(rt, length))
plt.plot(wavsignal)


plt.show()

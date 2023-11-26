import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt

Threshold_value_high = 2 #0.5
Threshold_value_low = 0.000000000000000000000000000000000000000000000000000000001
Interval_high = 50
Interval_low = 23
Interval_check = 100000
Threshold_value_mean = 1
SampleNumber = 17

def checkenergy(index_low,index_high,data,energy_mean):
    sum = 0
    for index in range(index_high-index_low):
        sum += data[index + index_low] ** 2   #        sum += data[index + index_low][1] ** 2
    if sum > (Threshold_value_mean * energy_mean):
        return True
    return False

def endpoint_detection(wav_file_path):

    sample_rate, data = wav.read(wav_file_path)# 读取wav文件
    data = data.astype(np.float64)# 转换为浮点数数组
    # energy = np.sum(data ** 2, axis=0)# 计算能量    ##axis
    energy = np.array([elem ** 2 for elem in data])
    energy /= np.max(abs(energy))# 归一化能量    energy /= np.max(energy)
    threshold_value_high = Threshold_value_high * np.mean(energy)
    threshold_value_low = Threshold_value_low * np.mean(energy) # 设置阈值

    startpoints_flag = True
    startpoints = []
    endpoints = []
    index = 0
    index_low,index_high=0,0
    while(index < len(energy)):
        if(startpoints_flag and energy[index] > threshold_value_high):
            index_low = index
            startpoints_flag = False
            index += int(len(energy)/Interval_high)
        elif(not startpoints_flag and energy[index] < threshold_value_low):
            startpoints_flag = True
            index_high = index
            if(checkenergy(index_low,index_high,data,np.mean(energy))):
                while(energy[index_high+int(len(energy)/Interval_check)] > threshold_value_low ):#* 10000000
                    index_high += int(len(energy)/Interval_check)
                startpoints.append(index_low)
                endpoints.append(index_high)
            index += int(len(energy)/Interval_low)
        else:
            index = index + 1

    # 绘制波形图
    time = np.arange(0, len(data)) / sample_rate
    plt.figure(figsize=(10, 6))
    plt.plot(time, data[:], label='Channel 1')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Waveform')

    # 在端点位置添加垂直线
    for startpoint in startpoints:
        plt.axvline(x=startpoint/sample_rate, color='r', linestyle='--')
    for endpoint in endpoints:
        plt.axvline(x=endpoint/sample_rate, color='b', linestyle='--')

    #plt.legend()
    plt.show()

    # 绘制能量图
    time = np.arange(0, len(energy)) / sample_rate
    plt.figure(figsize=(10, 4))
    plt.plot(time, energy)
    plt.xlabel('Time (s)')
    plt.ylabel('Energy')
    plt.title('Energy')
    #plt.axhline(y=threshold_value_high, color='r', linestyle='--', label='Threshold')
    #plt.legend()
    plt.show()

    return sample_rate, data, energy, startpoints, endpoints

file_path = 'D:\Desktop\DSP_lab\sample'
for index in range(SampleNumber):
    wav_file_path = file_path + '\original_' + str(index + 1) + '.wav'
    sample_rate, data, energy, startpoints, endpoints = endpoint_detection(wav_file_path)
    # 打印端点位置
    print(f"第{index+1}个端点位置(起点)：", startpoints)
    print(f"第{index+1}个端点位置(终点)：", endpoints)
import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import pandas as pd
import csv
import math
import json  #用于输出json格式文件
from checkenergy import checkenergy

file_path = 'D:\Desktop\DSP_lab\sample' #存放.wav文件的文件夹
OutputFilePath = 'D:\Desktop\DSP_lab\output_data'
Threshold_value_high = 2 #端点检测的阈值比例系数
Threshold_value_low = 0.000000000000000000000000000000000000000000000000000000001
Interval_high = 50  #间隔系数
Interval_low = 23
Interval_check = 100000
Threshold_value_mean = 1
FrameTime = 0.02 # 每帧信号的时长，单位秒
SampleNumber = 17  #wav文件数量

def detection(wav_file_path): # 端点检测函数

    sample_rate, data = wav.read(wav_file_path)# 读取wav文件
    data = data.astype(np.float64)# 转换为浮点数数组
    energy = np.array([elem ** 2 for elem in data]) #data中每个信号值对应的能量
    energy /= np.max(abs(energy))# 归一化能量
    threshold_value_high = Threshold_value_high * np.mean(energy)
    threshold_value_low = Threshold_value_low * np.mean(energy) # 设置阈值

    startpoints_flag = True
    startpoints = []
    endpoints = []
    index = 0
    index_low,index_high=0,0
    while(index < len(energy)):
        if(startpoints_flag and energy[index] > threshold_value_high): #是否检测到start端点
            index_low = index
            startpoints_flag = False
            index += int(len(energy)/Interval_high)
        elif(not startpoints_flag and energy[index] < threshold_value_low):  #是否检测到end端点
            startpoints_flag = True
            index_high = index
            if(checkenergy(index_low,index_high,data,np.mean(energy))):
                while(energy[index_high+int(len(energy)/Interval_check)] > threshold_value_low ):#* 10000000
                    index_high += int(len(energy)/Interval_check)
                startpoints.append(index_low)
                endpoints.append(index_high)
            index += int(len(energy)/Interval_low)
        else: #没有检测到端点
            index = index + 1

    # # 绘制波形图
    time = np.arange(0, len(data)) / sample_rate
    plt.figure(figsize=(10, 6))
    plt.plot(time, data[:], label='Channel 1')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Waveform')

    # 在start和end端点位置添加垂直线
    # for startpoint in startpoints:
    #     plt.axvline(x=startpoint/sample_rate, color='r', linestyle='--')
    #     # plt.axhline(y=energy[int(startpoint / sample_rate)], color='r', linestyle='--')
    # for endpoint in endpoints:
    #     plt.axvline(x=endpoint/sample_rate, color='b', linestyle='--')
    #     # plt.axhline(y=energy[int(endpoint / sample_rate)], color='g', linestyle='--')
    # plt.show()

    # 绘制能量图
    time = np.arange(0, len(energy)) / sample_rate
    # plt.figure(figsize=(10, 4))
    # plt.plot(time, energy)
    # plt.xlabel('Time (s)')
    # plt.ylabel('energy')
    # plt.title('energy')
    #plt.show()

    return sample_rate, data, energy, startpoints, endpoints  #返回采样率，读入的.wav总数据data，归一化后的能量列表energy，startpoint信号开始的端点，endpoint信号结束的端点

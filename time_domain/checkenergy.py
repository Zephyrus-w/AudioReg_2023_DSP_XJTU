import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import pandas as pd
import csv
import math
import json  #用于输出json格式文件

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

def checkenergy(index_low,index_high,data,energy_mean):  #检查函数，在端点检测中用于检测是否确实是end端点
    sum = 0
    for index in range(index_high-index_low):
        sum += data[index + index_low] ** 2   #        sum += data[index + index_low][1] ** 2
    if sum > (Threshold_value_mean * energy_mean):
        return True
    return False

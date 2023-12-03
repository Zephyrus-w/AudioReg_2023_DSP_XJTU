import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import pandas as pd
import csv
import math
import json  #用于输出json格式文件
from sgn import sgn  #符号函数
from checkenergy import checkenergy  #检查函数，在端点检测中用于检测是否确实是end端点
from detection import detection  #端点检测函数
PI = math.pi #圆周率

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

signal = [[[] for index_temp in range(10)] for index in range(SampleNumber)]  #用于端点检测后，收集每个人的0~9数字信号区间，后续处理中存在归一化操作，此处尚未归一化
signal_average = [[0 for index_temp in range(10)] for index in range(SampleNumber)] # 用于计算第index个人的数字number的平均幅值，用于去直流
signal_cutDirectCurrent = [[[] for index_temp in range(10)] for index in range(SampleNumber)] #signal去直流后的signal
energy = [[0 for index_temp in range(10)] for index in range(SampleNumber)]   #用于存放能量
energy_average = [[0 for index_temp in range(10)] for index in range(SampleNumber)]   #用于存放平均能量
Z = [[0 for index_temp in range(10)] for index in range(SampleNumber)]        #用于存放过零率
Z_average = [[0 for index_temp in range(10)] for index in range(SampleNumber)]        #用于存放平均过零率百分比
RMS = [[0 for index_temp in range(10)] for index in range(SampleNumber)]  #信号的均方根
variance = [[0 for index_temp in range(10)] for index in range(SampleNumber)]  #信号的方差
vector = [[[]for index in range(SampleNumber)]for index_temp in range(10)] #特征向量
#分帧之后的各个参数
signal_divide = [[[] for index_temp in range(10)] for index in range(SampleNumber)]  #存放分帧之后的各个信号段
signal_cutDirectCurrent_divide = [[[] for index_temp in range(10)] for index in range(SampleNumber)] #signal去直流后的signal
energy_average_divide = [[[] for index_temp in range(10)] for index in range(SampleNumber)]  #
# Z_divide = [[[] for index_temp in range(10)] for index in range(SampleNumber)]        #用于存放过零次数
Z_average_divide = [[[] for index_temp in range(10)] for index in range(SampleNumber)]        #用于存放平均过零率百分比
vector_divide = [[[]for index in range(SampleNumber)]for index_temp in range(10)]
#vector_divide是分帧后的特征向量大列表
#vector_divide[number][index][0]表示第index个人的数字number语音信号中各帧的平均平均能量
#vector_divide[number][index][1]表示第index个人的数字number语音信号中各帧的最大平均能量
#vector_divide[number][index][2]表示第index个人的数字number语音信号中各帧的平均过零率
#vector_divide[number][index][3]表示第index个人的数字number语音信号中各帧的最大过零率
for index in range(SampleNumber):#第index个人
    max = -1 #统计第index个人的0~9的发音中的幅值最大值，用于后续归一化处理
    print(f"第{index + 1}个人的语音数据样本信息：")
    wav_file_path = file_path + '\original_' + str(index + 1) + '.wav'  #读入第index个人的.wav文件
    sample_rate, data, energy_ori, startpoints, endpoints = detection(wav_file_path)  #端点检测
    # 打印端点位置
    print(f"第{index+1}个人的.wav文件中端点位置(起点)：", startpoints)
    print(f"第{index+1}个人的.wav文件中端点位置(终点)：", endpoints)
    potsNumber = int(FrameTime * sample_rate) #每帧信号的序列长度##############################
    for number in range(10):  # 第index个人的数字0~9的发音
        for index_data in range(endpoints[number] - startpoints[number] + 1):  #将第index个人的第number个数字语音从data中存入signal[index][number]列表(仅存入端点检测后的)
            signal[index][number].append(data[startpoints[number] + index_data])
            signal_average[index][number] += data[startpoints[number] + index_data]
            energy[index][number] += data[startpoints[number] + index_data] ** 2  #计算能量值
            if(abs(data[startpoints[number] + index_data]) > max):  #统计第index个人的0~9的发音中的幅值最大值，用于后续归一化处理
                max = abs(data[startpoints[number] + index_data])
        signal_average[index][number] = signal_average[index][number] / len(signal[index][number])  #signal_average未进行幅值归一化
        for index_data in range(len(signal[index][number])):
            signal_cutDirectCurrent[index][number].append(signal[index][number][index_data] - signal_average[index][number])  #去直流，未幅值归一化

    for number in range(10):  # 第index个人的数字0~9的发音的归一化处理
        #归一化处理
        for index_normalize in range(len(signal[index][number])):
            signal[index][number][index_normalize] /= max  #信号幅值归一化
        energy[index][number] /= max ** 2  #能量归一化
        energy_average[index][number] = energy[index][number] / len(signal[index][number]) #归一化后的能量的平均值
        RMS[index][number] = math.sqrt(energy_average[index][number])# 计算方均根
        mean = 0 #用于计算归一化幅值后信号的平均值
        for index_mean in range(len(signal[index][number])):
              mean += signal[index][number][index_mean ] #信号幅值归一化后的信号均值
        mean = mean / len(signal[index][number])
        var = 0;#用于计算方差
        for index_var in range(len(signal[index][number])):
            var += (signal[index][number][index_var] - mean) ** 2
        var /= len(signal[index][number])
        variance[index][number] = var


        print(f"第{index+1}个人的第{number}帧信号：")
        print("短时能量为:",energy[index][number])

        for index_data in range(len(signal_cutDirectCurrent[index][number]) - 1):  #用去直流化信号 统计第index个人的第number语音的过零率
            Z[index][number] += abs((sgn(signal_cutDirectCurrent[index][number][index_data + 1])-sgn(signal_cutDirectCurrent[index][number][index_data])))
        Z[index][number]  = Z[index][number]  / 2
        Z_average[index][number] = Z[index][number] / len(signal_cutDirectCurrent[index][number]) * 100
        print("过零率为：",Z[index][number])

    print()
    for number in range(10):
        # vector[number][index][0] = energy[index][number]# 幅值归一化后的能量
        # vector[number][index][1] = energy_average[index][number] #幅值归一化后的平均能量
        # vector[number][index][2] = Z[index][number] #去直流化信号的过零次数
        # vector[number][index][3] = Z_average[index][number] #去直流化信号的过零百分率
        # vector[number][index][4] = RMS[index][number]  #信号的方均根
        # vector[number][index][5] = variance[index][number] #信号的方差
        vector[number][index].append(energy_average[index][number])  # 幅值归一化后的平均能量
        vector[number][index].append(Z_average[index][number])  # 去直流化信号的过零百分率

    # 写数据处理文档
    fileName = OutputFilePath + '\OutputFile.txt'
    with open(fileName, "a") as fid:
        fid.write(f'第{index + 1}个人的数据：' + '\n')
        fid.write(f'    数据采样率：' + str(sample_rate) +  '\n')
        for number in range(10):
            fid.write(f'    数字{number}的端点：[' + str(startpoints[number]) + ',' + str(endpoints[number]) + ']'
                      + '    点长：' + str(endpoints[number] - startpoints[number] + 1) + '    短时能量：' + str(energy[index][number])
                      + '过零率：' + str(Z[index][number])
                      + '过零率百分比：' + str(Z_average[index][number]) + '%' + '\n')
        fid.write('\n')
        fid.close()
#分帧(并加窗)
    for number in range(10):
        start = 0
        end = endpoints[number] - startpoints[number] #end为第index个人的number数字语音信号的长度 - 1
        orisignal = signal[index][number]  #第index个人的number数字语音信号
        frameNumber = 0
        while (start < end):
            if (start + potsNumber <= end): #插入一段长度为potsNumer的序列，为一帧
                signal_divide[index][number].append([signal[index][number][index_elem + start] for index_elem in range(potsNumber)])
                frameNumber += 1
                start += potsNumber
            elif (start + potsNumber > end):  #插入剩余全部的序列，为最后一帧
                signal_divide[index][number].append([signal[index][number][index_elem + start] for index_elem in range(end - start + 1)])
                frameNumber += 1
                start = end
        #此时计算得到第index个人的number数字语音信号被分为frameNumber帧,存放在signal_divide四层列表中
        for index_frame in range(frameNumber):
            frameLength = len(signal_divide[index][number][index_frame])  #帧长
            HammingWindow = [(0.54-0.46*math.cos(2*PI*n/(frameLength - 1))) for n in range(frameLength)]
            signal_divide[index][number][index_frame] = [HammingWindow[j]*signal_divide[index][number][index_frame][j] for j in range(frameLength)]
            mean_signal = sum(signal_divide[index][number][index_frame]) / frameLength
            #对每一帧进行去直流化，存入signal_cutDirectCurrent_divide
            signal_cutDirectCurrent_divide[index][number].append([(elem - mean_signal) for elem in signal_divide[index][number][index_frame]])
            #对每一帧求其平均能量
            sumofenergy = 0
            for j in range(len(signal_divide[index][number][index_frame])):
                sumofenergy += signal_divide[index][number][index_frame][j]**2
            sumofenergy /= len(signal_divide[index][number][index_frame])
            energy_average_divide[index][number].append(sumofenergy)
            #对每一帧求其过零率
            sumZ = 0
            for index_Z in range(len(signal_cutDirectCurrent_divide[index][number][index_frame]) - 1):  # 用去直流化信号 统计第index个人的第number语音的过零率
                sumZ += abs((sgn(signal_cutDirectCurrent_divide[index][number][index_frame][index_Z + 1]) - sgn(signal_cutDirectCurrent_divide[index][number][index_frame][index_Z])))
            sumZ = sumZ / 2
            Z_average_divide[index][number].append(sumZ / len(signal_cutDirectCurrent_divide[index][number][index_frame]) * 100)
#对每一帧求四个指标并组成特征向量
        sum_energy_average_divide = 0
        max_energy_average_divide = -1
        sum_Z_average_divide = 0
        max_Z_average_divide = -1
        for index_frame in range(frameNumber):
            sum_energy_average_divide += energy_average_divide[index][number][index_frame]
            if(max_energy_average_divide < energy_average_divide[index][number][index_frame]):
                max_energy_average_divide = energy_average_divide[index][number][index_frame]
            sum_Z_average_divide += Z_average_divide[index][number][index_frame]
            if(max_Z_average_divide < Z_average_divide[index][number][index_frame]):
                max_Z_average_divide = Z_average_divide[index][number][index_frame]
        mean_energy_average_divide = sum_energy_average_divide / frameNumber
        mean_Z_average_divide = sum_Z_average_divide / frameNumber

        vector_divide[number][index].append(mean_energy_average_divide)
        vector_divide[number][index].append(max_energy_average_divide)
        vector_divide[number][index].append(mean_Z_average_divide)
        vector_divide[number][index].append(max_Z_average_divide)

with open('D:\Desktop\DSP_lab\output_data\signal_divide.json', 'w') as file:
    json.dump(signal_divide, file)
with open('D:\Desktop\DSP_lab\output_data\Vector_divide.json', 'w') as file:
    json.dump(vector_divide, file)

    # fileName1 = OutputFilePath + '\Vector' +str(index) +'.txt'
    # with open(fileName1, "a") as fid:
    #     for number in range(10):
    #         fid.write(str(vector[number][index][0]) + ',' + str(vector[number][index][1]) + ',' + str(vector[number][index][2]) + ',' + str(vector[number][index][3]) + '\n')
    #     fid.close()
#写数据处理结果excel表格
df = pd.DataFrame(energy)
df.to_excel(OutputFilePath + '\energy.xlsx', index=True, header=True)
df = pd.DataFrame(energy_average)
df.to_excel(OutputFilePath + '\energy_average.xlsx', index=True, header=True)
df = pd.DataFrame(Z)
df.to_excel(OutputFilePath + '\Z.xlsx', index=True, header=True)
df = pd.DataFrame(Z_average)
df.to_excel(OutputFilePath + '\Z_average.xlsx', index=True, header=True)


vectorfile = OutputFilePath + '\Vector'  + '.csv'
with open(vectorfile, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerows(vector)
# # 补零至2的L次方
# signal_add0 = [[[elem for elem in signal[index][index_temp]] for index_temp in range(10)] for index in range(SampleNumber)]
# for index in range(SampleNumber):
#     for index_temp in range(10):
#         if len(signal_add0[index][index_temp]) <= 2 ** 14 and len(signal_add0[index][index_temp]) > 2 ** 13:
#             for _ in range(2**14 - len(signal_add0[index][index_temp])):
#                 signal_add0[index][index_temp].append(0)
#         elif len(signal_add0[index][index_temp]) <= 2 ** 15 and len(signal_add0[index][index_temp]) > 2 ** 14:
#             for _ in range(2**15 - len(signal_add0[index][index_temp])):
#                 signal_add0[index][index_temp].append(0)
#         elif len(signal_add0[index][index_temp]) <= 2 ** 16 and len(signal_add0[index][index_temp]) > 2 ** 15:
#             for _ in range(2**16 - len(signal_add0[index][index_temp])):
#                 signal_add0[index][index_temp].append(0)
print()
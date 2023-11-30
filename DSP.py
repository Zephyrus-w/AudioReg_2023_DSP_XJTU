import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import pandas as pd
import csv
import math

file_path = 'D:\Desktop\DSP_lab\sample' #存放.wav文件的文件夹
OutputFilePath = 'D:\Desktop\DSP_lab\output_data'
Threshold_value_high = 2 #0.5
Threshold_value_low = 0.000000000000000000000000000000000000000000000000000000001
Interval_high = 50
Interval_low = 23
Interval_check = 100000
Threshold_value_mean = 1
SampleNumber = 17

def sgn(x): #符号函数
    if x > 0:
        return 1
    elif x < 0:
        return -1
    return 0

def checkenergy(index_low,index_high,data,energy_mean):  #检查函数，在端点检测中用于检测是否确实是end端点
    sum = 0
    for index in range(index_high-index_low):
        sum += data[index + index_low] ** 2   #        sum += data[index + index_low][1] ** 2
    if sum > (Threshold_value_mean * energy_mean):
        return True
    return False

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
    for startpoint in startpoints:
        plt.axvline(x=startpoint/sample_rate, color='r', linestyle='--')
        # plt.axhline(y=energy[int(startpoint / sample_rate)], color='r', linestyle='--')
    for endpoint in endpoints:
        plt.axvline(x=endpoint/sample_rate, color='b', linestyle='--')
        # plt.axhline(y=energy[int(endpoint / sample_rate)], color='g', linestyle='--')
    plt.show()

    # 绘制能量图
    time = np.arange(0, len(energy)) / sample_rate
    # plt.figure(figsize=(10, 4))
    # plt.plot(time, energy)
    # plt.xlabel('Time (s)')
    # plt.ylabel('energy')
    # plt.title('energy')
    #plt.show()

    return sample_rate, data, energy, startpoints, endpoints  #返回采样率，读入的.wav总数据data，归一化后的能量列表energy，startpoint信号开始的端点，endpoint信号结束的端点

# file_path = 'D:\Desktop\DSP_lab\sample' #存放.wav文件的文件夹    放到头部了
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

for index in range(SampleNumber):#第index个人
    max = -1 #统计第index个人的0~9的发音中的幅值最大值，用于后续归一化处理
    print(f"第{index + 1}个人的语音数据样本信息：")
    wav_file_path = file_path + '\original_' + str(index + 1) + '.wav'  #读入第index个人的.wav文件
    sample_rate, data, energy_ori, startpoints, endpoints = detection(wav_file_path)  #端点检测
    # 打印端点位置
    print(f"第{index+1}个人的.wav文件中端点位置(起点)：", startpoints)
    print(f"第{index+1}个人的.wav文件中端点位置(终点)：", endpoints)
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
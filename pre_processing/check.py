import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import os

# 构建数据文件的相对路径
data_file_path = os.path.join(os.path.dirname(__file__), '..', 'dataset','original', 'original_1.wav')

# 确保路径是绝对路径
absolute_data_file_path = os.path.abspath(data_file_path)

def average(list, start, length):#注意这里算的是绝对值
    sum = 0
    for i in range(length):
        sum += abs(list[start + i])
    return sum / length
def get_split_point(wavsignal):#和原来的split不完全相同

    split_points=[]
    # 单通道音频
    length = wavsignal.shape[0]

    #总共有length个点，我们可以新建1个10行数组，每一行都是一个数字的音频。
    processed_signals = [[] for _ in range(10)]

    #接下来朝这个数组里面添加数据，最重要的是找到对应的开始点i和结束点k   
    number=0
    i=0
    while( i<length and number < 10 ):
        if (wavsignal[i]>=0.02 and average(wavsignal, i, 1000)>0.03):

            '''
            这里说明已经进入了有效数字阶段，进入的坐标是i。
            接下来的任务是找结尾k，我们希望要么之后的一些都能<0.1，要么k已经足够大，接近length，才能说它已经不是有效字了
            '''

            k=i
            while(k<length):
                #第一种情况，k已经很大，以至于不能够统计接下来的1000个样本，此时认为k已经够大了，就是这个数字的结尾。
                #之后要做的就是结束整个循环
                if (k+1000>=length):
                    for index in range(i,k):
                        processed_signals[number].append(wavsignal[index])

                    print(f"数字{number}已完成采集，其长度为{k-i}")
                    split_points.append((i, k))
                    i=k
                    number=number+1
                    break

                elif( abs(wavsignal[k])<0.02 and average( wavsignal, k, 1000) < 0.01):
                #第二种情况，k还不够大，则我们要在样本够小的情况下，统计之后的1000个样本，如果这些样本的绝对值够小，那么则认为k已经是结尾。
                    
                    for index in range(i,k):
                        processed_signals[number].append(wavsignal[index])
                    
                    print(f"数字{number}已完成采集，其长度为{k-i}")
                    split_points.append((i, k))
                    number=number+1
                    i=k
                    break
    
                k+=1  
        i+=1
                            
    return split_points



# 加载音频文件
wavsignal, samplerate = sf.read(absolute_data_file_path)

# 执行分割
split_points = get_split_point(wavsignal)

# 绘制波形图
plt.figure(figsize=(12, 6))
plt.plot(wavsignal, label='Waveform')
plt.xlabel('Samples')
plt.ylabel('Amplitude')

# 标记分割点
for i, k in split_points:
    plt.axvline(x=i, color='r', linestyle='--', label='Start' if i == split_points[0][0] else "")
    plt.axvline(x=k, color='g', linestyle=':', label='End' if k == split_points[0][1] else "")

# 添加图例
plt.legend()
plt.title('Waveform with Split Points')
plt.show()

import os
import joblib
import numpy as np
import json
import soundfile as sf
import matplotlib.pyplot as plt
import os
import math

bayesian_model = joblib.load('bayesian_model.pkl')
fisher_model = joblib.load('fisher_model.pkl')
SVM_model=joblib.load('SVM_model.pkl')
decision_tree_model = joblib.load('decision_tree_model.pkl')

# 构建数据文件的相对路径
data_file_path = os.path.join(os.path.dirname(__file__), '..','test.wav')
# 确保路径是绝对路径
absolute_data_file_path = os.path.abspath(data_file_path)
wavsignal, rt = sf.read(absolute_data_file_path)

def average(list, start, length):#注意这里算的是绝对值
    sum = 0
    for i in range(length):
        sum += abs(list[start + i])
    return sum / length

def signal_cut(wavsignal):
    processed_signals = []
    head_and_tail=[0,0]
    length = len(wavsignal)
    for i in range(len(wavsignal)):
        if (wavsignal[i]>=0.02 and average(wavsignal, i, 2000)>0.025):
            head_and_tail[0]=i
            k=i
            while(k<length):
                #第一种情况，k已经很大，以至于不能够统计接下来的1000个样本，此时认为k已经够大了，就是这个数字的结尾。
                #之后要做的就是结束整个循环
                if ((k+1000>=length) or ( abs(wavsignal[k])<0.02 and average( wavsignal, k, 5000) < 0.01)):
                    head_and_tail[1]=k
                    for index in range(i,k):
                        processed_signals.append(wavsignal[index])
                    break
                k+=1  
            break
    return processed_signals, head_and_tail
def wav_display(wavsignal, head_and_tail):
        # 绘制原始信号
    plt.figure(figsize=(12, 6))
    plt.plot(wavsignal, label='Original Signal')

    # 在头部位置画红色虚线
    plt.axvline(x=head_and_tail[0], color='r', linestyle='--', label='Start')

    # 在尾部位置画蓝色虚线
    plt.axvline(x=head_and_tail[1], color='b', linestyle='--', label='End')

    plt.title("Signal with Head and Tail Marked")
    plt.legend()
    plt.show()

def extract_features(list_d2):
    features=[]  #存放特征向量:  [这一帧的平均平均能量，这一帧的最大平均能量，这一帧的平均过零率，帧的最大过零率]
    energy = []  #存放各帧平均能量
    Z = []  #存放各帧过零率
    #求各帧的能量和过零率
    for i in range(len(list_d2)):  #访问各帧
        #计算各帧平均能量
        sum = 0
        for j in range(list_d2[i]):
            sum += list_d2[i][j]**2
        sum = sum / len(list_d2[i])
        energy.append(sum)
        #计算各帧过零率
        sum = 0
        for j in range(len(list_d2)):
            sum += abs(sgn(list_d2[i][j+1])-sgn(list_d2[i][j]))
        sum /= 2
        sum /= len(list_d2[i])
        Z.append(sum)
    features.append(sum(energy) / len(energy))
    features.append(max(energy))
    features.append(sum(Z) / len(Z))
    features.append(max(Z))
    #features = [这一帧的平均平均能量，这一帧的最大平均能量，这一帧的平均过零率，帧的最大过零率]
    return features

def hamming_dtw_knn_predict_from_saved_data(X_test, k, training_data_file='dtw_knn_training_data.pkl'):
    # 加载训练数据
    X_train, y_train = joblib.load(training_data_file)
    # 调用预测函数
    predictions, top3_classes = dtw_knn_predict(X_train, y_train, X_test, k)

    return predictions, top3_classes
def get_prediction(wavsignal, domain, algorithm, window):
    processed_signal, head_and_tail = signal_cut(wavsignal)
    wav_display(wavsignal, head_and_tail)

    frame_length=512
    max_length=0

    for i in range(17):
        for j in range(10):
            max_length=max(max_length,len(processed_signal[i][j]))
    hamming_window = np.hamming(frame_length)
    hanning_window = np.hanning(frame_length)
    framed_processed_signal_hamming = [[[np.zeros(frame_length) for _ in range(math.ceil(max_length/frame_length))] for _ in range(17)] for _ in range(10)]
    framed_processed_signal_hanning = [[[np.zeros(frame_length) for _ in range(math.ceil(max_length/frame_length))] for _ in range(17)] for _ in range(10)]
    for k in range(math.ceil(max_length/frame_length)):
        start_index = k * frame_length
        end_index = start_index + frame_length
        frame = processed_signal[i][j][start_index:end_index]

        # 如果帧长度不够，使用零填充到完整的帧长度
        if len(frame) < frame_length:
            frame = np.pad(frame, (0, frame_length - len(frame)), 'constant')
        framed_processed_signal_hamming[j][i][k] = hamming_window * frame
        framed_processed_signal_hanning[j][i][k] = hanning_window * frame
    


    X_test_hamming = extract_features(framed_processed_signal_hamming)
    X_test_hanning = extract_features(framed_processed_signal_hanning)

    if domain == 'time':
        if algorithm == '1':
            prediction = bayesian_model.predict(X_test)
        if algorithm == '2':
            prediction = fisher_model.predict(X_test)
        if algorithm == '3':
            prediction = SVM_model.predict(X_test)
        if algorithm == '4':
            prediction = decision_tree_model.predict(X_test)
    elif domain == 'frequency':
        if window == 'hamming':
           prediction = 
        if window == 'hanning':
           prediction = 
        else:
            print('invalid window')
    
    else:
        print('invalid domain')

    return prediction

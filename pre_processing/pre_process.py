import os
import numpy as np
import soundfile as sf
import math
import json

from signal_split import signal_split



all_processed = [[[] for _ in range(10)] for _ in range(17)]


for i in range(0,17):

    # 构建数据文件的相对路径
    data_file_path = os.path.join(os.path.dirname(__file__), '..', 'dataset','original', f'original_{i+1}.wav')
    # 确保路径是绝对路径
    absolute_data_file_path = os.path.abspath(data_file_path)

    wavsignal,rt= sf.read(absolute_data_file_path)
    all_processed[i]=signal_split( wavsignal )
    print(f'第{i+1}个文件已经成功转化。')

'''
在拿到all_processed数组后，还需要对其中的数字进行归一化。
这个归一化应该是要消除各个人对同一个数字的大小差异。所以只需要把一个人说话的音强控制到[-1,+1]即可
'''
#遍历每个人
max_amplitude=0
for i in range(17):  # 遍历每个说话者
    for j in range(10):
        for k in range(len(all_processed[i][j])):
            max_amplitude= max(all_processed[i][j][k], max_amplitude)
    # 归一化该说话人的所有数字
    for j in range(10):
        for k in range(len(all_processed[i][j])):
            all_processed[i][j][k]/=max_amplitude


'''现在的all_processed是一个三维数组，第一维是说话者，第二维是数字，第三维是数字的音频数据。需要统计最大的音频长度，然后将所有音频数据补齐到这个长度。
所以要设置一个数组，这个数组是10x17x「(max_length/frame_length)向上取整」xframe_length'''
max_length=0
for i in range(17):
    for j in range(10):
        max_length=max(max_length,len(all_processed[i][j]))

frame_length=512

# 创建汉明窗和海宁窗
hamming_window = np.hamming(frame_length)
hanning_window = np.hanning(frame_length)

# 初始化列表
framed_processed_signal = [[[np.zeros(frame_length) for _ in range(math.ceil(max_length/frame_length))] for _ in range(17)] for _ in range(10)]
framed_processed_signal_hamming = [[[np.zeros(frame_length) for _ in range(math.ceil(max_length/frame_length))] for _ in range(17)] for _ in range(10)]
framed_processed_signal_hanning = [[[np.zeros(frame_length) for _ in range(math.ceil(max_length/frame_length))] for _ in range(17)] for _ in range(10)]

for i in range(17):
    for j in range(10):
        for k in range(math.ceil(max_length/frame_length)):
            start_index = k * frame_length
            end_index = start_index + frame_length
            frame = all_processed[i][j][start_index:end_index]

            # 如果帧长度不够，使用零填充到完整的帧长度
            if len(frame) < frame_length:
                frame = np.pad(frame, (0, frame_length - len(frame)), 'constant')

            framed_processed_signal[j][i][k] = frame
            framed_processed_signal_hamming[j][i][k] = hamming_window * frame
            framed_processed_signal_hanning[j][i][k] = hanning_window * frame

# 将 NumPy 数组转换为普通列表
def numpy_to_list(data):
    if isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, list):
        return [numpy_to_list(item) for item in data]
    else:
        return data

framed_processed_signal_list = numpy_to_list(framed_processed_signal)
framed_processed_signal_hamming_list = numpy_to_list(framed_processed_signal_hamming)
framed_processed_signal_hanning_list = numpy_to_list(framed_processed_signal_hanning)


# 现在 framed_processed_signal_hamming_list 是一个普通的嵌套列表，可以被序列化为 JSON
with open('data.json', 'w') as file:
    json.dump(framed_processed_signal_list, file)
with open('data_hamming.json', 'w') as file:
    json.dump(framed_processed_signal_hamming_list, file)
with open('data_hanning.json', 'w') as file:
    json.dump(framed_processed_signal_hanning_list, file)
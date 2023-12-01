# -----------------------------------------
# @Description: This file uses DTW method to recognize speech number signal.
# @Author: Yiwei Ren.
# @Date: 十一月 01, 2023, 星期三 14:42:25
# @Copyright (c) 2023 Yiwei Ren. All rights reserved.
# -----------------------------------------


import librosa
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
import numpy as np

class DTW():
    '''
        Class to dtw recognization
    '''

    def __init__(self, pattern_root, default_pattern = 0) -> None:
        '''
            Arguments:
                pattern_root: root path of patterns
                default_pattern: the index of pattern. Default=0
        '''
        self.pattern_root = pattern_root
        self.default_pattern = default_pattern
        self.fft_size = 512
        self.hop_length = 256
        self.coefficient = 0.97
        self.window = 'boxcar'
        self.n_mels = 128
        self.n_mfcc = 24

    def extract_mfcc(self, audio_path):
        y, sr = librosa.load(audio_path)
        y = np.append(y[0], y[1:] - self.coefficient * y[:-1])
        scale = max(y.max(), y.min())
        y /= scale
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=self.fft_size, hop_length=self.hop_length, window=self.window, n_mels=self.n_mels)
        mfccs = librosa.feature.mfcc(S=librosa.power_to_db(mel), n_mfcc=self.n_mfcc)
        return mfccs.T

    def dtw_distance(self, template, test):
        distance, _ = fastdtw(template, test, dist=euclidean)
        return distance

    def __call__(self, test_file):
        '''
            Arguments:
                test_file: the .wav file need to be DTW recognized
        '''
        import os
        test_mfcc = self.extract_mfcc(test_file)
        for num in range(10):
            pattern_file = os.path.join(self.pattern_root, f'{num}_{self.default_pattern}.wav')
            pattern_mfcc = self.extract_mfcc(pattern_file)
            dis = self.dtw_distance(pattern_mfcc, test_mfcc)
            if num == 0:
                predict = 0
                mindis = dis
            else:
                if dis < mindis:
                    mindis = dis
                    predict = num
        return predict

if __name__ == '__main__':
    classifier = DTW('LiAudio')
    classifier('RenAudio/5_9.wav')


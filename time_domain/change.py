import os
import pandas as pd
import numpy as np
import math
import json

data_hamming_path = os.path.join(os.path.dirname(__file__), '..', 'dataset','pre_processed', 'data_hamming.json')
absolute_hamming_path = os.path.abspath(data_hamming_path)

data_hanning_path = os.path.join(os.path.dirname(__file__), '..', 'dataset','pre_processed', 'data_hanning.json')
absolute_hanning_path = os.path.abspath(data_hanning_path)

with open(absolute_hamming_path, 'r') as file:
    data_hamming = json.load(file)
with open(absolute_hanning_path, 'r') as file:
    data_hanning = json.load(file)

'''data_hamming和data_hanning是嵌套列表'''
"""
数据预处理模块 OK

1. 读取数据
2. 按id进行划分
3. 归一化

created by qwk on December 18, 2018
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import minmax_scale


def get_time_series_from_dataframe(dataframe):
    """
    将 数据 按照 KPI ID 进行划分。
    将各个 ID 的所有value存入一个list
    将各个 ID 的所有label存入一个list
    :param dataframe: [timestamp, value, label, KPI ID]
    :return: [ [k1_v1,k1_v2,...,k1_v] , [k2_v1,k2_v2,...,k2_v] ... [km_v1,km_v2,...,km_v] ]
            [ [k1_label1,k1_label2,...,k1_label] , [k2_label1,k2_label2,...,k2_label] ... [km_label1,km_label2,...,km_label] ]
    """
    # training time series extraction
    ts_ids, ts_indexes, ts_point_counts = np.unique(dataframe['KPI ID'],return_index=True, return_counts=True)
    print('Extract are %d time series in the dataframe:' % (len(ts_ids)))

    # extract time series using ts_indexes
    ts_indexes.sort()
    ts_indexes = np.append(ts_indexes, len(dataframe))  # full ranges for extracting time series

    set_of_time_series = []
    set_of_time_series_label = []

    for i in np.arange(len(ts_indexes) - 1):
        print('Extracting %d th time series with index %d and %d (exclusive)'
              % (i, ts_indexes[i], ts_indexes[i + 1]))
        set_of_time_series.append(np.asarray(dataframe['value']
                                             [ts_indexes[i]:ts_indexes[i + 1]]))
        set_of_time_series_label.append(np.asarray(dataframe['label']
                                                   [ts_indexes[i]:ts_indexes[i + 1]]))

    return set_of_time_series, set_of_time_series_label


def preprocess_data(data_initial):
    """
    针对各个 id 的所有 Value 组成的 list 进行归一化
    :param data_initial: data
    :return: [ [k1_v1,k1_v2,...,k1_v] , [k2_v1,k2_v2,...,k2_v] ... [km_v1,km_v2,...,km_v] ]
    """

    # 按照 id 进行划分排序
    dataset, dataset_label = get_time_series_from_dataframe(data_initial)
    # 归一化，针对每个id
    dataset_size = len(dataset)

    dataset_scaled = []
    for i in np.arange(dataset_size):
        dataset_scaled.append(minmax_scale(dataset[i]))

    return dataset_scaled, dataset_label

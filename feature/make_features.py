"""
制作特征模块 OK

1. 特征提取方案：
- t时刻拟合值与实际值的对比特征
- t和t-1的对比特征
- 滑动窗口统计特征：均值、方差和分位数
- 滑动窗口 + 对比
特征集：多重窗口 * 统计特征 * 对比特征

2. 过采样

Created by qwk on December 18, 2018
"""

from statsmodels.tsa.api import SARIMAX, ExponentialSmoothing, SimpleExpSmoothing, Holt
import numpy as np
from sklearn.preprocessing import scale


# 特征提取方法之一
def get_feature_logs(time_series):
    return np.log(time_series + 1e-2)


# 特征提取方法之一
def get_feature_SARIMA_residuals(time_series):
    predict = SARIMAX(time_series,
                      trend='n').fit().get_prediction()
    return time_series - predict.predicted_mean


# 特征提取方法之一
def get_feature_AddES_residuals(time_series):
    predict = ExponentialSmoothing(time_series, trend='add').fit(smoothing_level=1)
    return time_series - predict.fittedvalues


# 特征提取方法之一
def get_feature_SimpleES_residuals(time_series):
    predict = SimpleExpSmoothing(time_series).fit(smoothing_level=1)
    return time_series - predict.fittedvalues


# 特征提取方法之一
def get_feature_Holt_residuals(time_series):
    predict = Holt(time_series).fit(smoothing_level=1)
    return time_series - predict.fittedvalues


# 传入的是一个指标的时间序列，返回的是这个指标的针对每个时刻的特征数组组合而成的list
def get_features_and_labels_from_a_time_series(time_series, time_series_label, Windows, delay):
    """
    Input: time_series, time_series_label, Window, delay (for determining vital data)

    In a time series dataset, it maintains a list of values.
    We'll convert the list of values into a list of feature vectors,
    each feature vector corresponds to a time point in the time series.

    For example: a time series [1,2,3,4,5] --> a featured dataset [[1,2,3],[2,3,4],[3,4,5]] (use one window size 3)

    The labels for the feature vectors are remained and returned.

    time_series: a list of values, an array
    time_series_label: a list of labels, an array
    Windows: the window sizes for time series feature extraction, an array
    delay: the maximum delay for effectively detect an anomaly

    Output: features_for_the_timeseries (a list of arrays),
            labels_for_the_timeseries (a list of arrays),
            vital_labels_for_the_timeseries (a list of arrays)
    """
    data = []
    data_label = []
    data_label_vital = []

    start_point = 2 * max(Windows)
    start_accum = 0

    # features from tsa models
    time_series_SARIMA_residuals = get_feature_SARIMA_residuals(time_series)
    time_series_AddES_residuals = get_feature_AddES_residuals(time_series)
    time_series_SimpleES_residuals = get_feature_SimpleES_residuals(time_series)
    time_Series_Holt_residuals = get_feature_Holt_residuals(time_series)

    # features from tsa models for time series logarithm
    time_series_logs = get_feature_logs(time_series)

    # 针对每一个时刻，得到一系列特征（ 9 + 12 * n ）
    for i in np.arange(start_point, len(time_series)):
        # the datum to put into the data pool
        datum = []
        datum_label = time_series_label[i]

        # fill the datum with f01-f09
        diff_plain = time_series[i] - time_series[i - 1]
        start_accum = start_accum + time_series[i]
        mean_accum = (start_accum) / (i - start_point + 1)

        # f01-f04: residuals
        datum.append(time_series_SARIMA_residuals[i])
        datum.append(time_series_AddES_residuals[i])
        datum.append(time_series_SimpleES_residuals[i])
        datum.append(time_Series_Holt_residuals[i])
        # f05: logarithm
        datum.append(time_series_logs[i])

        # f06: diff
        datum.append(diff_plain)
        # f07: diff percentage
        datum.append(diff_plain / (time_series[i - 1] + 1e-10))  # to avoid 0, plus 1e-10
        # f08: diff of diff - derivative
        datum.append(diff_plain - (time_series[i - 1] - time_series[i - 2]))
        # f09: diff of accumulated mean and current value
        datum.append(time_series[i] - mean_accum)

        # fill the datum with features related to windows
        # loop over different windows size to fill the datum
        for k in Windows:
            mean_w = np.mean(time_series[i - k:i + 1])
            var_w = np.mean((np.asarray(time_series[i - k:i + 1]) - mean_w) ** 2)
            # var_w = np.var(time_series[i-k:i+1])

            mean_w_and_1 = mean_w + (time_series[i - k - 1] - time_series[i]) / (k + 1)
            var_w_and_1 = np.mean((np.asarray(time_series[i - k - 1:i]) - mean_w_and_1) ** 2)
            # mean_w_and_1 = np.mean(time_series[i-k-1:i])
            # var_w_and_1 = np.var(time_series[i-k-1:i])

            mean_2w = np.mean(time_series[i - 2 * k:i - k + 1])
            var_2w = np.mean((np.asarray(time_series[i - 2 * k:i - k + 1]) - mean_2w) ** 2)
            # var_2w = np.var(time_series[i-2*k:i-k+1])

            # diff of sliding windows
            diff_mean_1 = mean_w - mean_w_and_1
            diff_var_1 = var_w - var_w_and_1

            # diff of jumping windows
            diff_mean_w = mean_w - mean_2w
            diff_var_w = var_w - var_2w

            # f1
            datum.append(mean_w)  # [0:2] is [0,1]
            # f2
            datum.append(var_w)
            # f3
            datum.append(diff_mean_1)
            # f4
            datum.append(diff_mean_1 / (mean_w_and_1 + 1e-10))
            # f5
            datum.append(diff_var_1)
            # f6
            datum.append(diff_var_1 / (var_w_and_1 + 1e-10))
            # f7
            datum.append(diff_mean_w)
            # f8
            datum.append(diff_mean_w / (mean_2w + 1e-10))
            # f9
            datum.append(diff_var_w)
            # f10
            datum.append(diff_var_w / (var_2w + 1e-10))

            # diff of sliding/jumping windows and current value
            # f11
            datum.append(time_series[i] - mean_w_and_1)
            # f12
            datum.append(time_series[i] - mean_2w)

        data.append(np.asarray(datum))  # 将此时刻产生的特征转换为 ndarray ，放入data列表中
        data_label.append(np.asarray(datum_label))

        # an important step is to identify the start anomalous points which are said to be critical
        # if the anomaly is detected within delay window of the occurence of the first anomaly
        if datum_label == 1 and sum(time_series_label[i - delay:i]) < delay:
            # 如果成立，说明是在时延内首次出现异常
            data_label_vital.append(np.asarray(1))
        else:
            data_label_vital.append(np.asarray(0))

    return data, data_label, data_label_vital


# 过采样
def get_expanded_featuers_and_labels(data_pool, data_pool_label, data_pool_label_vital, oversample=0):
    assert (len(data_pool) == len(data_pool_label) == len(data_pool_label_vital))

    if oversample == 0:
        return data_pool, data_pool_label

    data_pool_len = len(data_pool)

    # the data points and labels to be appended into the data/label pool
    data_pool_plus = []
    data_pool_plus_label = []
    for i in np.arange(data_pool_len):
        if data_pool_label[i] == 1:  # anomalous point
            data_pool_plus.append(data_pool[i])
            data_pool_plus_label.append(data_pool_label[i])

    # the data points and labels to be appended into the data/label pool (critical ones)
    data_pool_vital = []
    data_pool_vital_label = []
    for i in np.arange(data_pool_len):
        if data_pool_label_vital[i] == 1:  # vital anomalous point
            data_pool_vital.append(data_pool[i])
            data_pool_vital_label.append(data_pool_label_vital[i])

    # oversample abnormal data instances and vital abnormal data instances to balance the dataset
    data_pool_complete = data_pool + \
                         oversample * data_pool_plus + \
                         oversample * data_pool_vital
    # list可以直接 + 和 * ，相当于组合到一起
    # a = [[1,2,3],[5,6,7]]   b = [[11,22,33]]    c = a + 2* b  ====>  [[1, 2, 3], [5, 6, 7], [11, 22, 33], [11, 22, 33]]
    data_pool_complete_label = data_pool_label + \
                               oversample * data_pool_plus_label + \
                               oversample * data_pool_vital_label

    assert (len(data_pool_complete) == len(data_pool_complete_label))
    print('The augment size of the dataset: %d = %d + %d * %d + %d * %d' % (len(data_pool_complete),
                                                                            len(data_pool),
                                                                            oversample,
                                                                            len(data_pool_plus),
                                                                            oversample,
                                                                            len(data_pool_vital)))

    # data_pool_complete (X) and data_pool_complete_label (y) should be ready for training
    return data_pool_complete, data_pool_complete_label


# 对外提供调用方法
def features_service(train_time_series_dataset_scaled, train_time_series_dataset_label):
    train_time_series_dataset_size = len(train_time_series_dataset_scaled)
    # 1) feature engineering for training dataset
    # specify the set of window sizes
    # the maximum number is 125 means the start point to consider anomalies is 250, i.e., max(2W).
    W = np.asarray([2, 5, 10, 25, 50, 100])
    delay = 7

    # training: data pool for labeled data points (presented by 6n+2 features)
    train_data_pool = []
    train_data_pool_label = []
    train_data_pool_label_vital = []

    # loop over all the time series
    for i in np.arange(train_time_series_dataset_size):
        # loop over all the data points in each time series
        data, \
        data_label, \
        data_label_vital = get_features_and_labels_from_a_time_series(train_time_series_dataset_scaled[i],
                                                                      train_time_series_dataset_label[i],
                                                                      W, delay)
        train_data_pool = train_data_pool + list(scale(np.asarray(data)))
        # train_data_pool = train_data_pool + list(minmax_scale(abs(np.asarray(data))))
        # train_data_pool = train_data_pool + list(maxabs_scale(np.asarray(data)))

        train_data_pool_label = train_data_pool_label + data_label
        train_data_pool_label_vital = train_data_pool_label_vital + data_label_vital

    # 2) over sampling
    # the methodology to achieve over sampling is to pick samples from train_data according to train_data_label
    # data_pool + data_pool_plus + data_pool_vital, there are three datasets to be merged
    # data_pool_label + data_pool_plus_label + data_pool_vital_label, there are three label datasets to be merged
    train_data_pool_complete, \
    train_data_pool_complete_label = get_expanded_featuers_and_labels(train_data_pool,
                                                                      train_data_pool_label,
                                                                      train_data_pool_label_vital,
                                                                      1)

    # now oversampling is set to 1, use oversampling currenlty .
    # default oversampling is set to 0, do not use oversampling .
    return train_data_pool_complete,train_data_pool_complete_label

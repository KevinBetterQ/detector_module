"""
训练模型选择 接口层

用于根据参数进行模型选择，然后训练模型

Created by qwk on January 06, 2019
"""
from algorithm import xgboosting


def model_train(model_name, data_features_arr, data_features_label_arr):
    """
    用于给训练部分调用，根据模型名称选择模型
    :param model_name: 目前仅有 xgb 、 lg
    :param data_features_arr: 样本特征集
    :param data_features_label_arr: 样本标签集
    """
    if model_name == 'xgb':
        xgboosting.model_train(data_features_arr, data_features_label_arr)
    elif model_name == 'lg':
        print("none")
    else:
        print("error")


"""
模型训练部分

1. 读取数据
2. 预处理
3. 特征制作
4. 模型训练
5. 保存模型

Created by qwk on December 19, 2018
"""


import pandas as pd
import numpy as np
from detector_module.feature import make_features
from detector_module.preprocess import data_preprocess
from sys import argv
from detector_module.algorithm import model_select

MODEL_DEFAULT = "xgb"
DATA_PATH_DEFAULT = "./sample/train_sameid.csv"

if __name__ == "__main__":
    # 设置数据地址 和 选择模型
    if len(argv) == 1:
        data_path = DATA_PATH_DEFAULT
        model = MODEL_DEFAULT
    else:
        data_path = argv[1]
        model = argv[2]

    data = pd.read_csv(data_path)
    data_preprocess, data_preprocess_label = data_preprocess.preprocess_data(data)
    data_features, data_features_label = make_features.features_service(data_preprocess, data_preprocess_label)

    data_features_arr = np.array(data_features)
    data_features_label_arr = np.array(data_features_label)

    model_select.model_train(model, data_features_arr, data_features_label_arr)

    print(" train success !")




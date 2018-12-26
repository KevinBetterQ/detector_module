
"""
异常检测模块

1. 读取数据 + 预处理
2. 调用特征模块，制作特征 ===> 得到特征数据集
3. 加载模型，进行异常检测

created by qwk on December 17, 2018
"""

import pandas as pd
import numpy as np
from sklearn.externals import joblib
import os
from detector_module.feature import make_features
from detector_module.preprocess import data_preprocess
from sys import argv
MODEL_PATH = os.path.join(os.path.dirname(__file__), './model/')


model_name = MODEL_PATH + "xgb_model_835"
data = pd.read_csv('./sample/train_data.csv')
data['label'] = 0
data_preprocess, data_preprocess_label = data_preprocess.preprocess_data(data)
data_features, data_features_label = make_features.features_service(data_preprocess, data_preprocess_label)
model = joblib.load(model_name)

data_features_arr = np.array(data_features)
data_features_label_arr = np.array(data_features_label)

y_pred = model.predict(data_features_arr)
y_temp = y_pred[0:200]
y_temp[:] = 0
y_pred = np.append(y_temp, y_pred)
result = data[['timestamp', 'value', 'label']]
result['label'] = y_pred
result = result.loc[result['label'] == 1]
result = result.drop(['label'], axis=1)
result.to_csv('./output/result.csv', index=False)
print("success")

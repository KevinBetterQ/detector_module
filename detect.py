
"""
异常检测模块

1. 读取数据 + 预处理
2. 调用特征模块，制作特征 ===> 得到特征数据集
3. 加载默认模型，进行异常检测

created by qwk on December 17, 2018
"""

import pandas as pd
import numpy as np
from sklearn.externals import joblib
import os
from feature import make_features
from preprocess import data_preprocess
from sys import argv
import matplotlib.pyplot as plt
from matplotlib.dates import AutoDateLocator, DateFormatter

MODEL_PATH = os.path.join(os.path.dirname(__file__), './model/')
DATA_PATH_DEFAULT = "./sample/train_data.csv"


if __name__ == "__main__":
    """
    运行程序有2个参数：
    @:param data_path 要检测数据文件的路径，绝对路径
    @:param model_name 提供可选择模型，输入模型名称
    """

    # 1、参数获取：数据文件地址 和 模型名称
    if len(argv) == 1:
        data_path = DATA_PATH_DEFAULT
        model_name = MODEL_PATH + "xgb"
    else:
        data_path = argv[1]
        model_name = MODEL_PATH + argv[2]

    # 2、读取文件
    data = pd.read_csv(data_path)
    data = data[0:10000]
    data['label'] = 0
    # 3、数据预处理 + 特征提取
    data_preprocess, data_preprocess_label = data_preprocess.preprocess_data(data)
    data_features, data_features_label = make_features.features_service(data_preprocess, data_preprocess_label)
    # 4、加载模型
    model = joblib.load(model_name)
    # 5、检测
    data_features_arr = np.array(data_features)  # 将特征格式转化为 narray 格式，以便带入模型
    data_features_label_arr = np.array(data_features_label)
    y_pred = model.predict(data_features_arr)  # 异常检测
    # 6、检测结果处理
    # 由于窗口设置问题，前200个数据是没有检测结果的，所以统一设为 0，补充在检测结果中
    y_temp = y_pred[0:200]
    y_temp[:] = 0
    y_pred = np.append(y_temp, y_pred)
    # 仅输出异常的时间数据，所以对检测结果进行筛选导成文件输出
    result = data[['timestamp', 'value', 'label']]
    result['label'] = y_pred
    res_ab = result.loc[result['label'] == 1]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(result['timestamp'], result['value'])
    plt.scatter(res_ab['timestamp'], res_ab['value'], color='red')
    plt.xlabel('timestamp')
    plt.ylabel('value')

    imgName = 'res_xgb.png'
    imgPath = './output/'
    plt.savefig(imgPath + imgName)
    plt.show()

    result.to_csv("./output/result.csv", index=0)

    print(imgPath + imgName + "successful")



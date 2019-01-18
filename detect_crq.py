"""
twitter  无监督方式异常检测

1. 读取数据
2. 预处理
3. 模型检测
4. 保存结果

Created by crq on December 24, 2018
"""


import pandas as pd
import numpy as np
from sklearn.externals import joblib
import os
from feature import make_features
from preprocess import data_preprocess
from sys import argv
from algorithm.twitter.pyculiarity import detect_ts
from algorithm.twitter import evaluate
import datetime
import time
'''
from detector_module.feature import make_features
from detector_module.preprocess import data_preprocess
from sys import argv
'''


#MODEL_DEFAULT = "xgboosting"
DATA_PATH_DEFAULT = "./sample/train_sameid.csv"

# 设置数据地址 和 选择模型
if len(argv) == 1:
    data_path = DATA_PATH_DEFAULT
    #model = MODEL_DEFAULT
else:
    data_path = argv[1]
    #model = argv[2]


data = pd.read_csv(data_path)
# 返回格式为list
data_preprocess, data_preprocess_label = data_preprocess.preprocess_data(data)
print(data_preprocess,data_preprocess_label)
data_detect=pd.DataFrame({'value':data_preprocess[0],'label':data_preprocess_label[0]})
# 如果timestamp是时间戳格式
print(data['timestamp'].dtype,np.int64)

# 如果时间为时间戳格式  转化为时间字符串格式
if (np.issubdtype(data['timestamp'][0],np.int64)):

    data_detect['timestamp']=data['timestamp'].apply(lambda timestamp:datetime.datetime.utcfromtimestamp(timestamp),"%Y-%m-%d %H:%M:%S")
else:
    # 如果是 字符串格式  就直接赋值
    data_detect['timestamp']=data['timestamp']

print(data_detect.head(2))
print(data.dtypes[0])
twitter_example_data = data_detect[['timestamp', 'value']]
label = pd.DataFrame(data_detect['timestamp'])
label['label']=data['label']
print(type(label))

results = detect_ts(twitter_example_data, max_anoms=0.05, alpha=0.001, direction='both', only_last=None,
                    longterm=True)

# format the twitter data nicely
twitter_example_data['timestamp'] = pd.to_datetime(twitter_example_data['timestamp'])
twitter_example_data.set_index('timestamp', drop=True)
# print(results['anoms'].index, results['anoms']['anoms'])
print(results['anoms'])
print(type(results['anoms']))

results['anoms'].reset_index(drop=True,inplace=True)  # 取消其时间索引列
df_anoms=pd.DataFrame(results['anoms'])
print(df_anoms['timestamp'].head(2))

# 模型评估
evaluate.evaluate(label, df_anoms)

# 如果数据本身时间是时间戳格式  转化回去
if (np.issubdtype(data['timestamp'][0],np.int64)):
    df_anoms['timestamp']=df_anoms['timestamp'].apply(lambda datetime_t:int(time.mktime(datetime_t.timetuple())))

'''
# 更改列名
if list(df_anoms.columns.values)!=['timestamp','anoms']:
        df_anoms.columns=['timestamp','anoms']
#print(df_anoms['timestamp'].head(2))
'''
# 保存至表格
df_anoms.to_csv('./output/result_twitter.csv', index=False)




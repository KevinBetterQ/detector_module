"""
测试使用
"""


# 分解训练集
def make_data():
    import pandas as pd
    data = pd.read_csv('sample/train_sameid.csv')
    train_label = data['label']
    train_data = data.drop(['label'], axis=1)
    train_data.to_csv('sample/train_data.csv', index=False)
    train_label.to_csv('sample/train_label.csv', index=False)

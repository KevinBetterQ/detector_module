import pandas as pd

# 传入原始值 和检测出的异常值

def evaluate(origin,returnData):

    if list(origin.columns.values)!=['timestamp','label']:
        origin.columns=['timestamp','label']
    positive=origin[origin['label']==0] # 原始正样本
    negetive=origin[origin['label']==1] #原始负样本
    '''
    returnData.reset_index(drop=True,inplace=True)  # 取消其时间索引列
    returnData=pd.DataFrame(returnData)
    '''
    #returnData['timestamp'] = returnData['timestamp'].astype('object')

    # 正确分类的负样本 真负例
    TN=len(negetive.merge(returnData,on='timestamp',how='inner'))

    # 错误分类的负样本
    noCorrect = pd.merge(negetive, returnData, how='left', indicator=True).query(
        "_merge=='left_only'").drop('_merge', 1)

    # 错误分类的负样本 假负例
    FP = len(negetive)-TN
    #FP=len(noCorrect)

    # 错误分类的正样本 假正例
    FN=len(positive.merge(returnData,on='timestamp',how='inner'))

    # 正确分类的正样本 真正例
    TP=len(positive)-FN

    # 分类正确的样本数/所有样本数  准确率
    accurancy=(TN+TP)/len(origin['label'])

    # 基于混淆矩阵
    #分类正确的正样本/所有预测为正的样本 精确率
    precision=TP/(TP+FP)

    # 分类正确的正样本/所有预测为正的样本数 召回率
    recall = TP / (TP + FN)

    #f1_score F1 值
    f1_score=(2*precision*recall)/(precision+recall)

    #print(accurancy,precision,recall,f1_score)

    print("模型评估：")
    print('Accuracy:', accurancy)
    print('F1 score:', f1_score)
    print('Recall:', recall)
    print('Precision:', precision)

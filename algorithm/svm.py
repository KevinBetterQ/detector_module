"""
训练svm模型

1. 划分数据集
2. 训练
3 评估
4. 返回

Created by qwk on Mar 31,2019
"""

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,\
    f1_score,log_loss,recall_score,\
    precision_score,roc_auc_score,classification_report
from sklearn.svm import SVR,SVC
import time
from sklearn.externals import joblib


def model_train(xtrain, ytrain):
    X_train, X_test, y_train, y_test = train_test_split(xtrain, ytrain, test_size=0.2, random_state=0)
    cls=SVC(probability=True,kernel='rbf',C=0.1,max_iter=10)
    start_time = time.time()
    cls.fit(X_train, y_train)
    end_time = time.time()
    print('It took %d seconds to train the model!' % (end_time - start_time))
    print()
    y_pred = cls.predict(X_test)
    print("模型及模型参数：")
    print(str(cls))
    print("模型评估：")
    print('Accuracy:', accuracy_score(y_test, y_pred))
    print('F1 score:', f1_score(y_test, y_pred))
    print('Recall:', recall_score(y_test, y_pred))
    print('Precision:', precision_score(y_test, y_pred))
    print('\n clasification report:\n', classification_report(y_test, y_pred))
    print('\n confussion matrix:\n', confusion_matrix(y_test, y_pred))

    # 保存模型
    model_name = "./model/" + "svm"
    joblib.dump(cls, model_name)


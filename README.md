# detector_module 
## ————时间序列异常检测系统的算法模块

## Done
- [x] 数据预处理完成
- [x] 特征提取完成
- [x] 简单跑一个算法 XGBoost
- [x] 训练接口 train 完成初步
- [x] 检测接口 detect 完成初步
- [x] 检测结果输出格式调整：time + value

## TODO
- 训练：
- [x] 制作特征后，不要列名，直接训练
- [ ] 设置一个训练模型的选择接口，代理分发

- 预测：
- [ ] 设置一个训练模型的选择接口，代理分发
- [x] 换上默认模型

## 问题
1. 窗口设置太随意
2. 特征制作的时候固定列名
3. 如何评估数据量需要的训练时长？
4. 训练进入如何查看？

# ANN识别手写数字mnist数据集
#### ANN recognize handwritten number
#### Hello world of machine learning

- ANN_recog_num 是带中文注释的主要版本
- ANN_recog_num_gui 是使用dearpygui的GUI版本
- NN_recog_num 是无隐藏层的原始版本
- parameters_0.9785 是学习率0.6训练100个epoch的模型文件，使用测试数据集的正确率：97.85%

```py
learn_rate = 0.6 
epoch_num = 100
parameters = train(parameters,learn_rate,epoch_num)
print(accuracy(parameters,"test")) # 0.9785
```

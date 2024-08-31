任务一：基于机器学习的文本分类
实现基于logistic/softmax regression的文本分类

任务二：基于深度学习的文本分类
熟悉Pytorch，用Pytorch重写《任务一》，实现CNN、RNN的文本分类；

任务三：基于注意力机制的文本匹配
输入两个句子判断，判断它们之间的关系。参考ESIM（可以只用LSTM，忽略Tree-LSTM），用双向的注意力机制实现。

任务四：基于LSTM+CRF的序列标注
用LSTM+CRF来训练序列标注模型：以Named Entity Recognition为例。

任务五：基于神经网络的语言模型
用LSTM、GRU来训练字符级的语言模型，计算困惑度


pytorch:1.9.0
torch_geometric:2.5.3
torch_scatter:2.0.9
torch_sparse:0.6.12
torchvision:0.10.1

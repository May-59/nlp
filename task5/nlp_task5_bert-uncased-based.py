#!/usr/bin/env python
# coding: utf-8

# In[ ]:



#数据预处理

with open('poetryFromTang.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# 创建字符到索引的映射
chars = sorted(list(set(text)))
char2idx = {c: i for i, c in enumerate(chars)}
idx2char = {i: c for i, c in enumerate(chars)}

# 将文本转换为索引
text_as_int = [char2idx[c] for c in text]


# In[ ]:


seq_length = 100  # 你可以根据需要调整这个长度
examples_per_epoch = len(text) // seq_length

inputs = []
targets = []

for i in range(examples_per_epoch):
    start_idx = i * seq_length
    end_idx = start_idx + seq_length + 1
    inputs.append(text_as_int[start_idx:end_idx-1])
    targets.append(text_as_int[start_idx+1:end_idx])

import torch
from torch.utils.data import Dataset, DataLoader

class CharDataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return torch.tensor(self.inputs[idx], dtype=torch.long), torch.tensor(self.targets[idx], dtype=torch.long)

dataset = CharDataset(inputs, targets)
data_loader = DataLoader(dataset, batch_size=64, shuffle=True)


# In[ ]:


#定义模型
import torch.nn as nn

class CharModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, rnn_type='lstm'):
        super(CharModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        if rnn_type == 'lstm':
            self.rnn = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        elif rnn_type == 'gru':
            self.rnn = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embedding(x)
        output, hidden = self.rnn(x, hidden)
        output = self.fc(output)
        return output, hidden


# In[ ]:


# 训练模型
import torch.optim as optim

def train(model, data_loader, epochs, vocab_size, rnn_type='lstm'):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(epochs):
        hidden = None
        total_loss = 0
        for batch, (x, y) in enumerate(data_loader):
            optimizer.zero_grad()
            output, hidden = model(x, hidden)
            loss = criterion(output.view(-1, vocab_size), y.view(-1))
            #loss.backward()
            optimizer.step()
            total_loss += loss.item()

        perplexity = torch.exp(torch.tensor(total_loss / len(data_loader)))
        print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}, Perplexity: {perplexity:.4f}')

# 初始化模型并开始训练
vocab_size = len(chars)
embed_dim = 128
hidden_dim = 256
model = CharModel(vocab_size, embed_dim, hidden_dim, rnn_type='lstm')
train(model, data_loader, epochs=10, vocab_size=vocab_size)


# In[ ]:


from transformers import BertTokenizer, BertModel

# 加载BERT分词器和模型
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

class BertCharModel(nn.Module):
    def __init__(self, hidden_dim, rnn_type='lstm'):
        super(BertCharModel, self).__init__()
        self.bert = bert_model
        if rnn_type == 'lstm':
            self.rnn = nn.LSTM(self.bert.config.hidden_size, hidden_dim, batch_first=True)
        elif rnn_type == 'gru':
            self.rnn = nn.GRU(self.bert.config.hidden_size, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, self.bert.config.vocab_size)

    def forward(self, input_ids, attention_mask=None):
        with torch.no_grad():
            bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        output, hidden = self.rnn(bert_output.last_hidden_state)
        output = self.fc(output)
        return output, hidden

# 使用BERT的嵌入进行训练
model = BertCharModel(hidden_dim, rnn_type='lstm')
train(model, data_loader, epochs=10, vocab_size=vocab_size)


# In[ ]:


try:   
    get_ipython().system('jupyter nbconvert --to python nlp_task5_bert-uncased-based.ipynb')
    # python即转化为.py，script即转化为.html
    # file_name.ipynb即当前module的文件名
except:
    pass


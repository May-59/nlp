#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import random
import re
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

def data_split(data, test_rate=0.3):
    train = list()
    test = list()
    i = 0
    for datum in data:
      i += 1
      if random.random() > test_rate:
          train.append(datum)
      else:
          test.append(datum)
    return train, test


class Random_embedding():
    def __init__(self, data, test_rate=0.3):
        self.dict_words = dict()
        _data = [item.split('\t') for item in data]
        self.data = [[item[5], item[6], item[0]] for item in _data]
        self.data.sort(key=lambda x:len(x[0].split()))
        self.len_words = 0
        self.train, self.test = data_split(self.data, test_rate=test_rate)
        self.type_dict = {'-': 0, 'contradiction': 1, 'entailment': 2, 'neutral': 3}
        self.train_y = [self.type_dict[term[2]] for term in self.train]  # Relation in training set
        self.test_y = [self.type_dict[term[2]] for term in self.test]  # Relation in test set
        self.train_s1_matrix = list()
        self.test_s1_matrix = list()
        self.train_s2_matrix = list()
        self.test_s2_matrix = list()
        self.longest = 0

    def get_words(self):
        pattern = '[A-Za-z|\']+'
        for term in self.data:
            for i in range(2):
                s = term[i]
                s = s.upper()
                words = re.findall(pattern, s)
                for word in words:  # Process every word
                    if word not in self.dict_words:
                        self.dict_words[word] = len(self.dict_words)+1
        self.len_words = len(self.dict_words)

    def get_id(self):
        pattern = '[A-Za-z|\']+'
        for term in self.train:
            s = term[0]
            s = s.upper()
            words = re.findall(pattern, s)
            item = [self.dict_words[word] for word in words]
            self.longest = max(self.longest, len(item))
            self.train_s1_matrix.append(item)
            s = term[1]
            s = s.upper()
            words = re.findall(pattern, s)
            item = [self.dict_words[word] for word in words]
            self.longest = max(self.longest, len(item))
            self.train_s2_matrix.append(item)
        for term in self.test:
            s = term[0]
            s = s.upper()
            words = re.findall(pattern, s)
            item = [self.dict_words[word] for word in words]
            self.longest = max(self.longest, len(item))
            self.test_s1_matrix.append(item)
            s = term[1]
            s = s.upper()
            words = re.findall(pattern, s)
            item = [self.dict_words[word] for word in words]
            self.longest = max(self.longest, len(item))
            self.test_s2_matrix.append(item)
        self.len_words+=1


class Glove_embedding():
    def __init__(self, data, trained_dict, test_rate=0.3):
        self.dict_words = dict()
        _data = [item.split('\t') for item in data]
        self.data = [[item[5], item[6], item[0]] for item in _data]
        self.data.sort(key=lambda x:len(x[0].split()))
        self.trained_dict = trained_dict
        self.len_words = 0
        self.train, self.test = data_split(self.data, test_rate=test_rate)
        self.type_dict = {'-': 0, 'contradiction': 1, 'entailment': 2, 'neutral': 3}
        self.train_y = [self.type_dict[term[2]] for term in self.train]  # Relation in training set
        self.test_y = [self.type_dict[term[2]] for term in self.test]  # Relation in test set
        self.train_s1_matrix = list()
        self.test_s1_matrix = list()
        self.train_s2_matrix = list()
        self.test_s2_matrix = list()
        self.longest = 0
        self.embedding = list()

    def get_words(self):
        self.embedding.append([0]*50)
        pattern = '[A-Za-z|\']+'
        for term in self.data:
            for i in range(2):
                s = term[i]
                s = s.upper()
                words = re.findall(pattern, s)
                for word in words:  # Process every word
                    if word not in self.dict_words:
                        self.dict_words[word] = len(self.dict_words)
                        if word in self.trained_dict:
                            self.embedding.append(self.trained_dict[word])
                        else:
                            # print(word)
                            # raise Exception("words not found!")
                            self.embedding.append([0] * 50)
        self.len_words = len(self.dict_words)

    def get_id(self):
        pattern = '[A-Za-z|\']+'
        for term in self.train:
            s = term[0]
            s = s.upper()
            words = re.findall(pattern, s)
            item = [self.dict_words[word] for word in words]
            self.longest = max(self.longest, len(item))
            self.train_s1_matrix.append(item)
            s = term[1]
            s = s.upper()
            words = re.findall(pattern, s)
            item = [self.dict_words[word] for word in words]
            self.longest = max(self.longest, len(item))
            self.train_s2_matrix.append(item)
        for term in self.test:
            s = term[0]
            s = s.upper()
            words = re.findall(pattern, s)
            item = [self.dict_words[word] for word in words]
            self.longest = max(self.longest, len(item))
            self.test_s1_matrix.append(item)
            s = term[1]
            s = s.upper()
            words = re.findall(pattern, s)
            item = [self.dict_words[word] for word in words]
            self.longest = max(self.longest, len(item))
            self.test_s2_matrix.append(item)
        self.len_words+=1


class ClsDataset(Dataset):
    """ 文本分类数据集 """
    def __init__(self, sentence1,sentence2, relation):
        self.sentence1 = sentence1
        self.sentence2 = sentence2
        self.relation = relation

    def __getitem__(self, item):
        return self.sentence1[item], self.sentence2[item],self.relation[item]

    def __len__(self):
        return len(self.relation)


def collate_fn(batch_data):
    """ 自定义一个batch里面的数据的组织方式 """

    sents1,sents2, labels = zip(*batch_data)
    sentences1 = [torch.LongTensor(sent) for sent in sents1]
    padded_sents1 = pad_sequence(sentences1, batch_first=True, padding_value=0)
    sentences2 = [torch.LongTensor(sent) for sent in sents2]
    padded_sents2 = pad_sequence(sentences2, batch_first=True, padding_value=0)
    return torch.LongTensor(padded_sents1), torch.LongTensor(padded_sents2),  torch.LongTensor(labels)


def get_batch(x1,x2,y,batch_size):
    dataset = ClsDataset(x1,x2, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False,drop_last=True,collate_fn=collate_fn)
    return dataloader


# In[ ]:


import torch
import torch.nn as nn
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Input_Encoding(nn.Module):
    def __init__(self, len_feature, len_hidden, len_words,longest, weight=None, layer=1, batch_first=True, drop_out=0.5):
        super(Input_Encoding, self).__init__()
        self.len_feature = len_feature
        self.len_hidden = len_hidden
        self.len_words = len_words
        self.layer = layer
        self.longest=longest
        self.dropout = nn.Dropout(drop_out)
        if weight is None:
            x = nn.init.xavier_normal_(torch.Tensor(len_words, len_feature))
            self.embedding = nn.Embedding(num_embeddings=len_words, embedding_dim=len_feature, _weight=x).to(device)
        else:
            self.embedding = nn.Embedding(num_embeddings=len_words, embedding_dim=len_feature, _weight=weight).to(device)
        self.lstm = nn.LSTM(input_size=len_feature, hidden_size=len_hidden, num_layers=layer, batch_first=batch_first,
                            bidirectional=True).to(device)

    def forward(self, x):
        x = torch.LongTensor(x).to(device)
        x = self.embedding(x)
        x = self.dropout(x)
        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)
        return x


class Local_Inference_Modeling(nn.Module):
    def __init__(self):
        super(Local_Inference_Modeling, self).__init__()
        self.softmax_1 = nn.Softmax(dim=1).to(device)
        self.softmax_2 = nn.Softmax(dim=2).to(device)

    def forward(self, a_bar, b_bar):
        e = torch.matmul(a_bar, b_bar.transpose(1, 2)).to(device)

        a_tilde = self.softmax_2(e)
        a_tilde = a_tilde.bmm(b_bar)
        b_tilde = self.softmax_1(e)
        b_tilde = b_tilde.transpose(1, 2).bmm(a_bar)

        m_a = torch.cat([a_bar, a_tilde, a_bar - a_tilde, a_bar * a_tilde], dim=-1)
        m_b = torch.cat([b_bar, b_tilde, b_bar - b_tilde, b_bar * b_tilde], dim=-1)

        return m_a, m_b


class Inference_Composition(nn.Module):
    def __init__(self, len_feature, len_hidden_m, len_hidden, layer=1, batch_first=True, drop_out=0.5):
        super(Inference_Composition, self).__init__()
        self.linear = nn.Linear(len_hidden_m, len_feature).to(device)
        self.lstm = nn.LSTM(input_size=len_feature, hidden_size=len_hidden, num_layers=layer, batch_first=batch_first,
                            bidirectional=True).to(device)
        self.dropout = nn.Dropout(drop_out).to(device)

    def forward(self, x):
        x = self.linear(x)
        x = self.dropout(x)
        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)

        return x


class Prediction(nn.Module):
    def __init__(self, len_v, len_mid, type_num=4, drop_out=0.5):
        super(Prediction, self).__init__()
        self.mlp = nn.Sequential(nn.Dropout(drop_out), nn.Linear(len_v, len_mid), nn.Tanh(),
                                 nn.Linear(len_mid, type_num)).to(device)

    def forward(self, a,b):

        v_a_avg=a.sum(1)/a.shape[1]
        v_a_max = a.max(1)[0]

        v_b_avg = b.sum(1) / b.shape[1]
        v_b_max = b.max(1)[0]

        out_put = torch.cat((v_a_avg, v_a_max,v_b_avg,v_b_max), dim=-1)

        return self.mlp(out_put)


class ESIM(nn.Module):
    def __init__(self, len_feature, len_hidden, len_words,longest, type_num=4, weight=None, layer=1, batch_first=True,
                 drop_out=0.5):
        super(ESIM, self).__init__()
        self.len_words=len_words
        self.longest=longest
        self.input_encoding = Input_Encoding(len_feature, len_hidden, len_words,longest, weight=weight, layer=layer,
                                             batch_first=batch_first, drop_out=drop_out)
        self.local_inference_modeling = Local_Inference_Modeling()
        self.inference_composition = Inference_Composition(len_feature, 8 * len_hidden, len_hidden, layer=layer,
                                                           batch_first=batch_first, drop_out=drop_out)
        self.prediction=Prediction(len_hidden*8,len_hidden,type_num=type_num,drop_out=drop_out)

    def forward(self,a,b):
        a_bar=self.input_encoding(a)
        b_bar=self.input_encoding(b)

        m_a,m_b=self.local_inference_modeling(a_bar,b_bar)

        v_a=self.inference_composition(m_a)
        v_b=self.inference_composition(m_b)

        out_put=self.prediction(v_a,v_b)

        return out_put


# In[ ]:


import matplotlib.pyplot
import torch
import torch.nn.functional as F
# from feature_batch import get_batch
from torch import optim
# from Neural_Network_batch import ESIM
import random
import numpy


def NN_embdding(model, train, test, learning_rate, iter_times):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fun = F.cross_entropy
    train_loss_record = list()
    test_loss_record = list()
    train_record = list()
    test_record = list()
    # torch.autograd.set_detect_anomaly(True)

    for iteration in range(iter_times):
      torch.cuda.empty_cache()
      model.train()
      for i, batch in enumerate(train):
        torch.cuda.empty_cache()
        x1, x2, y = batch
        pred = model(x1, x2).to(device)
        optimizer.zero_grad()
        y=y.to(device)
        loss = loss_fun(pred, y).to(device)
        loss.backward()
        optimizer.step()
      with torch.no_grad():
        model.eval()
        train_acc = list()
        test_acc = list()
        train_loss = 0
        test_loss = 0
        for i, batch in enumerate(train):
          torch.cuda.empty_cache()
          x1, x2, y = batch
          y=y.to(device)
          pred = model(x1, x2).to(device)
          loss = loss_fun(pred, y).to(device)
          train_loss += loss.item()
          _, y_pre = torch.max(pred, -1)
          acc = torch.mean((torch.tensor(y_pre == y, dtype=torch.float)))
          train_acc.append(acc)

        for i, batch in enumerate(test):
          torch.cuda.empty_cache()
          x1, x2, y = batch
          y=y.to(device)
          pred = model(x1, x2).to(device)
          loss = loss_fun(pred, y).to(device)
          test_loss += loss.item()
          _, y_pre = torch.max(pred, -1)
          acc = torch.mean((torch.tensor(y_pre == y, dtype=torch.float)))
          test_acc.append(acc)

      trains_acc = sum(train_acc) / len(train_acc)
      tests_acc = sum(test_acc) / len(test_acc)

      train_loss_record.append(train_loss / len(train_acc))
      test_loss_record.append(test_loss/ len(test_acc))
      train_record.append(trains_acc.cpu())
      test_record.append(tests_acc.cpu())
      print("---------- Iteration", iteration + 1, "----------")
      print("Train loss:", train_loss/ len(train_acc))
      print("Test loss:", test_loss/ len(test_acc))
      print("Train accuracy:", trains_acc)
      print("Test accuracy:", tests_acc)

    return train_loss_record, test_loss_record, train_record, test_record


def NN_plot(random_embedding, glove_embedding, len_feature, len_hidden, learning_rate, batch_size, iter_times):
    train_random = get_batch(random_embedding.train_s1_matrix, random_embedding.train_s2_matrix,
                             random_embedding.train_y,batch_size)
    test_random = get_batch(random_embedding.test_s1_matrix, random_embedding.test_s2_matrix,
                            random_embedding.test_y,batch_size)
    train_glove = get_batch(glove_embedding.train_s1_matrix, glove_embedding.train_s2_matrix,
                            glove_embedding.train_y,batch_size)
    test_glove = get_batch(glove_embedding.test_s1_matrix, glove_embedding.test_s2_matrix,
                           glove_embedding.test_y,batch_size)
    random.seed(2024)
    numpy.random.seed(2024)
    torch.cuda.manual_seed(2024)
    torch.manual_seed(2024)
    random_model = ESIM(len_feature, len_hidden, random_embedding.len_words, longest=random_embedding.longest)
    random.seed(2024)
    numpy.random.seed(2024)
    torch.cuda.manual_seed(2024)
    torch.manual_seed(2024)
    glove_model = ESIM(len_feature, len_hidden, glove_embedding.len_words, longest=glove_embedding.longest,
                       weight=torch.tensor(glove_embedding.embedding, dtype=torch.float))
    random.seed(2024)
    numpy.random.seed(2024)
    torch.cuda.manual_seed(202)
    torch.manual_seed(2024)
    trl_ran, tsl_ran, tra_ran, tea_ran = NN_embdding(random_model, train_random, test_random, learning_rate,
                                                     iter_times)
    random.seed(2024)
    numpy.random.seed(2024)
    torch.cuda.manual_seed(2024)
    torch.manual_seed(2024)
    trl_glo, tsl_glo, tra_glo, tea_glo = NN_embdding(glove_model, train_glove, test_glove, learning_rate,
                                                     iter_times)
    x = list(range(1, iter_times + 1))
    matplotlib.pyplot.subplot(2, 2, 1)
    matplotlib.pyplot.plot(x, trl_ran, 'r--', label='random')
    matplotlib.pyplot.plot(x, trl_glo, 'g--', label='glove')
    matplotlib.pyplot.legend(fontsize=10)
    matplotlib.pyplot.title("Train Loss")
    matplotlib.pyplot.xlabel("Iterations")
    matplotlib.pyplot.ylabel("Loss")
    matplotlib.pyplot.subplot(2, 2, 2)
    matplotlib.pyplot.plot(x, tsl_ran, 'r--', label='random')
    matplotlib.pyplot.plot(x, tsl_glo, 'g--', label='glove')
    matplotlib.pyplot.legend(fontsize=10)
    matplotlib.pyplot.title("Test Loss")
    matplotlib.pyplot.xlabel("Iterations")
    matplotlib.pyplot.ylabel("Loss")
    matplotlib.pyplot.subplot(2, 2, 3)
    matplotlib.pyplot.plot(x, tra_ran, 'r--', label='random')
    matplotlib.pyplot.plot(x, tra_glo, 'g--', label='glove')
    matplotlib.pyplot.legend(fontsize=10)
    matplotlib.pyplot.title("Train Accuracy")
    matplotlib.pyplot.xlabel("Iterations")
    matplotlib.pyplot.ylabel("Accuracy")
    matplotlib.pyplot.ylim(0, 1)
    matplotlib.pyplot.subplot(2, 2, 4)
    matplotlib.pyplot.plot(x, tea_ran, 'r--', label='random')
    matplotlib.pyplot.plot(x, tea_glo, 'g--', label='glove')
    matplotlib.pyplot.legend(fontsize=10)
    matplotlib.pyplot.title("Test Accuracy")
    matplotlib.pyplot.xlabel("Iterations")
    matplotlib.pyplot.ylabel("Accuracy")
    matplotlib.pyplot.ylim(0, 1)
    matplotlib.pyplot.tight_layout()
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(8, 8, forward=True)
    matplotlib.pyplot.savefig('main_plot.jpg')
    matplotlib.pyplot.show()


# In[ ]:


# from feature_batch import Random_embedding, Glove_embedding, get_batch
import random
# from comparison_plot_batch import NN_plot, NN_embdding
# from Neural_Network_batch import ESIM

with open('snli_1.0_train.txt', 'r') as f:
    temp = f.readlines()

with open('glove.6B.50d.txt', 'rb') as f:  # for glove embedding
    lines = f.readlines()

# Construct dictionary with glove

trained_dict = dict()
n = len(lines)
for i in range(n):
    line = lines[i].split()
    trained_dict[line[0].decode("utf-8").upper()] = [float(line[j]) for j in range(1, 51)]

data = temp[1:]
# max_item = 100000
# data = data[:max_item]
learning_rate = 0.01
len_feature = 50
len_hidden = 50
iter_times = 50
batch_size = 1000

# random embedding
random.seed(2024)
random_embedding = Random_embedding(data=data)
random_embedding.get_words()
random_embedding.get_id()

# trained embedding : glove
random.seed(2024)
glove_embedding = Glove_embedding(data=data, trained_dict=trained_dict)
glove_embedding.get_words()
glove_embedding.get_id()

NN_plot(random_embedding, glove_embedding, len_feature, len_hidden, learning_rate, batch_size, iter_times)


# In[ ]:


try:   
    get_ipython().system('jupyter nbconvert --to python nlp_task3.ipynb')
    # python即转化为.py，script即转化为.html
    # file_name.ipynb即当前module的文件名
except:
    pass


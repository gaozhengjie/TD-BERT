import torch
import torch.nn as nn
import torch.nn.functional as F

from modeling import BertModel

# import sys
# print(sys.path)

class CNN(nn.Module):
    def __init__(self, config, opt):
        super(CNN, self).__init__()
        n_filters = opt.n_filters  # 卷积核个数
        filter_sizes = opt.filter_sizes  # 卷积核尺寸，多个尺寸则传递过来的是一个列表
        embedding_dim = opt.embed_dim  # embedding维度
        output_dim = opt.output_dim  # 输出维度，此处为3，分别代表负面，中性，正面
        dropout = opt.dropout
        # self.embedding = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.bert = BertModel(config)
        self.convs = nn.ModuleList(
            [nn.Conv2d(in_channels=1, out_channels=n_filters, kernel_size=(fs, embedding_dim)) for fs in filter_sizes])
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)  # 全连接层
        self.dropout = nn.Dropout(dropout)

    # def forward(self, x):
    def forward(self, input_ids, token_type_ids, attention_mask, labels=None, input_t_ids=None, input_t_mask=None, segment_t_ids=None):
        all_encoder_layers, _ = self.bert(input_ids, token_type_ids, attention_mask)
        # x = x[0]
        # x = x.permute(1, 0)  # 行列互换
        # embedded = self.embedding(x)
        embedded = all_encoder_layers[-1] # 取最后一个作为embedding的结果
        embedded = embedded.unsqueeze(1)  # 表示第一维度值为1，则不变，否则添加那个维度
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        cat = self.dropout(torch.cat(pooled, dim=1))

        logits = self.fc(cat)
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss, logits
        else:
            return logits

        # return self.fc(cat)

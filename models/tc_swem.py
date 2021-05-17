# Author: GaoZhengjie
# Description: Target-connection CNN

import torch
import torch.nn as nn
import torch.nn.functional as F

from modeling import BertModel


class TC_SWEM(nn.Module):
    def __init__(self, config, opt):
        super(TC_SWEM, self).__init__()
        self.opt = opt
        n_filters = 1  # 卷积核个数
        filter_sizes = opt.filter_sizes  # 卷积核尺寸，多个尺寸则传递过来的是一个列表
        embedding_dim = opt.embed_dim  # embedding维度
        output_dim = opt.output_dim  # 输出维度，此处为3，分别代表负面，中性，正面
        self.bert = BertModel(config)
        # self.convs = nn.ModuleList(
        #     [nn.Conv2d(in_channels=1, out_channels=n_filters, kernel_size=(fs, embedding_dim*2)) for fs in
        #      filter_sizes])
        self.dropout = nn.Dropout(opt.keep_dropout)
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)  # 全连接层

    def forward(self, input_ids, token_type_ids, attention_mask, labels=None, input_t_ids=None, input_t_mask=None,
                segment_t_ids=None):
        all_encoder_layers, _ = self.bert(input_ids, token_type_ids, attention_mask)
        _, t_pooled_output = self.bert(input_t_ids, segment_t_ids, input_t_mask)
        sentence_embed = all_encoder_layers[-1]
        t_pooled_output = t_pooled_output.unsqueeze(1).expand(-1, sentence_embed.shape[1], -1)  # -1表示维度扩张的时候不涉及那个维度
        target_connection = torch.cat([sentence_embed, t_pooled_output], dim=2)
        target_connection = target_connection.unsqueeze(1)

        kernel_list = [torch.ones(1, 1, kernel_size, self.opt.embed_dim*2).to(self.opt.device) for kernel_size in self.opt.filter_sizes]
        conved = [F.relu(F.conv2d(target_connection, kernel).squeeze(3)) for kernel in kernel_list]

        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        cat = self.dropout(torch.cat(pooled, dim=1))
        logits = self.fc(cat)
        # logits = torch.tanh(logits)
        # nn.Softmax()(logits)
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss, logits
        else:
            return logits

# Author: GaoZhengjie
# Description: 该模型将target词的向量和基于句子对分类的结果拼接后接入多层感知机

import torch
import torch.nn as nn
import torch.nn.functional as F

from modeling import BertModel


class TD_BERT_QA(nn.Module):
    def __init__(self, config, opt):
        super(TD_BERT_QA, self).__init__()
        self.opt = opt
        n_filters = opt.n_filters  # 卷积核个数
        filter_sizes = opt.filter_sizes  # 卷积核尺寸，多个尺寸则传递过来的是一个列表
        embedding_dim = opt.embed_dim  # embedding维度
        output_dim = opt.output_dim  # 输出维度，此处为3，分别代表负面，中性，正面
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(opt.keep_dropout)
        self.fc = nn.Linear(embedding_dim, output_dim)
        self.bn = nn.BatchNorm1d(embedding_dim)


    def forward(self, input_ids, token_type_ids, attention_mask, labels=None, input_t_ids=None, input_t_mask=None,
                segment_t_ids=None, input_left_ids=None, input_left_mask=None, segment_left_ids=None):
        all_encoder_layers, pooled_output = self.bert(input_ids, token_type_ids, attention_mask)
        sentence_embed = all_encoder_layers[-1]
        target_in_sent_embed = torch.zeros(input_ids.size()[0], sentence_embed.size()[-1]).to(self.opt.device)  # 目标词在句子中的embedding向量
        left_len = torch.sum(input_left_ids != 0, dim=-1) - 1  # 注意首尾还有 [CLS] 和 [SEP]
        target_len = torch.sum(input_t_ids != 0, dim=1) - 2  # 注意首尾还有 [CLS] 和 [SEP]
        target_in_sent_idx = torch.cat([left_len.unsqueeze(-1), (left_len + target_len).unsqueeze(-1)], dim=-1)

        for i in range(input_ids.size()[0]):  # 遍历 batch 中的每一个
            target_embed = sentence_embed[i][target_in_sent_idx[i][0]:target_in_sent_idx[i][1]]  # batch_size * max_seq_len * embedding_dim
            target_in_sent_embed[i] = torch.max(target_embed, dim=0)[0]  # 转化成 1 * embedding_dim

        # cat = self.dropout(torch.cat([pooled_output, target_in_sent_embed], dim=1))  # 直接拼接
        # 先归一化，然后对应位置相乘，类似于attention
        # pooled_output = self.bn(pooled_output)
        target_in_sent_embed = self.bn(target_in_sent_embed)
        target_in_sent_embed = target_in_sent_embed.mul(pooled_output)  # 点乘，对应元素相乘不求和
        cat = self.dropout(target_in_sent_embed)

        logits = self.fc(cat)
        logits = torch.tanh(logits)
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss, logits
        else:
            return logits

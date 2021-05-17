# Author: gaozhengjie
# Description: 将原句和目标词军用bert直接表示，然后拼接在一起接全连接

import torch
import torch.nn as nn
import torch.nn.functional as F

from modeling import BertModel


class BBFC(nn.Module):
    def __init__(self, config, opt):
        super(BBFC, self).__init__()
        embedding_dim = opt.embed_dim  # embedding维度
        output_dim = opt.output_dim  # 输出维度，此处为3，分别代表负面，中性，正面
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.fc = nn.Linear(embedding_dim * 2, output_dim)  # 全连接层

    def forward(self, input_ids, token_type_ids, attention_mask, labels=None, input_t_ids=None, input_t_mask=None,
                segment_t_ids=None):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask)
        _, t_pooled_output = self.bert(input_t_ids, segment_t_ids, input_t_mask)
        cat = self.dropout(torch.cat([pooled_output, t_pooled_output], dim=1))
        logits = self.fc(cat)
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss, logits
        else:
            return logits

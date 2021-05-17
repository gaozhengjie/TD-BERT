# Author: GaoZhengjie
# Description: 该模型直接采用target词的向量作为分类的依据，因为在BERT中，词是会考虑上下文的，所以有可能只考虑target词就可以达到好的分类效果

import torch
import torch.nn as nn
import torch.nn.functional as F

from modeling import BertModel


# CrossEntropyLoss for Label Smoothing Regularization
class CrossEntropyLoss_LSR(nn.Module):
    def __init__(self, device, para_LSR=0.2):
        super(CrossEntropyLoss_LSR, self).__init__()
        self.para_LSR = para_LSR
        self.device = device
        self.logSoftmax = nn.LogSoftmax(dim=-1)

    def _toOneHot_smooth(self, label, batchsize, classes):
        prob = self.para_LSR * 1.0 / classes
        # one_hot_label = torch.zeros(batchsize, classes) + prob
        one_hot_label = torch.zeros(batchsize, classes)
        for i in range(batchsize):
            index = label[i]
            # one_hot_label[i, :] += prob * 2
            # one_hot_label[i, index] += (1.0 - self.para_LSR)
            if index != 1:  # 如果为中性，则平滑；否则，不平滑
                one_hot_label[i, :] += prob
                one_hot_label[i, index] += (1.0 - self.para_LSR)
            else:
                one_hot_label[i, index] = 1
        return one_hot_label

    def forward(self, pre, label, size_average=True):
        b, c = pre.size()
        one_hot_label = self._toOneHot_smooth(label, b, c).to(self.device)
        loss = torch.sum(-one_hot_label * self.logSoftmax(pre), dim=1)
        if size_average:
            return torch.mean(loss)
        else:
            return torch.sum(loss)


class TD_BERT(nn.Module):
    def __init__(self, config, opt):
        super(TD_BERT, self).__init__()
        self.opt = opt
        n_filters = opt.n_filters  # 卷积核个数
        filter_sizes = opt.filter_sizes  # 卷积核尺寸，多个尺寸则传递过来的是一个列表
        embedding_dim = opt.embed_dim  # embedding维度
        output_dim = opt.output_dim  # 输出维度，此处为3，分别代表负面，中性，正面
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(opt.keep_dropout)
        self.fc = nn.Linear(embedding_dim, output_dim)  # 全连接层 bbfc
        # self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)  # 全连接层 tc_cnn
        # self.bn1 = nn.BatchNorm1d(output_dim)

    def forward(self, input_ids, token_type_ids, attention_mask, labels=None, input_t_ids=None, input_t_mask=None,
                segment_t_ids=None, input_left_ids=None, input_left_mask=None, segment_left_ids=None):
        all_encoder_layers, _ = self.bert(input_ids, token_type_ids, attention_mask)
        sentence_embed = all_encoder_layers[-1]  # 使用最后一层编码结果进行分类
        # sentence_embed = sum(all_encoder_layers)  # 所有层进行叠加，已经尝试过了，效果并不好，收敛比较慢，且上限较低
        target_in_sent_embed = torch.zeros(input_ids.size()[0], sentence_embed.size()[-1]).to(
            self.opt.device)  # 目标词在句子中的embedding向量
        left_len = torch.sum(input_left_ids != 0, dim=-1) - 1  # 注意首尾还有 [CLS] 和 [SEP]
        target_len = torch.sum(input_t_ids != 0, dim=1) - 2  # 注意首尾还有 [CLS] 和 [SEP]
        target_in_sent_idx = torch.cat([left_len.unsqueeze(-1), (left_len + target_len).unsqueeze(-1)], dim=-1)

        for i in range(input_ids.size()[0]):  # 遍历 batch 中的每一个
            target_embed = sentence_embed[i][target_in_sent_idx[i][0]:target_in_sent_idx[i][
                1]]  # batch_size * max_seq_len * embedding_dim
            target_in_sent_embed[i] = torch.max(target_embed, dim=0)[0]  # 转化成 1 * embedding_dim，取最大值效果最好
            # target_in_sent_embed[i] = target_embed.sum(dim=0)  # 求和
            # target_in_sent_embed[i] = torch.mean(target_embed, dim=0)[0]  # 取均值

        cat = self.dropout(target_in_sent_embed)
        logits = self.fc(cat)
        logits = torch.tanh(logits)
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()  # 标准的交叉熵损失函数
            # loss_fct = CrossEntropyLoss_LSR(device=self.opt.device, para_LSR=self.opt.para_LSR)  # 标签平滑处理后的损失函数，para_LSR的区间是0.1~0.9
            loss = loss_fct(logits, labels)
            return loss, logits
        else:
            return logits

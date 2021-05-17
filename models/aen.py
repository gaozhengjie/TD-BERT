from layers.dynamic_rnn import DynamicLSTM
from layers.squeeze_embedding import SqueezeEmbedding
from layers.attention import Attention, NoQueryAttention
from layers.point_wise_feed_forward import PositionwiseFeedForward
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
        one_hot_label = torch.zeros(batchsize, classes) + prob
        for i in range(batchsize):
            index = label[i]
            one_hot_label[i, index] += (1.0 - self.para_LSR)
        return one_hot_label

    def forward(self, pre, label, size_average=True):
        b, c = pre.size()
        one_hot_label = self._toOneHot_smooth(label, b, c).to(self.device)
        loss = torch.sum(-one_hot_label * self.logSoftmax(pre), dim=1)
        if size_average:
            return torch.mean(loss)
        else:
            return torch.sum(loss)


class AEN_BERT(nn.Module):
    def __init__(self, config, opt):
        super(AEN_BERT, self).__init__()
        self.opt = opt
        self.squeeze_embedding = SqueezeEmbedding()
        self.dropout = nn.Dropout(opt.dropout)
        self.bert = BertModel(config)
        self.attn_k = Attention(opt.embed_dim, out_dim=opt.hidden_dim, n_head=8, score_function='mlp',
                                dropout=opt.dropout)
        self.attn_q = Attention(opt.embed_dim, out_dim=opt.hidden_dim, n_head=8, score_function='mlp',
                                dropout=opt.dropout)
        self.ffn_c = PositionwiseFeedForward(opt.hidden_dim, dropout=opt.dropout)
        self.ffn_t = PositionwiseFeedForward(opt.hidden_dim, dropout=opt.dropout)

        self.attn_s1 = Attention(opt.hidden_dim, n_head=8, score_function='mlp', dropout=opt.dropout)

        self.dense = nn.Linear(opt.hidden_dim * 3, opt.output_dim)

    def forward(self, input_ids, token_type_ids, attention_mask, labels=None, input_t_ids=None, input_t_mask=None,
                segment_t_ids=None):
        context_len = torch.sum(input_ids != 0, dim=1)
        target_len = torch.sum(input_t_ids != 0, dim=1)

        context = self.squeeze_embedding(input_ids, context_len)
        all_encoder_layers, _ = self.bert(input_ids, token_type_ids, attention_mask)
        context = all_encoder_layers[-1]
        context = self.dropout(context)
        target = self.squeeze_embedding(input_t_ids, target_len)
        all_encoder_layers, _ = self.bert(input_t_ids, segment_t_ids, input_t_mask)
        target = all_encoder_layers[-1]
        target = self.dropout(target)

        hc, _ = self.attn_k(context, context)
        hc = self.ffn_c(hc)
        ht, _ = self.attn_q(context, target)
        ht = self.ffn_t(ht)

        s1, _ = self.attn_s1(hc, ht)

        context_len = torch.tensor(context_len, dtype=torch.float).to(self.opt.device)
        target_len = torch.tensor(target_len, dtype=torch.float).to(self.opt.device)

        hc_mean = torch.div(torch.sum(hc, dim=1), context_len.view(context_len.size(0), 1))
        ht_mean = torch.div(torch.sum(ht, dim=1), target_len.view(target_len.size(0), 1))
        s1_mean = torch.div(torch.sum(s1, dim=1), context_len.view(context_len.size(0), 1))

        x = torch.cat((hc_mean, s1_mean, ht_mean), dim=-1)
        out = self.dense(x)
        logits = out
        # logits = torch.nn.Softmax(dim=1)(logits)  # 在XCY的建议下，加上softmax，映射成概率
        if labels is not None:
            # loss_fct = nn.CrossEntropyLoss()
            loss_fct = CrossEntropyLoss_LSR(device=self.opt.device)
            loss = loss_fct(logits, labels)
            return loss, logits
        else:
            return logits

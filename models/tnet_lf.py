import torch
import torch.nn as nn

import torch.nn.functional as F
import numpy
from layers.dynamic_rnn import DynamicLSTM
from modeling import BertModel


class Absolute_Position_Embedding(nn.Module):
    def __init__(self, opt, size=None, mode='sum'):
        self.opt = opt
        self.size = size  # 必须为偶数
        self.mode = mode
        super(Absolute_Position_Embedding, self).__init__()

    def forward(self, x, pos_inx):
        if (self.size is None) or (self.mode == 'sum'):
            self.size = int(x.size(-1))
        batch_size, seq_len = x.size()[0], x.size()[1]
        weight = self.weight_matrix(pos_inx, batch_size, seq_len).to(self.opt.device)
        x = weight.unsqueeze(2) * x
        return x



    def weight_matrix(self, pos_inx, batch_size, seq_len):
        pos_inx = pos_inx.cpu().numpy()
        weight = [[] for i in range(batch_size)]
        for i in range(batch_size):
            for j in range(pos_inx[i][1]):
                relative_pos = pos_inx[i][1] - j
                weight[i].append(1 - relative_pos / 40)
            for j in range(pos_inx[i][1], seq_len):
                relative_pos = j - pos_inx[i][0]
                weight[i].append(1 - relative_pos / 40)
        weight = torch.tensor(weight)
        return weight

class TNet_LF(nn.Module):
    def __init__(self, config, opt):
        super(TNet_LF, self).__init__()
        print("this is TNet_LF model")
        self.bert = BertModel(config)
        self.position = Absolute_Position_Embedding(opt)
        self.opt = opt
        D = opt.embed_dim  # 模型词向量维度
        C = opt.output_dim  # 分类数目
        L = opt.max_seq_length
        HD = opt.hidden_dim
        self.lstm1 = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.lstm2 = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.convs3 = nn.Conv1d(2 * HD, 50, 3, padding=1)
        self.fc1 = nn.Linear(4 * HD, 2 * HD)
        self.fc = nn.Linear(50, C)

    def forward(self, input_ids, token_type_ids, attention_mask, labels=None, input_t_ids=None, input_t_mask=None,
                    segment_t_ids=None, input_left_ids=None, input_left_mask=None, segment_left_ids=None):
        feature_len = torch.sum(input_ids != 0, dim=-1)
        aspect_len = torch.sum(input_t_ids != 0, dim=-1)
        all_encoder_layers, _ = self.bert(input_ids, token_type_ids, attention_mask)
        feature = all_encoder_layers[-1]
        all_encoder_layers, _ = self.bert(input_t_ids, segment_t_ids, input_t_mask)
        aspect = all_encoder_layers[-1]
        left_context_len = torch.sum(input_left_ids != 0, dim=-1)
        aspect_in_text = torch.cat([left_context_len.unsqueeze(1), (left_context_len + aspect_len).unsqueeze(1)], dim=1)
        v, (_, _) = self.lstm1(feature, feature_len)
        e, (_, _) = self.lstm2(aspect, aspect_len)
        v = v.transpose(1, 2)
        e = e.transpose(1, 2)
        for i in range(2):
            a = torch.bmm(e.transpose(1, 2), v)
            a = F.softmax(a, 1)  # (aspect_len,context_len)
            aspect_mid = torch.bmm(e, a)
            aspect_mid = torch.cat((aspect_mid, v), dim=1).transpose(1, 2)
            aspect_mid = F.relu(self.fc1(aspect_mid).transpose(1, 2))
            v = aspect_mid + v
            v = self.position(v.transpose(1, 2), aspect_in_text).transpose(1, 2)
        z = F.relu(self.convs3(v))  # [(N,Co,L), ...]*len(Ks)
        z = F.max_pool1d(z, z.size(2)).squeeze(2)
        out = self.fc(z)

        logits = out
        # logits = torch.nn.Softmax(dim=1)(logits)  # 在XCY的建议下，加上softmax，映射成概率
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss, logits
        else:
            return logits
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import torch.nn.functional as F

from modeling import BertModel

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))  # 对句子做一维卷积
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=3, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)  # 6层
        for i in range(num_levels):
            dilation_size = 2 ** i  # 偏移量 1, 2, 4, 8, 16, 32
            in_channels = num_inputs if i == 0 else num_channels[i-1]  # 下一层的输入维度就是上一层的输出维度
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class TCN(nn.Module):
    def __init__(self, config, opt):
        super(TCN, self).__init__()
        self.opt = opt
        self.num_input = opt.embed_dim
        self.num_channels = [256, 128, 64, 32, 32, 32]
        # self.kernel_size = [2, 3, 4]

        self.bert = BertModel(config)
        # linear
        self.hidden2label1 = nn.Linear(opt.max_seq_length, opt.max_seq_length // 2)  # 全连接，200 -> 100, 两个斜杠表示去尾整除
        self.hidden2label2 = nn.Linear(opt.max_seq_length // 2, opt.label_size)  # 全连接， 100 -> len(label)
        self.net = TemporalConvNet(self.num_input, self.num_channels)  # 输入维度，输出维度
        self.dropout = nn.Dropout(opt.keep_dropout)
        self.bn1 = nn.BatchNorm1d(opt.max_seq_length//2)  # 64
        self.bn2 = nn.BatchNorm1d(opt.label_size)

    def forward(self, input_ids, token_type_ids, attention_mask, labels=None, input_t_ids=None, input_t_mask=None,
                segment_t_ids=None):
        all_encoder_layers, _ = self.bert(input_ids, token_type_ids, attention_mask)
        x = all_encoder_layers[-1]  # 1,128,768 batch_size, seq_len, emb_dim
        x = x.permute(0, 2, 1)  # 1,768,128
        x = self.net(x)  # 1,32,128
        # x = x.permute(2, 0, 1)
        # x = F.max_pool2d(x, kernel_size=(x.size(1), 1)).squeeze(1)
        x = F.avg_pool2d(x, kernel_size=(x.size(1), 1)).squeeze(1)  # 1,128
        x = self.dropout(x)
        x = self.hidden2label1(x)  # 1,64
        x = self.bn1(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.hidden2label2(x)
        x = self.bn2(x)
        x = torch.tanh(x)

        # return x
        logits = x
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss, logits
        else:
            return logits




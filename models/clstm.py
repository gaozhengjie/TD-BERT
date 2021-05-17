import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from modeling import BertModel

class CLSTM(nn.Module):
    def __init__(self, config, opt):
        super(CLSTM, self).__init__()
        self.model_name = 'clstm'
        self.opt = opt
        # self.batch_size = opt.batch_size
        self.batch_size = opt.train_batch_size
        self.hidden_dim = opt.hidden_dim
        self.num_layers = opt.lstm_layers
        self.mean = opt.lstm_mean
        # self.vocab_size = opt.vocab_size
        self.vocab_size = config.vocab_size
        self.embedding_dim = opt.embed_dim
        self.label_size = opt.label_size
        self.in_channel = 1
        self.kernel_nums = 150
        self.kernel_sizes = 3
        self.mean = opt.lstm_mean
        self.use_gpu = torch.cuda.is_available()
        # self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim, padding_idx=self.vocab_size - 1,
        #                               _weight=opt.embeddings)
        self.bert = BertModel(config)
        # self.embedding.weight = nn.Parameter(opt.embeddings)
        # cnn
        self.convs1 = nn.Conv2d(1, self.kernel_nums, (self.kernel_sizes, opt.embed_dim))
        # LSTM
        self.lstm = nn.LSTM(self.kernel_nums, self.hidden_dim, num_layers=self.num_layers, dropout=opt.keep_dropout)
        # linear
        self.hidden2label1 = nn.Linear(self.hidden_dim, self.hidden_dim // 2)
        self.hidden2label2 = nn.Linear(self.hidden_dim // 2, self.label_size)
        # dropout
        self.dropout = nn.Dropout(opt.keep_dropout)
        self.bn1 = nn.BatchNorm1d(self.hidden_dim)
        self.bn2 = nn.BatchNorm1d(self.hidden_dim // 2)

    def init_hidden(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size

        if self.use_gpu:
            h0 = Variable(torch.zeros(1 * self.num_layers, batch_size, self.hidden_dim).cuda())
            c0 = Variable(torch.zeros(1 * self.num_layers, batch_size, self.hidden_dim).cuda())
        else:
            h0 = Variable(torch.zeros(1 * self.num_layers, batch_size, self.hidden_dim))
            c0 = Variable(torch.zeros(1 * self.num_layers, batch_size, self.hidden_dim))
        return (h0, c0)

    def forward(self, input_ids, token_type_ids, attention_mask, labels=None):
        all_encoder_layers, _ = self.bert(input_ids, token_type_ids, attention_mask)
        x = embed = all_encoder_layers[-1] # 取最后一个作为embedding的结果
        # CNN
        cnn_x = embed
        cnn_x = self.dropout(cnn_x)
        cnn_x = cnn_x.unsqueeze(1)
        cnn_x = F.relu(self.convs1(cnn_x)).squeeze(3)
        cnn_x = cnn_x.permute(2, 0, 1)
        # LSTM
        self.hidden = self.init_hidden(batch_size=x.size()[0])
        lstm_out, self.hidden = self.lstm(cnn_x, self.hidden)  # 输入 seq_num * batch_size * input_dim
        lstm_out = lstm_out.permute(1, 2, 0)
        if self.mean == "mean":
            lstm_out = torch.mean(lstm_out, 2)
        elif self.mean == "last":
            lstm_out = lstm_out[:, :, -1]
        elif self.mean == "maxpool":
            lstm_out = F.max_pool1d(lstm_out, lstm_out.size(2)).squeeze(2)
        elif self.mean == "keyword":
            lstm_out = lstm_out.permute(0, 2, 1)
            lstm_out = lstm_out[range(x.size()[0]), lstm_out.sum(dim=2).max(dim=1)[1], :]  # 64x128
        elif self.mean == "attention":
            lstm_out = lstm_out.permute(0, 2, 1)
            h_n, c_n = self.hidden
            lstm_out = self.attention(lstm_out, h_n)
        # linear
        # lstm_out = self.hidden2label1(F.tanh(lstm_out))
        lstm_out = self.hidden2label1(torch.tanh(lstm_out))
        # lstm_out = self.dropout(lstm_out)
        # lstm_out = self.hidden2label2(F.tanh(lstm_out))
        lstm_out = self.hidden2label2(torch.tanh(lstm_out))

        logits = lstm_out
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss, logits
        else:
            return logits
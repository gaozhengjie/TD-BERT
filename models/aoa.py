from layers.dynamic_rnn import DynamicLSTM
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from modeling import BertModel


class AOA(nn.Module):
    def __init__(self, config, opt):
        super(AOA, self).__init__()
        self.opt = opt
        # self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.bert = BertModel(config)
        self.ctx_lstm = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.asp_lstm = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.dense = nn.Linear(2 * opt.hidden_dim, opt.output_dim)

    def forward(self, input_ids, token_type_ids, attention_mask, labels=None, input_t_ids=None, input_t_mask=None,
                    segment_t_ids=None):
        # text_raw_indices = inputs[0] # batch_size x seq_len
        # aspect_indices = inputs[1] # batch_size x seq_len

        ctx_len = torch.sum(input_ids != 0, dim=1)
        asp_len = torch.sum(input_t_ids != 0, dim=1)
        all_encoder_layers, _ = self.bert(input_ids, token_type_ids, attention_mask)
        ctx = all_encoder_layers[-1]
        all_encoder_layers, _ = self.bert(input_t_ids, segment_t_ids, input_t_mask)
        asp = all_encoder_layers[-1]
        ctx_out, (_, _) = self.ctx_lstm(ctx, ctx_len) #  batch_size x (ctx) seq_len x 2*hidden_dim
        asp_out, (_, _) = self.asp_lstm(asp, asp_len) # batch_size x (asp) seq_len x 2*hidden_dim
        interaction_mat = torch.matmul(ctx_out, torch.transpose(asp_out, 1, 2)) # batch_size x (ctx) seq_len x (asp) seq_len
        alpha = F.softmax(interaction_mat, dim=1) # col-wise, batch_size x (ctx) seq_len x (asp) seq_len
        beta = F.softmax(interaction_mat, dim=2) # row-wise, batch_size x (ctx) seq_len x (asp) seq_len
        beta_avg = beta.mean(dim=1, keepdim=True) # batch_size x 1 x (asp) seq_len
        gamma = torch.matmul(alpha, beta_avg.transpose(1, 2)) # batch_size x (ctx) seq_len x 1
        weighted_sum = torch.matmul(torch.transpose(ctx_out, 1, 2), gamma).squeeze(-1) # batch_size x 2*hidden_dim
        out = self.dense(weighted_sum) # batch_size x polarity_dim

        logits = out
        # logits = torch.nn.Softmax(dim=1)(logits)  # 在XCY的建议下，加上softmax，映射成概率
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss, logits
        else:
            return logits
from layers.dynamic_rnn import DynamicLSTM
from layers.attention import Attention
import torch
import torch.nn as nn
from modeling import BertModel

class IAN(nn.Module):
    def __init__(self, config, opt):
        super(IAN, self).__init__()
        self.opt = opt
        self.bert = BertModel(config)
        self.lstm_context = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True)
        self.lstm_aspect = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True)
        self.attention_aspect = Attention(opt.hidden_dim, score_function='bi_linear')
        self.attention_context = Attention(opt.hidden_dim, score_function='bi_linear')
        self.dense = nn.Linear(opt.hidden_dim*2, opt.output_dim)

    def forward(self, input_ids, token_type_ids, attention_mask, labels=None, input_t_ids=None, input_t_mask=None,
                    segment_t_ids=None):
        # text_raw_indices, aspect_indices = inputs[0], inputs[1]
        text_raw_len = torch.sum(input_ids != 0, dim=-1)
        aspect_len = torch.sum(input_t_ids != 0, dim=-1)

        # context = self.embed(text_raw_indices)
        # aspect = self.embed(aspect_indices)
        all_encoder_layers, _ = self.bert(input_ids, token_type_ids, attention_mask)
        context = all_encoder_layers[-1]
        all_encoder_layers, _ = self.bert(input_t_ids, segment_t_ids, input_t_mask)
        aspect = all_encoder_layers[-1]
        context, (_, _) = self.lstm_context(context, text_raw_len)
        aspect, (_, _) = self.lstm_aspect(aspect, aspect_len)

        aspect_len = torch.tensor(aspect_len, dtype=torch.float).to(self.opt.device)
        aspect_pool = torch.sum(aspect, dim=1)
        aspect_pool = torch.div(aspect_pool, aspect_len.view(aspect_len.size(0), 1))

        text_raw_len = torch.tensor(text_raw_len, dtype=torch.float).to(self.opt.device)
        context_pool = torch.sum(context, dim=1)
        context_pool = torch.div(context_pool, text_raw_len.view(text_raw_len.size(0), 1))

        aspect_final, _ = self.attention_aspect(aspect, context_pool)
        aspect_final = aspect_final.squeeze(dim=1)
        context_final, _ = self.attention_context(context, aspect_pool)
        context_final = context_final.squeeze(dim=1)

        x = torch.cat((aspect_final, context_final), dim=-1)
        out = self.dense(x)

        logits = out
        # logits = torch.nn.Softmax(dim=1)(logits)  # 在XCY的建议下，加上softmax，映射成概率
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss, logits
        else:
            return logits

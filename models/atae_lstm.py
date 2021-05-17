from layers.attention import Attention, NoQueryAttention
from layers.dynamic_rnn import DynamicLSTM
from layers.squeeze_embedding import SqueezeEmbedding
import torch
import torch.nn as nn
from modeling import BertModel


class ATAE_LSTM(nn.Module):
    def __init__(self, config, opt):
        super(ATAE_LSTM, self).__init__()
        self.opt = opt
        # self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.bert = BertModel(config)
        self.squeeze_embedding = SqueezeEmbedding()
        self.lstm = DynamicLSTM(opt.embed_dim*2, opt.hidden_dim, num_layers=1, batch_first=True)
        self.attention = NoQueryAttention(opt.hidden_dim+opt.embed_dim, score_function='bi_linear')
        self.dense = nn.Linear(opt.hidden_dim, opt.output_dim)

    def forward(self, input_ids, token_type_ids, attention_mask, labels=None, input_t_ids=None, input_t_mask=None,
                    segment_t_ids=None):
        # text_raw_indices, aspect_indices = inputs[0], inputs[1]

        x_len = torch.sum(input_ids != 0, dim=-1)
        x_len_max = torch.max(x_len)
        aspect_len = torch.tensor(torch.sum(input_t_ids != 0, dim=-1), dtype=torch.float).to(self.opt.device)

        # x = self.embed(text_raw_indices)
        all_encoder_layers, _ = self.bert(input_ids, token_type_ids, attention_mask)
        x = all_encoder_layers[-1]
        x = self.squeeze_embedding(x, x_len)
        # aspect = self.embed(aspect_indices)
        all_encoder_layers, _ = self.bert(input_t_ids, segment_t_ids, input_t_mask)
        aspect = all_encoder_layers[-1]
        aspect_pool = torch.div(torch.sum(aspect, dim=1), aspect_len.view(aspect_len.size(0), 1))
        aspect = torch.unsqueeze(aspect_pool, dim=1).expand(-1, x_len_max, -1)
        x = torch.cat((aspect, x), dim=-1)

        h, (_, _) = self.lstm(x, x_len)
        ha = torch.cat((h, aspect), dim=-1)
        _, score = self.attention(ha)
        output = torch.squeeze(torch.bmm(score, h), dim=1)

        out = self.dense(output)
        logits = out
        # logits = torch.nn.Softmax(dim=1)(logits)  # 在XCY的建议下，加上softmax，映射成概率
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss, logits
        else:
            return logits
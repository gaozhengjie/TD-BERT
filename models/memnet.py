from layers.attention import Attention
import torch
import torch.nn as nn
from layers.squeeze_embedding import SqueezeEmbedding
from modeling import BertModel


class MemNet(nn.Module):
    def locationed_memory(self, memory, memory_len):
        # here we just simply calculate the location vector in Model2's manner
        for i in range(memory.size(0)):
            for idx in range(memory_len[i]):
                memory[i][idx] *= (1 - float(idx) / int(memory_len[i]))
        return memory

    def __init__(self, config, opt):
        super(MemNet, self).__init__()
        self.opt = opt
        self.bert = BertModel(config)
        self.squeeze_embedding = SqueezeEmbedding(batch_first=True)
        self.attention = Attention(opt.embed_dim, score_function='mlp')
        self.x_linear = nn.Linear(opt.embed_dim, opt.embed_dim)
        self.dense = nn.Linear(opt.embed_dim, opt.output_dim)

    def forward(self, input_ids, token_type_ids, attention_mask, labels=None, input_t_ids=None, input_t_mask=None,
                    segment_t_ids=None):
        memory_len = torch.sum(input_ids != 0, dim=-1)
        aspect_len = torch.sum(input_t_ids != 0, dim=-1)
        nonzeros_aspect = torch.tensor(aspect_len, dtype=torch.float).to(self.opt.device)

        all_encoder_layers, _ = self.bert(input_ids, token_type_ids, attention_mask)
        memory = all_encoder_layers[-1]
        memory = self.squeeze_embedding(memory, memory_len)
        # memory = self.locationed_memory(memory, memory_len)
        all_encoder_layers, _ = self.bert(input_t_ids, segment_t_ids, input_t_mask)
        aspect = all_encoder_layers[-1]
        aspect = torch.sum(aspect, dim=1)
        aspect = torch.div(aspect, nonzeros_aspect.view(nonzeros_aspect.size(0), 1))
        x = aspect.unsqueeze(dim=1)
        for _ in range(self.opt.hops):
            x = self.x_linear(x)
            out_at, _ = self.attention(memory, x)
            x = out_at + x
        x = x.view(x.size(0), -1)
        out = self.dense(x)
        logits = out
        # logits = torch.nn.Softmax(dim=1)(logits)  # 在XCY的建议下，加上softmax，映射成概率
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss, logits
        else:
            return logits
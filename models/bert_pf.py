import torch
import torch.nn as nn
import torch.nn.functional as F

from modeling import BertModel


class Bert_PF(nn.Module):
    def __init__(self, config, opt):
        super(Bert_PF, self).__init__()
        self.n_filters = opt.n_filters  # 卷积核个数
        filter_sizes = opt.filter_sizes  # 卷积核尺寸，多个尺寸则传递过来的是一个列表
        embedding_dim = opt.embed_dim  # embedding维度
        output_dim = opt.output_dim  # 输出维度，此处为3，分别代表负面，中性，正面
        dropout = 0.5

        self.bert = BertModel(config)
        self.convs_target_1 = nn.ModuleList(
            [nn.Conv2d(in_channels=1,
                       out_channels=int(filter_sizes[0] * self.n_filters * embedding_dim / len(filter_sizes)),
                       kernel_size=(fs, embedding_dim)) for fs in filter_sizes])
        self.convs_target_2 = nn.ModuleList(
            [nn.Conv2d(in_channels=1,
                       out_channels=int(filter_sizes[1] * self.n_filters * embedding_dim / len(filter_sizes)),
                       kernel_size=(fs, embedding_dim)) for fs in filter_sizes])
        self.convs_target_3 = nn.ModuleList(
            [nn.Conv2d(in_channels=1,
                       out_channels=int(filter_sizes[2] * self.n_filters * embedding_dim / len(filter_sizes)),
                       kernel_size=(fs, embedding_dim)) for fs in filter_sizes])
        self.convs_target_4 = nn.ModuleList(
            [nn.Conv2d(in_channels=1,
                       out_channels=int(filter_sizes[3] * self.n_filters * embedding_dim / len(filter_sizes)),
                       kernel_size=(fs, embedding_dim)) for fs in filter_sizes])
        self.convs_t = nn.ModuleList(
            [nn.Conv2d(in_channels=1, out_channels=self.n_filters, kernel_size=(fs, embedding_dim)) for fs in
             filter_sizes])
        for i in range(len(filter_sizes)):
            self.convs_t._modules['0'].weight.requires_grad = False
        # self.fc = nn.Linear(len(filter_sizes) * self.n_filters * 2, output_dim)  # 全连接层
        self.fc = nn.Linear(len(filter_sizes) * self.n_filters + embedding_dim, output_dim)  # 全连接层
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, token_type_ids, attention_mask, labels=None, input_t_ids=None, input_t_mask=None,
                segment_t_ids=None):
        all_encoder_layers, pooled_output = self.bert(input_ids, token_type_ids, attention_mask)
        embedded = all_encoder_layers[-1]
        embedded = embedded.unsqueeze(1)  # 表示第一维度值为1，则不变，否则添加那个维度
        pooled_output = self.dropout(pooled_output)


        all_encoder_layers, _ = self.bert(input_t_ids, segment_t_ids, input_t_mask)
        t_embedded = all_encoder_layers[-1]
        t_embedded = t_embedded.unsqueeze(1)
        t_conved_1 = [F.relu(conv(t_embedded)).squeeze(3) for conv in self.convs_target_1]
        t_conved_2 = [F.relu(conv(t_embedded)).squeeze(3) for conv in self.convs_target_2]
        t_conved_3 = [F.relu(conv(t_embedded)).squeeze(3) for conv in self.convs_target_3]
        t_conved_4 = [F.relu(conv(t_embedded)).squeeze(3) for conv in self.convs_target_4]
        t_pooled_1 = [F.avg_pool1d(conv, conv.shape[2]).squeeze(2) for conv in t_conved_1]
        t_pooled_2 = [F.avg_pool1d(conv, conv.shape[2]).squeeze(2) for conv in t_conved_2]
        t_pooled_3 = [F.avg_pool1d(conv, conv.shape[2]).squeeze(2) for conv in t_conved_3]
        t_pooled_4 = [F.avg_pool1d(conv, conv.shape[2]).squeeze(2) for conv in t_conved_4]
        # t_cat = torch.cat(t_pooled, dim=1).unsqueeze(0).unsqueeze(0)  # torch.cat() 在给定维度上对输入的张量序列进行连接操作，此处拼接成 1*1*25*300
        t_cat_1 = torch.cat(t_pooled_1, dim=1).unsqueeze(1).unsqueeze(2)
        t_cat_2 = torch.cat(t_pooled_2, dim=1).unsqueeze(1).unsqueeze(2)
        t_cat_3 = torch.cat(t_pooled_3, dim=1).unsqueeze(1).unsqueeze(2)
        t_cat_4 = torch.cat(t_pooled_4, dim=1).unsqueeze(1).unsqueeze(2)
        p_pooled = []
        for i in range(embedded.shape[0]):
            filter_list = []
            filter_list.append(
                t_cat_1[i].squeeze(0).squeeze(0).view(self.n_filters, 1, 768))  # 100*1*768 squeeze去除size为1的维度，包括行和列
            filter_list.append(t_cat_2[i].squeeze(0).squeeze(0).view(self.n_filters, 2, 768))
            filter_list.append(t_cat_3[i].squeeze(0).squeeze(0).view(self.n_filters, 3, 768))
            filter_list.append(t_cat_4[i].squeeze(0).squeeze(0).view(self.n_filters, 4, 768))
            p_conved = []
            for idx, conv in enumerate(self.convs_t):
                p_filter = filter_list[idx]
                conv.weight.data = p_filter.unsqueeze(1)
                p_conved.append(F.relu(conv(embedded[i].unsqueeze(0))).squeeze(3))
            p_pooled.append(torch.cat([F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in p_conved], dim=1))

        cat = self.dropout(torch.cat([pooled_output, torch.cat(p_pooled, dim=0)], dim=1))

        logits = self.fc(cat)
        logits = torch.tanh(logits)
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss, logits
        else:
            return logits

            # return self.fc(cat)

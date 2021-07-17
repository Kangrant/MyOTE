# -*- coding: utf-8 -*-

import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.dynamic_rnn import DynamicRNN
from tag_utils import  bio2bieos,bieos2span, find_span_with_end,bieos2span_express
from transformers import BertModel, BertConfig


class Biaffine(nn.Module):
    def __init__(self, opt, in1_features, in2_features, out_features, bias=(True, True)):
        super(Biaffine, self).__init__()
        self.opt = opt
        self.in1_features = in1_features
        self.in2_features = in2_features
        self.out_features = out_features
        self.bias = bias
        self.linear_input_size = in1_features + int(bias[0])
        self.linear_output_size = out_features * (in2_features + int(bias[1]))
        self.linear = nn.Linear(in_features=self.linear_input_size,
                                out_features=self.linear_output_size,
                                bias=False)

    def forward(self, input1, input2):
        batch_size, len1, dim1 = input1.size()
        batch_size, len2, dim2 = input2.size()
        if self.bias[0]:
            ones = torch.ones(batch_size, len1, 1).to(self.opt.device)
            input1 = torch.cat((input1, ones), dim=2)  ##[batch,max_seq_len,dim1+1]
            dim1 += 1
        if self.bias[1]:
            ones = torch.ones(batch_size, len2, 1).to(self.opt.device)
            input2 = torch.cat((input2, ones), dim=2)
            dim2 += 1
        affine = self.linear(input1)  # [batch,max_seq_len,linear_output_size]
        affine = affine.view(batch_size, len1 * self.out_features, dim2)
        input2 = torch.transpose(input2, 1, 2)
        biaffine = torch.bmm(affine, input2)
        biaffine = torch.transpose(biaffine, 1, 2)
        biaffine = biaffine.contiguous().view(batch_size, len2, len1, self.out_features)

        return biaffine  # [batch,max_seq_len,max_seq_len,4]


class OTE(nn.Module):
    authorized_unexpected_keys = [r"pooler"]

    def __init__(self,embedding_matrix ,opt, idx2tag, idx2polarity,idx2target,idx2express):
        super(OTE, self).__init__()
        self.opt = opt
        self.idx2tag = idx2tag
        self.tag_dim = len(self.idx2tag)
        self.idx2polarity = idx2polarity
        self.idx2express = idx2express
        self.idx2target = idx2target
        self.target_dim = len(self.idx2target)
        self.express_dim = len(self.idx2express)
        if embedding_matrix == torch.tensor([0]):
            self.config = BertConfig.from_pretrained('./bert-base-chinese')
            self.bert = BertModel.from_pretrained('./bert-base-chinese', config=self.config, add_pooling_layer=False)

            self.ap_fc = nn.Linear(self.config.hidden_size, 200)
            self.op_fc = nn.Linear(self.config.hidden_size, 200)
            self.express_fc = nn.Linear(self.config.hidden_size,200)
            self.sen_polarity_fc = nn.Linear(self.config.hidden_size,opt.polarities_dim)
        else:
            self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
            self.embed_dropout = nn.Dropout(0.5)
            self.lstm = DynamicRNN(opt.embed_dim, opt.hidden_dim, batch_first=True, bidirectional=True)

            self.ap_fc = nn.Linear(2 * opt.hidden_dim, 200)
            self.op_fc = nn.Linear(2 * opt.hidden_dim, 200)
            self.express_fc = nn.Linear(2 * opt.hidden_dim,200)
            self.sen_polarity_fc = nn.Linear(2 * opt.hidden_dim,opt.polarities_dim)


        self.triplet_biaffine = Biaffine(opt, 100, 100, opt.polarities_dim, bias=(True, False))
        self.target_biaffine = Biaffine(opt, 100, 100, self.target_dim, bias=(True, False))
        self.ap_tag_fc = nn.Linear(100, self.tag_dim)
        self.op_tag_fc = nn.Linear(100, self.tag_dim)
        self.express_tag_fc = nn.Linear(200,self.express_dim)
    def calc_loss(self, outputs, targets):
        ap_out, op_out, triplet_out, sen_polarity_out,target_out,express_out= outputs
        ap_tag, op_tag, triplet, mask,sentece_polarity_tag,target_gold , express_label= targets
        # tag loss
        ap_tag_loss = F.cross_entropy(ap_out.flatten(0, 1), ap_tag.flatten(0, 1), reduction='none',
                                      ignore_index=-100)  # [batch * seq_len]
        ap_tag_loss = ap_tag_loss.masked_select(mask.flatten(0, 1)).sum() / mask.sum()
        op_tag_loss = F.cross_entropy(op_out.flatten(0, 1), op_tag.flatten(0, 1), reduction='none', ignore_index=-100)
        op_tag_loss = op_tag_loss.masked_select(mask.flatten(0, 1)).sum() / mask.sum()
        tag_loss = ap_tag_loss + op_tag_loss
        # sentiment loss
        mat_mask = mask.unsqueeze(2) * mask.unsqueeze(1)  # [batch,max_seq_len,max_seq_len]
        sentiment_loss = F.cross_entropy(triplet_out.view(-1, self.opt.polarities_dim), triplet.view(-1),
                                         reduction='none',ignore_index=-100)
        sentiment_loss = sentiment_loss.masked_select(mat_mask.view(-1)).sum() / mat_mask.sum()

        target_loss = F.cross_entropy(target_out.view(-1, self.target_dim), target_gold.view(-1),
                                         reduction='none',ignore_index=-100)
        target_loss = target_loss.masked_select(mat_mask.view(-1)).sum() / mat_mask.sum()

        sen_polarity_loss = F.cross_entropy(sen_polarity_out,sentece_polarity_tag,reduction='mean')

        express_loss = F.cross_entropy(express_out.flatten(0, 1),express_label.flatten(0, 1),reduction='none')
        express_loss = express_loss.masked_select(mask.flatten(0, 1)).sum() / mask.sum()

        return tag_loss + sentiment_loss + sen_polarity_loss + target_loss + express_loss


    def forward(self, inputs,opt):
        text_indices, text_mask = inputs  # [batch_size,max_seq_len]
        text_len = torch.sum(text_mask, dim=-1)
        if opt.useBert:
            # output: (last_hidden_state,pooler_output)
            # last_hidden_state:[batch,seq_len,768]    pooler_output:[batch,768]
            output = self.bert(input_ids=text_indices, attention_mask=text_mask)
            out = output[0]  # [batch,seq_len,768]
        else:
            embed = self.embed(text_indices) #[batch_size,max_seq_len,embedding_dim]
            embed = self.embed_dropout(embed)
            out, (_, _) = self.lstm(embed, text_len) #out->[batch,max_seq_len,hidden_size * num_directions]

        ap_rep = F.relu(self.ap_fc(out))  # [batch,seq_len,200]
        op_rep = F.relu(self.op_fc(out))  # [batch,seq_len,200]
        express_rep = F.relu(self.express_fc(out))

        ap_node, ap_rep = torch.chunk(ap_rep, 2, dim=2)  # 分块 与cat操作相反
        op_node, op_rep = torch.chunk(op_rep, 2, dim=2)  ##[batch,max_seq_len,100]

        ap_out = self.ap_tag_fc(ap_rep)  ##[batch,max_seq_len,tag_size]
        op_out = self.op_tag_fc(op_rep)

        triplet_out = self.triplet_biaffine(ap_node, op_node)  # [batch,max_seq_len,max_seq_len,polarities_dim]
        target_out = self.target_biaffine(ap_node, op_node)  # [batch,max_seq_len,max_seq_len,target_dim]

        #池化[batch,seq,768]->[batch,768]
        sen_polarity_rep = out.mean(dim=1)
        #FC [batch,768]-> [batch,polarity_dim]
        sen_polarity_out = self.sen_polarity_fc(sen_polarity_rep)

        express_out = self.express_tag_fc(express_rep)

        return [ap_out, op_out, triplet_out,sen_polarity_out,target_out,express_out]

    def inference(self, inputs, text_indices, text_mask):
        text_len = torch.sum(text_mask, dim=-1)#[batch_size]
        ap_out, op_out, triplet_out, sen_polarity_out,target_out,express_out= inputs

        batch_size = text_len.size(0)

        ap_tags = [[] for _ in range(batch_size)]
        op_tags = [[] for _ in range(batch_size)]
        express_tags = [[] for _ in range(batch_size)]
        start_time = time.time()
        # for b in range(batch_size):
        #     for i in range(text_len[b]):
        #         ap_tags[b].append(ap_out[b, i, :].argmax(0).item())
        #         # op_tags[b].append(op_out[b, i, :].argmax(0).item())
        #         # express_tags[b].append(express_out[b, i, :].argmax(0).item())
        ap_tags = torch.argmax(ap_out, dim=2).cpu().numpy().tolist()
        op_tags = torch.argmax(op_out, dim=2).cpu().numpy().tolist()
        express_tags = torch.argmax(express_out, dim=2).cpu().numpy().tolist()
        for idx, (sent_a, sent_o, sent_e) in enumerate(zip(ap_tags, op_tags, express_tags)):
            ap_tags[idx] = sent_a[: text_len[idx]]
            op_tags[idx] = sent_o[: text_len[idx]]
            express_tags[idx] = sent_e[: text_len[idx]]            
        # temp_true = torch.argmax(ap_out, dim=2, keepdim=True)
        # for b in range(batch_size):
        #     for i in range(text_len[b]):
        #         op_tags[b].append(op_out[b, i, :].argmax(0).item())
        # for b in range(batch_size):
        #     for i in range(text_len[b]):
        #         express_tags[b].append(express_out[b, i, :].argmax(0).item())


        text_indices = text_indices.cpu().numpy().tolist()

        start_time = time.time()
        ap_spans = self.aspect_decode(text_indices, ap_tags, self.idx2tag)
        op_spans = self.opinion_decode(text_indices, op_tags, self.idx2tag)
        express_spans = self.express_decode(text_indices,express_tags,self.idx2express)
        end_time = time.time()


        mat_mask = (text_mask.unsqueeze(2) * text_mask.unsqueeze(1)).unsqueeze(3).expand(
            -1, -1, -1, self.opt.polarities_dim)  # batch x seq x seq x polarity

        mat_mask_target = (text_mask.unsqueeze(2) * text_mask.unsqueeze(1)).unsqueeze(3).expand(
            -1, -1, -1, self.target_dim)

        # triplet_indices->[batch,max_seq_len,max_seq_len,tag_dim]
        triplet_indices = torch.zeros_like(triplet_out).to(self.opt.device)
        triplet_indices = triplet_indices.scatter_(3, triplet_out.argmax(dim=3, keepdim=True), 1) * mat_mask.float()
        triplet_indices = torch.nonzero(triplet_indices).cpu().numpy().tolist()
        triplets = self.sentiment_decode(text_indices, ap_tags, op_tags, triplet_indices, self.idx2tag,
                                         self.idx2polarity)

        target_indices = torch.zeros_like(target_out).to(self.opt.device)
        target_indices = target_indices.scatter_(3, target_out.argmax(dim=3, keepdim=True), 1) * mat_mask_target.float()
        target_indices = torch.nonzero(target_indices).cpu().numpy().tolist()
        target = self.target_decode(text_indices, ap_tags, op_tags, target_indices, self.idx2tag,
                                         self.idx2target)



        sen_polarity_tags= sen_polarity_out.argmax(1)
        #sen_polarity_tags = self.sen_polarity_decode(sen_polarity_ids,self.idx2polarity)

        return [ap_spans, op_spans, triplets, sen_polarity_tags,target,express_spans]

    @staticmethod
    def aspect_decode(text_indices, tags, idx2tag):
        # text_indices = text_indices.cpu().numpy().tolist()
        batch_size = len(tags)
        result = [[] for _ in range(batch_size)]
        for i, tag_seq in enumerate(tags):
            _tag_seq = list(map(lambda x: idx2tag[x], tag_seq))
            result[i] = bieos2span(bio2bieos(_tag_seq), tp='')  # 训练时使用BIO标签,解码时再转化为BIEOS
        return result

    @staticmethod
    def opinion_decode(text_indices, tags, idx2tag):
        # text_indices = text_indices.cpu().numpy().tolist()
        batch_size = len(tags)
        result = [[] for _ in range(batch_size)]
        for i, tag_seq in enumerate(tags):
            _tag_seq = list(map(lambda x: idx2tag[x], tag_seq))
            result[i] = bieos2span(bio2bieos(_tag_seq), tp='')
        return result

    @staticmethod
    def sentiment_decode(text_indices, ap_tags, op_tags, triplet_indices, idx2tag, idx2polarity):
        # text_indices = text_indices.cpu().numpy().tolist()
        batch_size = len(ap_tags)
        result = [[] for _ in range(batch_size)]
        for i in range(len(triplet_indices)):
            b, ap_i, op_i, po = triplet_indices[i]
            if po == 0:
                continue
            _ap_tags = list(map(lambda x: idx2tag[x], ap_tags[b]))
            _op_tags = list(map(lambda x: idx2tag[x], op_tags[b]))
            ap_beg, ap_end = find_span_with_end(ap_i, text_indices[b], _ap_tags, tp='')
            op_beg, op_end = find_span_with_end(op_i, text_indices[b], _op_tags, tp='')
            triplet = (ap_beg, ap_end, op_beg, op_end, po)
            result[b].append(triplet)
        return result

    @staticmethod
    def sen_polarity_decode(sen_polarity_ids,idx2polarity):
        sen_polarity_ids = sen_polarity_ids.cpu().numpy().tolist()
        tag_seq = [idx2polarity[id] for id in sen_polarity_ids]
        return tag_seq

    @staticmethod
    def target_decode(text_indices, ap_tags, op_tags, target_indices, idx2tag, idx2target):
        # text_indices = text_indices.cpu().numpy().tolist()
        batch_size = len(ap_tags)
        result = [[] for _ in range(batch_size)]
        for i in range(len(target_indices)):
            b, ap_i, ap_j, po = target_indices[i]
            if po == 0 or ap_i!=ap_j:
                continue
            _ap_tags = list(map(lambda x: idx2tag[x], ap_tags[b]))
            # _op_tags = list(map(lambda x: idx2tag[x], op_tags[b]))
            ap_beg, ap_end = find_span_with_end(ap_i, text_indices[b], _ap_tags, tp='')
            # op_beg, op_end = find_span_with_end(op_i, text_indices[b], _op_tags, tp='')
            po = idx2target[po]
            target = (ap_beg, ap_end,  po)
            result[b].append(target)
        return result

    @staticmethod
    def express_decode(text_indices, tags, idx2tag):
        batch_size = len(tags)
        result = [[] for _ in range(batch_size)]
        for i, tag_seq in enumerate(tags):
            _tag_seq = list(map(lambda x: idx2tag[x], tag_seq))
            result[i] = bieos2span_express(_tag_seq)
        return result






# -*- coding: utf-8 -*-

import os
import pickle
import torch
import torch.nn.functional as F
import argparse
from bucket_iterator import BucketIterator
from data_utils import ABSADataReader,build_tokenizer,build_embedding_matrix
from models import  OTE
from transformers import BertTokenizer


class Inferer:
    """A simple inference example"""

    def __init__(self, opt):
        self.opt = opt

        absa_data_reader = ABSADataReader(data_dir=opt.data_dir,opt=opt)

        if opt.useBert:
            #bert
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
            embedding_matrix = []
        else:
            self.tokenizer = build_tokenizer(data_dir=opt.data_dir)
            embedding_matrix = build_embedding_matrix(opt.data_dir, self.tokenizer.word2idx, opt.embed_dim, opt.dataset)


        self.idx2tag, self.idx2polarity,self.idx2target ,self.id2express= absa_data_reader.reverse_tag_map, \
                                                          absa_data_reader.reverse_polarity_map, \
                                                          absa_data_reader.reverse_target_map,absa_data_reader.reverse_express_map

        self.model = opt.model_class(embedding_matrix=embedding_matrix,
                                     opt=opt,
                                     idx2tag=self.idx2tag,
                                     idx2polarity=self.idx2polarity,
                                     idx2target = self.idx2target,
                                     idx2express=self.id2express
                                    ).to(opt.device)
        print('loading model {0} ...'.format(opt.model_name))
        self.model.load_state_dict(torch.load(opt.state_dict_path, map_location=lambda storage, loc: storage))
        # switch model to evaluation mode
        self.model.eval()
        torch.autograd.set_grad_enabled(False)

    def evaluate(self, text, opt):
        # text_indices = self.tokenizer.encode(text,add_special_tokens=False)
        # text_mask = [1] * len(text_indices)
        if opt.useBert:
            out = self.tokenizer(text, add_special_tokens=False, padding=True)
            text_indices = out['input_ids']
            text_mask = out['attention_mask']
        else:
            for batch_text in text:
                text_indices = []
                text_mask = []
                if isinstance(batch_text,str):#一句话
                    text_indices.append(self.tokenizer.text_to_sequence(batch_text))
                    text_mask.append([1] * len(text_indices))
                elif isinstance(batch_text,list):

                    max_len = len(max(batch_text,key=len))
                    for t in batch_text:
                        t_indices = self.tokenizer.text_to_sequence(t)
                        text_indices.append(t_indices)
                        text_padding = [0] * (max_len - len(t))
                        text_mask.append(t_indices+text_padding)



        t_sample_batched = {
            'text_indices': torch.tensor(text_indices),
            'text_mask': torch.tensor(text_mask, dtype=torch.uint8),
        }
        with torch.no_grad():
            t_inputs = [t_sample_batched[col].to(self.opt.device) for col in self.opt.input_cols]
            infer_outpus = self.model(t_inputs,opt)
            t_ap_spans_pred, t_op_spans_pred, t_triplets_pred, t_senPolarity_pred,t_target_pred ,t_express_pred= self.model.inference(infer_outpus,
                                                                                                         t_inputs[0],
                                                                                                         t_inputs[1])
        t_senPolarity_pred = t_senPolarity_pred.cpu().numpy().tolist()

        return [t_ap_spans_pred, t_op_spans_pred, t_triplets_pred, t_senPolarity_pred,t_target_pred,t_express_pred]




if __name__ == '__main__':
    dataset = 'test'

    # set your trained models here
    model_state_dict_paths = {
        #'ote': 'state_dict/ote_' + dataset + '.pkl',
        'ote_Bert': 'state_dict/ote_Bert_' + 'test' + '.pkl',
        'ote_LSTM': 'state_dict/ote_LSTM_' + 'test' + '.pkl',
    }
    model_classes = {
        'ote_Bert': OTE,
        'ote_LSTM': OTE,
    }
    input_colses = {
        'ote_Bert': ['text_indices', 'text_mask'],
        'ote_LSTM': ['text_indices', 'text_mask'],
    }
    target_colses = {
        'ote_Bert': ['ap_indices', 'op_indices', 'triplet_indices', 'text_mask', 'sentece_polarity','target_indices'],
        'ote_LSTM': ['ap_indices', 'op_indices', 'triplet_indices', 'text_mask', 'sentece_polarity','target_indices'],
    }
    data_dirs = {
        'hotel': 'hotelDatasets/hotel',
        'test': 'hotelDatasets/test'
    }


    class Option(object):
        pass

    opt = Option()
    opt.useBert = False
    opt.dataset = dataset
    if opt.useBert:
        opt.model_name = 'ote_Bert'
    else:
        opt.model_name = 'ote_LSTM'
    opt.eval_cols = ['ap_spans', 'op_spans', 'triplets', 'sentece_polarity','targets']
    opt.model_class = model_classes[opt.model_name]
    opt.input_cols = input_colses[opt.model_name]
    opt.target_cols = target_colses[opt.model_name]
    opt.state_dict_path = model_state_dict_paths[opt.model_name]
    opt.embed_dim = 300
    opt.hidden_dim = 300
    opt.polarities_dim = 4
    opt.batch_size = 32
    opt.data_dir = data_dirs[opt.dataset]
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    inf = Inferer(opt)

    polarity_map = {0: 'N', 1: 'NEU', 2: 'NEG', 3: 'POS'}


    text = ['酒店距离中南商圈很近']
    pred_out = inf.evaluate(text,opt)

    print()










# -*- coding: utf-8 -*-

import os
import pickle
import torch
import torch.nn.functional as F
import argparse
from bucket_iterator import BucketIterator
from data_utils import ABSADataReader
from models import CMLA, HAST, OTE
from transformers import BertTokenizer


class Inferer:
    """A simple inference example"""

    def __init__(self, opt):
        self.opt = opt

        absa_data_reader = ABSADataReader(data_dir=opt.data_dir)
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
        self.idx2tag, self.idx2polarity,self.idx2target = absa_data_reader.reverse_tag_map, \
                                                          absa_data_reader.reverse_polarity_map, \
                                                          absa_data_reader.reverse_target_map
        self.model = opt.model_class(
                                    opt=opt,
                                    idx2tag=self.idx2tag,
                                    idx2polarity=self.idx2polarity,
                                    idx2target = self.idx2target
                                    ).to(opt.device)
        print('loading model {0} ...'.format(opt.model_name))
        self.model.load_state_dict(torch.load(opt.state_dict_path, map_location=lambda storage, loc: storage))
        # switch model to evaluation mode
        self.model.eval()
        torch.autograd.set_grad_enabled(False)

    def evaluate(self, text):
        # text_indices = self.tokenizer.encode(text,add_special_tokens=False)
        # text_mask = [1] * len(text_indices)
        out = self.tokenizer(text, add_special_tokens=False, padding=True)
        text_indices = out['input_ids']
        text_mask = out['attention_mask']

        t_sample_batched = {
            'text_indices': torch.tensor(text_indices),
            'text_mask': torch.tensor(text_mask, dtype=torch.uint8),
        }
        with torch.no_grad():
            t_inputs = [t_sample_batched[col].to(self.opt.device) for col in self.opt.input_cols]
            infer_outpus = self.model(t_inputs)
            t_ap_spans_pred, t_op_spans_pred, t_triplets_pred, t_senPolarity_pred,t_target_pred = self.model.inference(infer_outpus,
                                                                                                         t_inputs[0],
                                                                                                         t_inputs[1])
        t_senPolarity_pred = t_senPolarity_pred.cpu().numpy().tolist()

        return [t_ap_spans_pred, t_op_spans_pred, t_triplets_pred, t_senPolarity_pred,t_target_pred]


def get_text(input_path):
    text = []
    f_text = open(input_path, 'r', encoding='utf-8')
    lines = f_text.readlines()
    for line in lines:
        text.append(line.strip())
    f_text.close()
    return text


if __name__ == '__main__':
    dataset = 'hotel'
    # set your trained models here
    model_state_dict_paths = {
        #'ote': 'state_dict/ote_' + dataset + '.pkl',
        'ote': 'state_dict/ote_' + 'test' + '.pkl',
    }
    model_classes = {
        'ote': OTE,
    }
    input_colses = {
        'ote': ['text_indices', 'text_mask'],
    }
    target_colses = {
        'ote': ['ap_indices', 'op_indices', 'triplet_indices', 'text_mask', 'sentece_polarity','target_indices'],
    }
    data_dirs = {
        'laptop14': 'datasets/14lap',
        'rest14': 'datasets/14rest',
        'rest15': 'datasets/15rest',
        'rest16': 'datasets/16rest',
        'hotel': 'hotelDatasets/hotel'
    }


    class Option(object):
        pass


    opt = Option()
    opt.dataset = dataset
    opt.model_name = 'ote'
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


    text = ['朋友一行合肥打球,选择这家酒店,房间干净整洁,前台小妹妹很热情,退房时因天气热,还送了瓶水给我,感觉很好,下次有机会去,还会住这家酒店']
    pred_out = inf.evaluate(text)

    print()










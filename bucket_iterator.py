# -*- coding: utf-8 -*-

import math
import random
import torch
import numpy
from transformers import BertTokenizer

class BucketIterator(object):
    def __init__(self, data, batch_size, shuffle=True, sort=True):
        self.shuffle = shuffle
        self.sort = sort
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        self.batches = self.sort_and_pad(data, batch_size)
        self.batch_len = len(self.batches)


    def sort_and_pad(self, data, batch_size):
        num_batch = int(math.ceil(len(data) / batch_size))
        if self.sort:
            sorted_data = sorted(data, key=lambda x: len(x['text']))  # 按sequence_len排序
        else:
            sorted_data = data
        batches = []
        for i in range(num_batch):
            batches.append(self.pad_data(sorted_data[i * batch_size: (i + 1) * batch_size],self.tokenizer))
        return batches

    @staticmethod
    def pad_data(batch_data,tokenizer):

        batch_ap_indices = []
        batch_op_indices = []
        batch_triplet_indices = []
        batch_ap_spans = []
        batch_op_spans = []
        batch_triplets = []

        batch_text_indices,batch_text_mask  = BucketIterator.tokenize(batch_data,tokenizer)
        max_len = len(batch_text_indices[0])

        for item in batch_data:
            text, ap_indices, op_indices,  ap_spans, op_spans, triplets ,triplet_indices= \
                item['text'], item['ap_indices'], item['op_indices'],  \
                item['ap_spans'], item['op_spans'], item['triplets'],item['triplet_indices']

            ap_indices = [-100] + ap_indices + [-100]
            op_indices = [-100] + op_indices + [-100]

            # 0-padding because 0 stands for 'O'
            text_padding = [-100] * (max_len - len(ap_indices))

            batch_ap_indices.append(ap_indices + text_padding)
            batch_op_indices.append(op_indices + text_padding)

            batch_triplet_indices.append(
                numpy.pad(triplet_indices, ((1, max_len - len(ap_indices)+1 ), (1, max_len - len(ap_indices)+1 )),
                          'constant',constant_values=(-100,-100)))
            batch_ap_spans.append(ap_spans)
            batch_op_spans.append(op_spans)
            batch_triplets.append(triplets)


        return {
            'text_indices': torch.tensor(batch_text_indices),
            'text_mask': torch.tensor(batch_text_mask, dtype=torch.bool),
            'ap_indices': torch.tensor(batch_ap_indices),  # 带ap标签的序列
            'op_indices': torch.tensor(batch_op_indices),
            'triplet_indices': torch.tensor(batch_triplet_indices),
            'ap_spans': batch_ap_spans,  # ap的区间
            'op_spans': batch_op_spans,
            'triplets': batch_triplets,

        }
    #@staticmethod
    def tokenize(batch_data,tokenizer):
        batch_text = []
        for item in batch_data:
            text = item['text']
            batch_text.append(text)

        output = tokenizer(batch_text,is_split_into_words=True,padding=True)
        text_indices = output['input_ids']
        text_mask = output['attention_mask']

        return text_indices,text_mask

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.batches)
        for idx in range(self.batch_len):
            yield self.batches[idx]

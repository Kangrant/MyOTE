# -*- coding: utf-8 -*-

import os
import pickle
import numpy as np
from transformers import BertTokenizer


class ABSADataReader(object):
    def __init__(self, data_dir):

        self.tag_map, self.reverse_tag_map, self.target_map, self.reverse_target_map = self._get_tag_map()
        self.polarity_map = {'N': 0, 'NEU': 1, 'NEG': 2, 'POS': 3}  # NO_RELATION is 0
        self.reverse_polarity_map = {v: k for k, v in self.polarity_map.items()}
        self.data_dir = data_dir
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

    def get_train(self, tokenizer):
        return self._create_dataset('train', tokenizer)

    def get_dev(self, tokenizer):
        return self._create_dataset('dev', tokenizer)

    def get_test(self, tokenizer):
        return self._create_dataset('test', tokenizer)

    @staticmethod
    def _get_tag_map():
        tag_list = ['O', 'B', 'I']
        tag_map = {tag: i for i, tag in enumerate(tag_list)}
        reverse_tag_map = {i: tag for i, tag in enumerate(tag_list)}

        target_list = ['0', '11', '12', '13', '14', '21', '22', '23', '24', '25', '26', '27', '28', '31', '32', '33',
                       '34', '35', '36', '41', '42', '43', '44', '45', '46', '47', '48', '49', '410', '411', '51', '52',
                       '53', '54']
        target_map = {tag: i for i, tag in enumerate(target_list)}
        reverse_target_map = {i: tag for i, tag in enumerate(target_list)}
        return tag_map, reverse_tag_map, target_map, reverse_target_map

    def _create_dataset(self, set_type, tokenizer):
        all_data = []

        filename = os.path.join(self.data_dir, '%s.all' % set_type)
        fp = open(filename, 'r', encoding='utf-8')
        lines = fp.readlines()
        fp.close()

        for i in range(0, len(lines), 5):
            text = lines[i].strip()
            text = text.split(' ')
            # text = text.split(' ')
            ap_tags, op_tags, ap_span = lines[i + 1].strip().split('####')
            ap_tags, op_tags, ap_spans = eval(ap_tags), eval(op_tags), eval(ap_span)
            pairs = lines[i + 2].strip().split(';')
            sentece_polarity_str = lines[i + 3].strip()
            sentece_polarity_indices = self.polarity_map[sentece_polarity_str]
            target_pairs = eval(lines[i + 4].strip())

            text_indices = self.tokenizer.encode(text, add_special_tokens=False, is_split_into_words=True)
            seq_len = len(text_indices)
            # ap_spans = []
            op_spans = []
            triplets = []
            targets = []
            triplet_indices = np.zeros((seq_len, seq_len), dtype=np.int64)
            target_indices = np.zeros((seq_len, seq_len), dtype=np.int64)
            for pair in pairs:  # pair [(ap_beg, ap_end),(op_beg, op_end),polarity]
                pair = eval(pair)
                ap_beg, ap_end = pair[0]
                op_beg, op_end = pair[1]
                polarity_str = pair[2]
                polarity = self.polarity_map[polarity_str]
                triplet_indices[ap_end][op_end] = polarity
                if (op_beg, op_end) not in op_spans:
                    op_spans.append((op_beg, op_end))
                triplets.append((ap_beg, ap_end, op_beg, op_end, polarity))

            for target_pair in target_pairs:
                beg,end,aspect,target_label = target_pair
                target_tag = self.target_map[target_label]
                target_indices[end][end] = target_tag
                targets.append((beg,end,target_label))


            ap_indices = [self.tag_map[tag] for tag in ap_tags]
            op_indices = [self.tag_map[tag] for tag in op_tags]
            # data 格式
            data = {
                'text_indices': text_indices,
                'ap_indices': ap_indices,
                'op_indices': op_indices,
                'ap_spans': ap_spans,
                'op_spans': op_spans,
                'triplets': triplets,
                'triplet_indices': triplet_indices,
                'sentece_polarity': sentece_polarity_indices,
                'target_indices': target_indices,
                'targets': targets
            }
            all_data.append(data)

        return all_data


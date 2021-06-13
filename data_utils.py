# -*- coding: utf-8 -*-

import os
import pickle
import numpy as np
from transformers import BertTokenizer


class ABSADataReader(object):
    def __init__(self, data_dir,opt):
        self.opt = opt
        self.tag_map, self.reverse_tag_map, self.target_map, self.reverse_target_map ,\
                                            self.express_map, self.reverse_express_map= self._get_tag_map()
        self.polarity_map = {'N': 0, 'NEU': 1, 'NEG': 2, 'POS': 3}  # NO_RELATION is 0
        self.reverse_polarity_map = {v: k for k, v in self.polarity_map.items()}
        self.data_dir = data_dir


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

        express_list = ['O', 'B-exp-fac', 'I-exp-fac', 'E-exp-fac','B-exp-sug', 'I-exp-sug', 'E-exp-sug','B-exp-con', 'I-exp-con', 'E-exp-con','B-exp-rea', 'I-exp-rea','E-exp-rea','S-exp-fac','S-exp-sug','S-exp-con','S-exp-rea']
        express_map = {express: i for i, express in enumerate(express_list)}
        reverse_express_map = {i: express for i, express in enumerate(express_list)}

        return tag_map, reverse_tag_map, target_map, reverse_target_map, express_map, reverse_express_map

    def _create_dataset(self, set_type, tokenizer):
        all_data = []

        filename = os.path.join(self.data_dir, '%s.all' % set_type)
        fp = open(filename, 'r', encoding='utf-8')
        lines = fp.readlines()
        fp.close()

        for i in range(0, len(lines), 6):
            text = lines[i].strip()
            # text = text.split(' ')
            ap_tags, op_tags, ap_span = lines[i + 1].strip().split('####')
            ap_tags, op_tags, ap_spans = eval(ap_tags), eval(op_tags), eval(ap_span)
            pairs = lines[i + 2].strip().split(';')
            sentece_polarity_str = lines[i + 3].strip()
            sentece_polarity_indices = self.polarity_map[sentece_polarity_str]
            target_pairs = eval(lines[i + 4].strip())

            if self.opt.useBert:
                text = text.split(' ')
                text_indices = tokenizer.encode(text, add_special_tokens=False, is_split_into_words=True)
            else:
                text_indices = tokenizer.text_to_sequence(text)
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
                beg, end, aspect, target_label = target_pair
                target_tag = self.target_map[target_label]
                target_indices[end][end] = target_tag
                targets.append((beg, end, target_label))

            ap_indices = [self.tag_map[tag] for tag in ap_tags]
            op_indices = [self.tag_map[tag] for tag in op_tags]

            express_label = eval(lines[i+5].strip())
            #express_label to span
            express_span = []
            beg, end = -1, -1
            pre_type = None
            for i, label in enumerate(express_label):
                if label !='O':
                    tag,_,type = label.split('-')
                    if tag == 'S':
                        # start position and end position are kept same for the singleton
                        express_span.append((i, i))
                    elif tag == 'B':
                        beg = i
                        pre_type = type
                    elif tag == 'I' and pre_type!=type:
                        beg = -1
                        pre_type = None
                    elif tag == 'E' and pre_type == type:
                        end = i
                        if beg>-1 and end > beg:
                            # only valid chunk is acceptable
                            express_span.append((beg, end))
                        beg,end = -1,-1
                        pre_type = None

            express_indices = [self.express_map[tag] for tag in express_label]
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
                'targets': targets,
                'express_label':express_span,
                'express_indices':express_indices,
            }
            all_data.append(data)

        return all_data


def load_word_vec(path, word2idx=None, embed_dim=300):
    fin = open(path, 'r', encoding='utf-8', newline='\n', errors='ignore')
    word_vec = {}
    for line in fin:
        tokens = line.rstrip().split()
        if len(tokens) != 2:
            word, vec = ' '.join(tokens[:-embed_dim]), tokens[-embed_dim:]
            if word in word2idx.keys():
                word_vec[word] = np.asarray(vec, dtype='float32')
    return word_vec

def build_embedding_matrix(data_dir, word2idx, embed_dim, type):
    embedding_matrix_file_name = '{0}_{1}_embedding_matrix.pkl'.format(str(embed_dim), type)
    if os.path.exists(os.path.join(data_dir, embedding_matrix_file_name)):
        print('>>> loading embedding matrix:', embedding_matrix_file_name)
        embedding_matrix = pickle.load(open(os.path.join(data_dir, embedding_matrix_file_name), 'rb'))
    else:
        print('>>> loading word vectors ...')
        # words not found in embedding index will be randomly initialized.
        embedding_matrix = np.random.uniform(-1/np.sqrt(embed_dim), 1/np.sqrt(embed_dim), (len(word2idx), embed_dim))
        # <pad>
        embedding_matrix[0, :] = np.zeros((1, embed_dim))
        fname = './pretrain_chinese_word/sgns.wiki.word'
        word_vec = load_word_vec(fname, word2idx=word2idx, embed_dim=embed_dim)
        print('>>> building embedding matrix:', embedding_matrix_file_name)
        for word, i in word2idx.items():
            vec = word_vec.get(word)
            if vec is not None:
                embedding_matrix[i] = vec
        pickle.dump(embedding_matrix, open(os.path.join(data_dir, embedding_matrix_file_name), 'wb'))
    return embedding_matrix

class Tokenizer(object):
    def __init__(self, word2idx=None):
        if word2idx is None:
            self.word2idx = {}
            self.idx2word = {}
            self.idx = 0
            self.word2idx['<pad>'] = self.idx
            self.idx2word[self.idx] = '<pad>'
            self.idx += 1
            self.word2idx['<unk>'] = self.idx
            self.idx2word[self.idx] = '<unk>'
            self.idx += 1
        else:
            self.word2idx = word2idx
            self.idx2word = {v:k for k,v in word2idx.items()}

    def fit_on_text(self, text):
        """
        将词添加到词表中
        """
        text = text.lower()
        words = text.split()
        for word in words:
            if word not in self.word2idx:
                self.word2idx[word] = self.idx
                self.idx2word[self.idx] = word
                self.idx += 1

    def text_to_sequence(self, text):
        """
        将text转换为id表示
        """
        # text = text.lower()
        words = text.split()
        unknownidx = 1
        sequence = [self.word2idx[w] if w in self.word2idx else unknownidx for w in words]
        if len(sequence) == 0:
            sequence = [0]
        return sequence

def build_tokenizer(data_dir):
    if os.path.exists(os.path.join(data_dir, 'word2idx.pkl')):
        print('>>> loading {0} tokenizer...'.format(data_dir))
        with open(os.path.join(data_dir, 'word2idx.pkl'), 'rb') as f:
            word2idx = pickle.load(f)
            tokenizer = Tokenizer(word2idx=word2idx)
    else:
        filenames = [os.path.join(data_dir, '%s.all' % set_type) for set_type in ['train', 'dev']]
        all_text = ''
        for filename in filenames:
            with open(filename, 'r', encoding='utf-8') as fp:
                lines = fp.readlines()
                for i in range(0,len(lines),6):
                    text = lines[i].strip()
                    all_text += (text + ' ')
        tokenizer = Tokenizer()
        tokenizer.fit_on_text(all_text)
        print('>>> saving {0} tokenizer...'.format(data_dir))
        with open(os.path.join(data_dir, 'word2idx.pkl'), 'wb') as f:
            pickle.dump(tokenizer.word2idx, f)

    return tokenizer

if __name__ == '__main__':
    path = r'test_word.txt'
    load_word_vec(path)
# from transformers import BertTokenizer
#
# import torch
#
# t = BertTokenizer.from_pretrained('bert-base-chinese')
#
# res = t.encode(['\U0001f92b', '景'],add_special_tokens=False,is_split_into_words=True)
# print(res)

import  string
punc = string.punctuation#标点符号
punc += '、'
print(punc)
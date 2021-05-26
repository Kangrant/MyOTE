# from transformers import BertTokenizer
#
import torch
#
# t = BertTokenizer.from_pretrained('bert-base-chinese')
#
# res = t.encode(['\U0001f92b', '景'],add_special_tokens=False,is_split_into_words=True)
# print(res)


# a = torch.tensor([[1.,2,3],[4,5,6],[7,8,9]])
# b= a.mean(dim=1)
# print(b)
import numpy as np
a = np.array([1,2,3,4])
b= np.array([1,2,3,5])
print(np.sum(a==b))
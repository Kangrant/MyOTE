"""

"""
import string

def prepro(filename):
    """
    return:

    text
    #(暂无)mix_label : one sequence contains ap_label and op_label
    (ap_label,op_label) : (sequence only contain ap_label,sequence only contain op_label)
    triplets

    """
    print(filename)
    fin = open(filename, 'r', encoding='utf-8')
    fout = open(filename.split('.')[0] + '.pair', 'w', encoding='utf-8')

    polarity_tag = {'0':'NEG','1':'NEU','2':'POS'}
    punc = string.punctuation#标点符号
    punc += '、。，！？；：“”‘’'
    text_all = []  # 每个元素代表每个sentence的text
    label_all = [] #  每个元素代表每个sentence的标签序列
    triplets_all = []  # 每个元素表示每个sentence中的所有三元组的列表
    sentence_polarity = []  # 每句话的情感极性

    words, labels = [], []
    for line in fin:
        content = line.strip().split()
        if len(content) == 2: #标签
            word, label = content
            if ('\u4e00' <= word <= '\u9fff' or word.isalnum() or word in punc):
                words.append(word)
                labels.append(label)

        elif len(content) == 1: #情感极性
            polarity_id = content[0].replace('-','')
            polarity = polarity_tag[polarity_id]
            sentence_polarity.append(polarity)

        else:# 遇到换行,一个sentence结束
            text = ' '.join(words)
            text_all.append(text)
            ap_label,op_label,triplet = process_label(labels)
            label_all.append((ap_label,op_label))
            triplets_all.append(triplet)
            words, labels = [], []

    assert len(text_all) == len(triplets_all) == len(sentence_polarity)

    for i in range(len(text_all)):
        if(triplets_all[i]):#有些标注没有三元组。这里只输出能构成三元组的
            fout.write(text_all[i]+'\n')
            ap_label,op_label = label_all[i]
            fout.write(str(ap_label)+'####'+str(op_label)+'\n')
            for j in range(len(triplets_all[i])):
                trp=str(triplets_all[i][j])
                if(j!=len(triplets_all[i])-1):
                    fout.write(trp+';')
                else:
                    fout.write(trp+'\n')




    fin.close()
    fout.close()


def process_label(labels):
    """
    将sentence中的标签序列处理为三元组形式
    """
    polarity_tag = {'0':'NEG','1':'NEU','2':'POS'}

    triplets = []
    aspect, opinion = [], []
    ap_label = ['O'] * len(labels)
    op_label =['O'] * len(labels)
    beg, end = 0, 0
    for id, label in enumerate(labels):  # label:  B-a0-0    I-e0-a-a1
        if label == 'O':
            continue
        label_info = label.split('-')
        length = len(label_info)
        # length ==3 是aspect标签
        if length == 3:
            label_type, label_index, polarity_id = label_info
            polarity = polarity_tag[polarity_id]
            if label_type == 'B':
                beg = id
                ap_label[id] = 'B'
            elif label_type == 'I':
                ap_label[id] = 'I'
                continue
            elif label_type == 'S':
                ap_label[id] = 'B'
                beg, end = id, id
                aspect.append([beg, end, label_index, polarity])
                eg, end = 0, 0
            else:
                end = id
                ap_label[id] = 'I'
                aspect.append([beg, end, label_index, polarity])
                beg, end = 0, 0

        # length >= 4 是opinion标签
        elif length >= 4:
            label_type, label_index, polarity_id = label_info[:3]
            aspect_index  = label_info[3:]
            polarity = polarity_tag[polarity_id]
            if label_type == 'B':
                op_label[id] = 'B'
                beg = id
            elif label_type == 'I':
                op_label[id] = 'I'
                continue
            elif label_type == 'S':
                op_label[id] = 'B'
                beg, end = id, id
                for index in aspect_index:
                    opinion.append([beg, end, label_index, polarity, index])
                beg, end = 0, 0
            else:
                end = id
                op_label[id] = 'I'
                for index in aspect_index:
                    opinion.append([beg, end, label_index, polarity, index])
                beg, end = 0, 0

    assert len(labels) == len(ap_label) == len(op_label)
    # print(aspect)
    # print(opinion)

    # 对aspect和opinion进行配对

    for ap in aspect:
        ap_beg, ap_end, ap_index, ap_polarity = ap
        for op in opinion:
            op_beg, op_end, _, op_polarity, op_index = op

            if ap_index == op_index:
                #assert ap_polarity == op_polarity 以aspect的极性标注为准
                ap_span = [ap_beg,ap_end]
                op_span = [op_beg,op_end]
                triplet = [ap_span,op_span,ap_polarity]
                triplets.append(triplet)

    #print(triplets)
    return ap_label,op_label,triplets

def bio2bieos(tags):
    new_tags = []
    for i, tag in enumerate(tags):
        if tag == 'O':
            new_tags.append(tag)
        # elif tag == 'S':
        #     new_tags.append(tag)
        elif tag[0] == 'B':
            if i + 1 != len(tags) and \
                    tags[i + 1][0] == 'I':
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace('B', 'S'))
        elif tag[0] == 'I':
            if  (i + 1 < len(tags) and tags[i + 1][0] == 'I')  :
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace('I', 'E'))
        else:
            raise Exception('Invalid IOB format!')
    return new_tags

def bieos2span(tags, tp='-AP'): # tp = '', '-AP', or '-OP'
    spans = []
    beg, end = -1, -1
    for i, tag in enumerate(tags):
        if tag == 'S'+tp:
            # start position and end position are kept same for the singleton
            spans.append((i, i))
        elif tag == 'B'+tp:
            beg = i
        elif tag == 'E'+tp:
            end = i
            if end > beg:
                # only valid chunk is acceptable
                spans.append((beg, end))
    return spans

if __name__ == '__main__':
    filename = 'hotel/dev.txt'
    # filename = 'hotel/train.txt'
    #filename = 'test/dev.txt'
    # filename = 'test/train.txt'
    prepro(filename=filename)

    # with open('./test/train.pair','r',encoding='utf-8') as f:
    #     lines = f.readlines()
    #     for i in range(0,len(lines),3):
    #         text = lines[i]
    #         ap_labels,op_labels = lines[i+1].strip().split('####')
    #         triplets = lines[i+2]
    #         break
    # ap_labels,op_labels = eval(ap_labels),eval(op_labels)
    # print(op_labels)
    # new_op = bio2bieos(op_labels)
    # print(new_op)
    # new_op_span = bieos2span(new_op,tp='')
    # print(new_op_span)



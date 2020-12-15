import re
import os
import json

import torch
from torch.utils.data import Dataset
import transformers
from tqdm import tqdm

here = os.path.dirname(os.path.abspath(__file__))  # 当前文件的目录


def get_not_need_label(file):
    """
    删除数据集中有，但是relation.txt不需要的标签
    """
    not_need_label_list = []
    with open(file, 'r+', encoding='utf-8') as f_in:
        not_need_label_list = re.split(r'\n', f_in.read().strip())
    return not_need_label_list


def get_special_type(file):
    """
    将special_type_file中的obj_type，统一使用[unused]系列替换，和不使用类型信息的标记一致
    """
    special_type = []
    with open(file, 'r+', encoding='utf-8') as f:
        special_type = re.split(r'\n', f.read().strip())
    return special_type


def get_additional_tokens(additional_file_path):
    """
    将守卫在实体两边的实体类型标签作为特殊字符加入到BERT
    :param file_name: additional_special_tokens.txt
    :return: ['<H-PERSON>', '</H-PERSON>', '<T-PERSON>', '</T-PERSON>']
    """
    special_tokens_list = []
    with open(additional_file_path, 'r+', encoding='utf-8') as f:
        for line in f:
            special_tokens_list.append(line.strip())
    return special_tokens_list


class Mytokenizer(object):
    """
    因为我们想要返回在分词后的两个实体的位置，以及施加entity_mask的策略，所以自己写一个tokenizer
    """

    def __init__(self, additional_file_path=None, pretrained_model_path=None, mask_entity=False,
                 is_add_entity_type=False, special_type_file=None):

        self.additional_tokens = get_additional_tokens(additional_file_path)
        self.pretrained_model_path = pretrained_model_path
        self.mask_entity = mask_entity
        self.bert_tokenizer = transformers.BertTokenizer.from_pretrained(
            self.pretrained_model_path)
        self.is_add_entity_type = is_add_entity_type
        self.additional_tokens = self.additional_tokens
        self.bert_tokenizer.add_special_tokens(
            {'additional_special_tokens': self.additional_tokens})  # 添加特殊标记
        self.special_type = get_special_type(
            special_type_file)  # 需要统一转换的18个NER外的标记

    def tokenize(self, item):
        """
        对一个样本进行分词（添加一些special token）
        :param item: one sample
        :param is_add_entity_type: whether add entity type information
        :return: tokens; position range for head entity in tokens ; position range for tail entity in tokens
        """
        sentence = item['text']
        # 下面这个是分词前的位置和
        pos_h = item['h']['pos']  # header_position = subject  数据集中给出的范围是左闭有开的
        pos_t = item['t']['pos']  # tailer_position = object
        h_type = ''
        t_type = ''
        if 'type' in item['h']:
            h_type = item['h']['type']
        if 'type' in item['t']:
            t_type = item['t']['type']

        # 找到哪个实体的位置更靠前，方便我们进行实体的mask
        if pos_h[0] < pos_t[0]:  # header 更靠前
            pos_min = pos_h
            pos_max = pos_t
        else:  # tailer 更靠前
            pos_min = pos_t
            pos_max = pos_h

        # 把这个text分为四段，方便对ent0和ent1进行替换
        sent0 = self.bert_tokenizer.tokenize(sentence[0:pos_min[0]])
        ent0 = self.bert_tokenizer.tokenize(sentence[pos_min[0]:pos_min[1]])
        sent1 = self.bert_tokenizer.tokenize(sentence[pos_min[1]:pos_max[0]])
        ent1 = self.bert_tokenizer.tokenize(sentence[pos_max[0]:pos_max[1]])
        sent2 = self.bert_tokenizer.tokenize(sentence[pos_max[1]:])

        if self.mask_entity:  # 这里不引入类型信息，后面再左右哨兵标记那里引入
            # 默认用特殊字符引入先验知识，作为meta-tokens不会被tokenizer分开，更好注意到特定部分
            if pos_h < pos_t:
                ent0 = '[unused5]'  # head
                ent1 = '[unused6]'  # tail
            else:
                ent1 = '[unused5]'  # head
                ent0 = '[unused6]'  # tail

        tokens = sent0 + ent0 + sent1 + ent1 + sent2  # all tokens

        # 得到两个实体**分词后**在tokens中的位置，因为加了mask后和不加mask不太一样
        if pos_h < pos_t:  # head在前
            pos_head = [
                len(sent0),
                len(sent0) + len(ent0)
            ]  # 左闭右开
            pos_tail = [
                len(sent0) + len(ent0) + len(sent1),
                len(sent0) + len(ent0) + len(sent1) + len(ent1)
            ]
        else:  # tail在前
            pos_tail = [
                len(sent0),
                len(sent0) + len(ent0)
            ]  # 左闭右开
            pos_head = [
                len(sent0) + len(ent0) + len(sent1),
                len(sent0) + len(ent0) + len(sent1) + len(ent1)
            ]

        re_tokens = ['[CLS]']
        pos1 = [0, 0]  # 用于**记录加完tag后**，head实体的开始结束位置，也是左开右闭
        pos2 = [0, 0]  # 用于记录加完tag后，tail实体的开始结束位置

        # 已经添加过了additional_special_tokens的标记，也可以不加tokenize
        if (h_type is not '') and (t_type is not '') and self.is_add_entity_type is True:  # 我们自己加的，在词表中没有的
            before_head = self.bert_tokenizer.tokenize(
                '<' + 'H-' + h_type + '>')
            before_tail = self.bert_tokenizer.tokenize(
                '<' + 'T-' + t_type + '>')
            after_head = self.bert_tokenizer.tokenize(
                '<' + '/' + 'H-' + h_type + '>')
            after_tail = self.bert_tokenizer.tokenize(
                '<' + '/' + 'T-' + t_type + '>')
        else:
            before_head = '[unused1]'
            before_tail = '[unused2]'
            after_head = '[unused3]'
            after_tail = '[unused4]'

        # 处理特殊标记，eg，JOB、NP
        if h_type in self.special_type:
            before_head = '[unused1]'
            after_head = '[unused3]'

        if t_type in self.special_type:
            before_tail = '[unused2]'
            after_tail = '[unused4]'

        # 用类型标签把实体包起来 [unused1] head [unused3]; [unused2] tail [unused4]
        for cur_pos in range(len(tokens)):
            token = tokens[cur_pos]
            if cur_pos == pos_head[0]:  # before head
                pos1[0] = len(re_tokens)  # 算上了标签
                re_tokens.extend(before_head)

            if cur_pos == pos_tail[0]:  # before tail
                pos2[0] = len(re_tokens)  # 算上了标签
                re_tokens.extend(before_tail)

            # 加入这个词本身 放在中间是因为可能pos_head[0] == pos_head[1]
            re_tokens.append(token)

            if cur_pos == pos_head[1] - 1:  # after head，-1 是因为是左开右闭的区间
                re_tokens.extend(after_head)
                pos1[1] = len(re_tokens)  # 算上了标签

            if cur_pos == pos_tail[1] - 1:  # after tail
                re_tokens.extend(after_tail)
                pos2[1] = len(re_tokens)  # 算上了标签

        re_tokens.append('[SEP]')
        # 返回的时候删除首尾的两个special token，因为后面encode的时候还会加上的
        return re_tokens[1:-1], pos1, pos2


def convert_pos_to_mask(e_pos, max_len=128):
    """
    将实体在tokens中的起始范围，变成max_len长度的list，该实体的起止的范围设置为1，其余为0
    :param pos: [start_pos, end_pos] 左开右闭
    :param max_len: len(tokens)
    :return: mask_list
    """
    e_pos_mask = [0] * max_len
    for i in range(e_pos[0], e_pos[1]):  # i range in [e_pos[0], e_pos[1]-1)
        e_pos_mask[i] = 1
    return e_pos_mask


def read_data(data_file_path, not_need_label_file, tokenizer=None, max_len=128):
    """
    read data from file used tokenizer
    :param data_file_path: 源文件的路径
    :param tokenizer: 我们刚刚定义好的tokenizer
    :param max_len: text允许的最大程度
    :return: tokens_list, e1_mask_list, e2_mask_list, labels
    """
    tokens_list = []
    e1_mask_list = []
    e2_mask_list = []
    labels = []

    not_need_label = get_not_need_label(not_need_label_file)

    with open(data_file_path, 'r+', encoding='utf-8') as f:
        for line in tqdm(f, desc='data'):
            line = line.strip()
            item = json.loads(line)  # (将string转换为dict)

            if item['relation'] in not_need_label:
                continue

            tokens, pos_e1, pos_e2 = tokenizer.tokenize(item=item)
            # 没有超出我们限制的长度
            if pos_e1[0] < max_len - 1 and pos_e1[1] < max_len \
                    and pos_e2[0] < max_len - 1 and pos_e2[1] < max_len:
                tokens_list.append(tokens)
                e1_mask = convert_pos_to_mask(pos_e1)
                e2_mask = convert_pos_to_mask((pos_e2))
                e1_mask_list.append(e1_mask)
                e2_mask_list.append(e2_mask)
                label = item['relation']
                labels.append(label)
        return tokens_list, e1_mask_list, e2_mask_list, labels


def get_label2idx(file):
    """
    将我们label文件中的label和id对应，根据label查id
    :param file: relation.txt
    :return: a dict, key-label, value-idx
    """
    with open(file, 'r+', encoding='utf-8') as f_in:
        labelset = re.split(r'\n', f_in.read().strip())
    dic = {}
    for idx, label in enumerate(labelset):
        dic[label] = idx
    return dic


def get_idx2label(file):
    """
    将我们label文件中的label和id对应，根据id查label
    :param file: relation.txt
    :return: a dict, key-idx, value-label
    """
    with open(file, 'r+', encoding='utf-8') as f_in:
        labelset = re.split(r'\n', f_in.read().strip())
    dic = {}
    for idx, label in enumerate(labelset):
        dic[idx] = label
    return dic


def save_checkpoint(checkpoint_dict, file):
    with open(file, 'w', encoding='utf-8') as f_out:
        json.dump(checkpoint_dict, f_out, ensure_ascii=False, indent=2)


def load_checkpoint(file):
    with open(file, 'r', encoding='utf-8') as f_in:
        checkpoint_dict = json.load(f_in)
    return checkpoint_dict


class WikiData(Dataset):
    def __init__(self, data_file_path, not_need_label_file, labels_path, additional_file_path, special_type_file, pretrained_model_path=None, max_len=128, is_add_entity_type=False):
        self.data_file_path = data_file_path
        self.labels_path = labels_path
        self.pretrained_model_path = pretrained_model_path or 'hfl/chinese-bert-wwm'
        self.max_len = max_len
        self.additional_file_path = additional_file_path
        # 增加实体类型
        self.is_add_entity_type = is_add_entity_type
        print("是否使用实体类型信息:{}".format(is_add_entity_type))
        self.tokenizer = Mytokenizer(pretrained_model_path=pretrained_model_path, mask_entity=False,
                                     is_add_entity_type=is_add_entity_type, additional_file_path=self.additional_file_path, special_type_file=special_type_file)
        self.tokens_list, self.e1_mask_list, self.e2_mask_list, self.labels = read_data(data_file_path=data_file_path,
                                                                                        not_need_label_file=not_need_label_file,
                                                                                        tokenizer=self.tokenizer,
                                                                                        max_len=self.max_len)
        self.label2idx = get_label2idx(self.labels_path)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()  # 如果是一个tensor类型，变为list
        sample_tokens = self.tokens_list[idx]
        sample_e1_mask = self.e1_mask_list[idx]
        sample_e2_mask = self.e2_mask_list[idx]
        sample_label = self.labels[idx]
        # bert的tokenizer会把放到一个list中以逗号分隔的编码为一个序列，所以我们传进入分词好的结果没有影响，只会被加上[CLS]和[SEP]
        encoded = self.tokenizer.bert_tokenizer.encode_plus(sample_tokens, max_length=self.max_len,
                                                            pad_to_max_length=True, truncation=True)
        # encoded['input_ids'][1:-1] 和 sample_tokens 相比，加了[PAD]
        sample_token_ids = encoded['input_ids']
        sample_token_type_ids = encoded['token_type_ids']
        sample_attention_mask = encoded['attention_mask']
        sample_label_id = self.label2idx[sample_label]  # 把实际的label，转换为idx

        sample = {
            'token_ids': torch.tensor(sample_token_ids),
            'token_type_ids': torch.tensor(sample_token_type_ids),
            'attention_mask': torch.tensor(sample_attention_mask),
            'e1_mask': torch.tensor(sample_e1_mask),
            'e2_mask': torch.tensor(sample_e2_mask),
            'label_id': torch.tensor(sample_label_id)
        }

        return sample

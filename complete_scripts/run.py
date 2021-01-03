import json
from pprint import pprint
import re
import stanza

from .hparams import hparams

zh_nlp = stanza.Pipeline(
    'zh', verbose=False, use_gpu=False)


def run(hparams, subject='伊丽莎白'):
    """ 对 hparams.documents_path 中的所有documents，进行句子切分，并获取三元组
    """
    documents_path = hparams.documents_path
    docs = json.load(open(documents_path, 'r+', encoding='utf-8'))['docs']
    all_sent = []
    for doc in docs:
        doc.replace('\n', '')
        sents = doc.split('。')  # 切割的句子
        for s in sents:
            if subject in s:
                all_sent.append(s)  # 加入包含主实体名的句子
    global zh_nlp
    res_all = []
    for sentence in all_sent:
        res = deal_one_sent(sentence=sentence, subject=subject, pad='')
        res_all.extend(res)
    return res_all


def deal_one_sent(sentence, subject, subj_type='PERSON', pad='', need_merged_pos_type=['PROPN'], not_need_merged_pos_type=['NOUN']):
    """
    根据NER和POS的结果，针对中文进行提取一个句子中的候选尾巴实体，
    注，只针对中文，因为在合并NP的时候是直接add的，英文需要加入空格
    注，因为 "年 NOUN、伯父 NOUN" 问题，只针对连续的PROPN进行合并，NOUN还是分开加入，不进行合并操作
    """
    sents = zh_nlp(sentence)

    only_object = [subject]  # 用于 object 去重，先把subject加入

    tokens = []  # 句子分词后的token
    poss = []  # 每个token对应的upos标记
    ners = []  # 所有抽出来的实体，[{subject, object, obj_type}]
    filtered_np = []  # 我们根据词性过滤的NP
    all_obj = []  # ners+filtered_np

    for sent in sents.sentences:
        for word in sent.words:
            # print(word)
            text = word.text
            upos = word.upos
            tokens.append(text)
            poss.append(upos)
            if text not in only_object and upos in not_need_merged_pos_type and text not in subject:  # 没有出现过的单个的NOUN, 也不是subject中的一部分
                only_object.append(text)  # 加入抽出的所有实体
                item = {}
                item['text'] = sentence
                item['subject'] = subject
                item['subj_type'] = subj_type
                item['object'] = text
                item['obj_type'] = upos
                filtered_np.append(item)

        for ent in sent.ents:
            if ent.text not in only_object and ent.text not in subject:
                only_object.append(ent.text)  # 加入抽出的所有实体
                item = {}
                item['text'] = sentence
                item['subject'] = subject
                item['subj_type'] = subj_type
                item['object'] = ent.text
                item['obj_type'] = ent.type
                ners.append(item)

    last_pos = ''
    complete_np = ''
    merged_pos = []  # 句子中所有的词性短语，将相同pos标记的token进行了合并
    for idx in range(len(tokens)):
        token = tokens[idx]
        upos = poss[idx]
        # print(token, upos)
        if last_pos == '':
            complete_np = complete_np + pad + token
        elif last_pos == upos:  # 和上一个token的pos标记相同
            complete_np = complete_np + pad + token
        elif last_pos != upos:  # 和上一个token的pos标记不同
            item = {}
            item['text'] = sentence
            item['subject'] = subject
            item['subj_type'] = subj_type
            item['object'] = complete_np
            item['obj_type'] = last_pos
            merged_pos.append(item)
            complete_np = token
        last_pos = upos

    for item in merged_pos:
        # 之前没有出现过，并且属于我们需要的词性标记
        if item['object'] not in only_object and item['obj_type'] in need_merged_pos_type and item['object'] not in subject:
            only_object.append(item['object'])
            filtered_np.append(item)

    all_obj.extend(ners)
    all_obj.extend(filtered_np)
    return all_obj


def deal_one_sent_only_ner(sentence, subject, subj_type='PERSON', pad=''):
    """
    根据NER的结果，针对中文进行提取一个句子中的候选尾巴实体
    """
    global zh_nlp
    sents = zh_nlp(sentence)

    only_object = [subject]  # 去重，先把subject加入

    ners = []  # 所有抽出来的实体，[{subject, object, obj_type}]

    for sent in sents.sentences:
        for ent in sent.ents:
            if ent.text not in only_object and ent.text not in subject:
                only_object.append(ent.text)  # 加入抽出的所有实体
                item = {}
                item['text'] = sentence
                item['subject'] = subject
                item['subj_type'] = subj_type
                item['object'] = ent.text
                item['obj_type'] = ent.type
                ners.append(item)
    return ners


def main():
    global zh_nlp
    subject = '伊丽莎白二世'
    x = run(hparams, subject=subject)
    f = open(hparams.char_dataset_path, 'w', encoding='utf-8')
    f.write(json.dumps(x, ensure_ascii=False, indent=4))
    f.close()


# if __name__ == '__main__':
#     main()

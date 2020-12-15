import os
import requests
import json
from pprint import pprint

from relation_extraction.predict import predict, predict_without_output
from relation_extraction.hparams import hparams

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
here = os.path.dirname(os.path.abspath(__file__))


def get_ner_res(text_list):
    """
    调用api，传入一个句子，获取NER_labels
    """
    all_need_predict = []
    for text in text_list:
        api_url = "http://10.60.1.100:10000/ner_str?text=" + text
        r = requests.post(api_url).json()
        if r['status'] == 'success':
            ner_labels = r['labels']
            ner_tokens = r['tokens']
            entitys = []
            rest = ''
            entity_name = ''
            for idx in range(len(ner_labels)):
                nl = ner_labels[idx]  # 标签
                if nl[0] == 'B':
                    rest = nl[2:]
                    if entity_name != '':
                        entitys.append(entity_name)
                    entity_name = ner_tokens[idx]
                elif nl[0] == 'I' and rest == nl[2:]:
                    entity_name += ner_tokens[idx]
                else:  # 遇到'o'
                    if entity_name != '':
                        entitys.append(entity_name)
                    entity_name = ''
                    rest = ''
            need_predict = []
            for subj_idx in range(len(entitys)):
                for obj_idx in range(subj_idx + 1, len(entitys)):
                    subj = entitys[subj_idx]
                    obj = entitys[obj_idx]
                    item = {}
                    item['text'] = text
                    item['subject'] = subj
                    item['object'] = obj
                    need_predict.append(item)
        all_need_predict.append(need_predict)
    return all_need_predict


def func(item):
    """
    用于对于list中的字典排序
    """
    return item['relation_prob']


def filter(cut_val=6.5, max_tri=3):
    """
    去掉重复的实体对；只保留概率大于cut_val的三元组；对于每个句子，最多预测出3个三元组
    """
    res = json.load(open('re_res.json', 'r+', encoding='utf-8'))
    prob = []
    all_res = []
    already = []
    for s in res:
        res_s = []
        for item in s:
            prob = item['relation_prob']
            h = item['h']['name']
            t = item['t']['name']
            if h + '=' + t in already:
                continue
            else:
                already.append(h + '=' + t)
                if prob >= cut_val:
                    res_s.append(item)
        all_res.append(res_s)

    all_r = []
    for s in all_res:
        s.sort(key=func, reverse=True)
        if len(s) > max_tri:
            need = s[0:max_tri]
            all_r.append(need)
        else:
            all_r.append(s)

    f = open('re_res_filtered.json', 'w', encoding='utf-8')
    f.write(json.dumps(all_r, ensure_ascii=False, indent=4))


def main():
    text_list = [
        '唐纳德·特朗普出生并成长于纽约州纽约市皇后区，为特朗普集团前任董事长兼总裁及特朗普娱乐公司的创办人。',
        '特朗普在福坦莫大学就读两年后，转至宾夕法尼亚大学沃顿商学院，在1968年取得经济金融的学士学位之后，特朗普进入父亲弗雷德·特朗普的房地产公司工作。',
        '2017年1月27日，特朗普签署《第13769号行政命令》，在该行政命令90天内禁止来自利比亚、伊朗、伊拉克、索马里、苏丹、叙利亚、也门等穆斯林世界7国的公民入境美国。',
        '2017年5月份突然毫无征兆地开除联邦调查局长詹姆斯·科米，是美国史上第二次FBI局长被开除。在纽约时报的一篇报道中显示，被特朗普免职的联邦调查局前局长詹姆斯·科米曾写过一份备忘录。'
    ]
    res = get_ner_res(text_list)
    f = open('ner_res.json', 'w', encoding='utf-8')
    f.write(json.dumps(res, ensure_ascii=False, indent=4))

    res = json.load(open('ner_res.json', 'r+', encoding='utf-8'))
    res_all = []
    for sent in res:
        r = predict_without_output(hparams, sent)
        res_all.append(r)
    f = open('re_res.json', 'w', encoding='utf-8')
    f.write(json.dumps(res_all, ensure_ascii=False, indent=4))
    filter(6.5)


if __name__ == '__main__':
    main()

# coding: utf-8
#
# Copyright 2020 Yequan Wang
# Author: Yequan Wang (tshwangyequan@gmail.com)
#
# pipeline file

import os
import sys
import requests
import json

sys.path.append(os.path.dirname(__file__))


def interface_model(input, url=None, proxies=None):
    """
    :param text: string or list, whose element is string
    :param proxies:
    :return: predict_text, follow the format of input text
    """
    if url is None:
        url = 'http://127.0.0.1:9803/recognize'
    headers = {'Content-Type': 'application/json'}
    params = input
    response = requests.post(url=url, headers=headers, data=json.dumps(params).encode('utf-8'), proxies=proxies)
    json_response = json.loads(response.text)
    return json_response


def get_docs(url, keyword, max_num_docs):
    input = {
        "index": "qbdata_global_news",
        "keyword": keyword,
        "source": ['content'],
        "fields": ["content"]
    }
    response_here = interface_model(input, url)
    if response_here['code'] == 20000:
        data_raw = response_here['data'][:max_num_docs]
        dict_avoid_repeat = dict()
        list_return = []
        for tmp in data_raw:
            if tmp['content'] not in dict_avoid_repeat:
                list_return.append(tmp)
                dict_avoid_repeat[tmp['content']] = None
        return list_return
    return []


def tackle_at_sentence_level(url, input, subject, do_ner=False):
    # data = ['不光要满足国内的需求，着急过圣诞的老外也给外贸企业加了不少订单，虽然人民币咔咔涨，但出口异常坚挺：光是11月的出口额就有1.8万亿元、增长14.9%，增速创下了今年的最高水平。\
    #             工厂一开工、电表就停不了，国内国外的需求赶在一起，用电量蹭蹭就上去了。\
    #             今年11月全发电量同比增长6.8%，但用电量同比增长9.4%，确实是很猛。\
    #             看总量缺口不大，那南方咋还缺电了呢?']
    if do_ner:
        tasks = ['_nlu_tokens', '_nlu_sentences', '_nlu_ner']
    else:
        tasks = ['_nlu_tokens', '_nlu_sentences']
    lang = 'zh'
    request_data = {
        'data': input,
        'lang': lang,
        'tasks': tasks
    }
    result = requests.post(url, json=request_data)
    result = result.content.decode()
    result = json.loads(result)
    assert result["status"] == 200
    list_paras = []
    for doc_res in result['result']['_nlu_sentences']:
        for sent_id, sent_words in doc_res.items():
            tmp_paras = ''.join(sent_words)
            if subject in tmp_paras:
                list_paras.append(tmp_paras)
            # print('{}: {}'.format(sent_id, ''.join(sent_words)))
    return list_paras
    # print('***************************** _nlu_ner *************************************')
    # pprint(result['result']['_nlu_ner'])


def get_relation(url, list_paras, subject):
    # print(list_paras)
    # list_paras = [
    #     "郭婞淳,中华台北女子举重队运动员,身高155厘米,1993年11月26日出生台湾省,2010年亚洲青少年举重锦标赛53公斤级亚军,2010年第一届青年奥运会亚军",
    #     "郭婞淳,中华台北女子举重队运动员,身高155厘米,1993年11月26日出生台湾省,2010年亚洲青少年举重锦标赛53公斤级亚军,2010年第一届青年奥运会亚军",
    #     "郭婞淳,中华台北女子举重队运动员,身高155厘米,1993年11月26日出生台湾省,2010年亚洲青少年举重锦标赛53公斤级亚军,2010年第一届青年奥运会亚军",
    # ]
    # subject = "郭婞淳"
    data = {'docs': list_paras, 'subject': subject}
    print(data)
    return interface_model(data, url)


if __name__ == "__main__":
    url_api_get_docs = "http://10.208.61.117:8868/goin_search/v1.0/document"
    url_api_sentence_level = 'http://10.61.2.148:5000/predict/_nlu_integrate'
    url_api_relation = 'http://10.208.61.123:20002/recognize'
    # url_api_relation = None
    subject = "特朗普"
    max_num_docs = 10
    data_docs = get_docs(url_api_get_docs, subject, max_num_docs=max_num_docs)
    list_docs = [tmp['content'] for tmp in data_docs]
    list_paras = tackle_at_sentence_level(url_api_sentence_level, list_docs, subject)
    print(subject, list_paras)
    print(get_relation(url_api_relation, list_paras, subject))

import os
import re
import torch
from pprint import pprint

from .data_utils import Mytokenizer, convert_pos_to_mask, get_idx2label
from .model import SentenceRE
from .data_utils import get_additional_tokens

here = os.path.dirname(os.path.abspath(__file__))


def do_predict(item, hparams, tokenizer, model, idx2label, isprint=False):
    device = hparams.device
    tokens, pos_e1, pos_e2 = tokenizer.tokenize(item)
    encoded = tokenizer.bert_tokenizer.batch_encode_plus(
        [(tokens, None)], return_tensors='pt')
    input_ids = encoded['input_ids'].to(device)
    token_type_ids = encoded['token_type_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)
    e1_mask = torch.tensor([convert_pos_to_mask(
        pos_e1, max_len=attention_mask.shape[1])]).to(device)
    e2_mask = torch.tensor([convert_pos_to_mask(
        pos_e2, max_len=attention_mask.shape[1])]).to(device)

    with torch.no_grad():
        logits = model(input_ids, token_type_ids,
                       attention_mask, e1_mask, e2_mask)[0]
        logits = logits.to(torch.device('cpu'))

        # print(logits)  # tensor([ 0.8768, 10.5473,  ...])
        # print(logits.argmax(0))  # tensor(1)  最大的那个位置
        # print(logits.argmax(0).item())  # 1

        if isprint:
            print('最大可能的关系是: {}'.format(idx2label[logits.argmax(0).item()]))

        top_ids = logits.argsort(0, descending=True).tolist()

        if isprint:
            for i, label_id in enumerate(top_ids, start=1):
                print('No.{}: 关系（{}）的可能性：{}'.format(
                    i, idx2label[label_id], logits[label_id]))

    return {'rel': idx2label[logits.argmax(0).item()], 'prob': logits[logits.argmax(0).item()].tolist()}


def predict(hparams):
    """
    前端展示，无限循环，用户输入句子，实体1，实体2；
    返回两个实体两个方向的各类别概率，并选择两个方向中概率最大的（如果是no relation 不返回结果）;
    如果一个方向是no_relation，一个方向不是，那么哪怕no_relation的概率较大，
    也返回那个不是no_relation的
    """
    device = hparams.device
    seed = hparams.seed
    torch.manual_seed(seed)

    pretrained_model_path = hparams.pretrained_model_path
    additional_file_path = hparams.additional_tokens_file
    label_set_file = hparams.label_set_file
    model_file = hparams.model_file
    is_add_entity_type = hparams.is_add_entity_type
    additional_tokens = get_additional_tokens(hparams.additional_tokens_file)
    special_type_file = hparams.special_type_file
    
    idx2label = get_idx2label(label_set_file)
    hparams.label_set_size = len(idx2label)
    model = SentenceRE(hparams).to(device)  # model 和所有的tensor都要放到GPU上！
    model.load_state_dict(torch.load(model_file, map_location=device))

    # 打印模型形状
    for name, parameters in model.named_parameters():
        print(name, ':', parameters.size())

    model.eval()

    tokenizer = Mytokenizer(pretrained_model_path=pretrained_model_path,
                            mask_entity=False,
                            is_add_entity_type=is_add_entity_type,
                            additional_file_path=additional_file_path, special_type_file=special_type_file)
    while True:
        entity_type = input('是否使用实体类型信息，y/n')
        if entity_type == 'y':
            is_add_entity_type = True
        else:
            is_add_entity_type = False

        text = input("输入中文句子：")

        entity1 = input("句子中的实体1：")
        entity1_type = ''
        if is_add_entity_type:
            entity1_type = input('输入实体1类型')
            if '<H-' + entity1_type + '>' not in additional_tokens:
                print('类型信息未预定义')

        entity2 = input("句子中的实体2：")
        entity2_type = ''

        if is_add_entity_type:
            entity2_type = input('输入实体2类型')
            if '<H-' + entity2_type + '>' not in additional_tokens:
                print('类型信息未预定义')

        try:
            match_obj1 = re.search(entity1, text)
            match_obj2 = re.search(entity2, text)
        except Exception as e:
            print(e)
            print('识别实体时发生错误！')
            continue

        if match_obj1 and match_obj2:
            e1_pos = match_obj1.span()  # 左开右闭的区间
            e2_pos = match_obj2.span()
            item_1 = {
                'h': {
                    'name': entity1,
                    'pos': e1_pos,
                    'type': entity1_type
                },
                't': {
                    'name': entity2,
                    'pos': e2_pos,
                    'type': entity2_type
                },
                'text': text
            }
            item_2 = {
                'h': {
                    'name': entity2,
                    'pos': e2_pos,
                    'type': entity2_type
                },
                't': {
                    'name': entity1,
                    'pos': e1_pos,
                    'type': entity1_type
                },
                'text': text
            }
            pre1 = do_predict(item_1, hparams, tokenizer,
                              model, idx2label, isprint=True)
            pre2 = do_predict(item_2, hparams, tokenizer,
                              model, idx2label, isprint=True)

            if pre1['rel'] == 'no_relation' and pre2['rel'] == 'no_relation':
                print('预测为无关系！')
                continue
            if pre1['rel'] != 'no_relation' and pre2['rel'] != 'no_relation':
                if pre1['prob'] > pre2['prob']:
                    flag = 1
                else:
                    flag = 2
            else:
                if pre1['rel'] != 'no_relation':
                    flag = 1
                else:
                    flag = 2

            if flag == 1:
                item_1['relation'] = pre1['rel']
                item_1['relation_prob'] = pre1['prob']
                print('头实体：', item_1['h']['name'], ', 尾实体：', item_1['t']['name'], ', 关系是：', item_1['relation'],
                      ', 可信概率为：', item_1['relation_prob'])
            elif flag == 2:
                item_2['relation'] = pre2['rel']
                item_2['relation_prob'] = pre2['prob']
                print('头实体：', item_2['h']['name'], ', 尾实体：', item_2['t']['name'], ', 关系是：', item_1['relation'],
                      ', 可信概率为：', item_2['relation_prob'])

        else:
            if match_obj1 is None:
                print('实体1不在句子中')
            if match_obj2 is None:
                print('实体2不在句子中')


def predict_without_output(hparams, items):
    """
    后端调用使用
    items: list, 其中的item包含：'text'、'subject'、'object'
    return: 返回list, 其中item样式如下
    {
        "h": {
            "name": "唐纳德·特朗普", 
            "pos": [0, 7]}, 
        "t": {
            "name": "纽约州", 
            "pos": [13, 16]}, 
        "text": "唐纳德·特朗普出生并成长于纽约州纽约市皇后区，为特朗普集团前任董事长兼总裁及特朗普娱乐公司的创办人。", 
        "relation": "出生地", 
        "relation_prob": 6.747187614440918}
    """
    device = hparams.device
    seed = hparams.seed
    torch.manual_seed(seed)

    pretrained_model_path = hparams.pretrained_model_path
    additional_file_path = hparams.additional_tokens_file
    label_set_file = hparams.label_set_file
    model_file = hparams.model_file
    is_add_entity_type = hparams.is_add_entity_type
    special_type_file = hparams.special_type_file

    idx2label = get_idx2label(label_set_file)
    hparams.label_set_size = len(idx2label)
    model = SentenceRE(hparams).to(device)  # model 和所有的tensor都要放到GPU上！
    model.load_state_dict(torch.load(model_file, map_location=device))
    model.eval()

    tokenizer = Mytokenizer(pretrained_model_path=pretrained_model_path,
                            mask_entity=False,
                            is_add_entity_type=is_add_entity_type,
                            additional_file_path=additional_file_path, special_type_file=special_type_file)
    res = []
    for item in items:
        entity1 = item['subject']
        entity1_type = item['subj_type']
        entity2 = item['object']
        entity2_type = item['obj_type']
        text = item['text']
        match_obj1 = re.search(entity1, text)
        match_obj2 = re.search(entity2, text)
        if match_obj1 and match_obj2:
            e1_pos = match_obj1.span()  # 左开右闭的区间
            e2_pos = match_obj2.span()
            item_1 = {
                'h': {
                    'name': entity1,
                    'pos': e1_pos,
                    'type': entity1_type
                },
                't': {
                    'name': entity2,
                    'pos': e2_pos,
                    'type': entity2_type
                },
                'text': text
            }
            item_2 = {
                'h': {
                    'name': entity2,
                    'pos': e2_pos,
                    'type': entity2_type
                },
                't': {
                    'name': entity1,
                    'pos': e1_pos,
                    'type': entity1_type
                },
                'text': text
            }
            pre1 = do_predict(item_1, hparams, tokenizer, model, idx2label)
            pre2 = do_predict(item_2, hparams, tokenizer, model, idx2label)
            if pre1['rel'] == 'no_relation' and pre2['rel'] == 'no_relation':
                continue
            if pre1['rel'] != 'no_relation' and pre2['rel'] != 'no_relation':
                if pre1['prob'] > pre2['prob']:
                    flag = 1
                else:
                    flag = 2
            else:
                if pre1['rel'] != 'no_relation':
                    flag = 1
                else:
                    flag = 2

            if flag == 1:
                item_1['relation'] = pre1['rel']
                item_1['relation_prob'] = pre1['prob']
                # pprint(item_1)
                if item_1['relation'] != 'no_relation':
                    res.append(item_1)
            elif flag == 2:
                item_2['relation'] = pre2['rel']
                item_2['relation_prob'] = pre2['prob']
                # pprint(item_2)
                if item_2['relation'] != 'no_relation':
                    res.append(item_2)

        else:
            if match_obj1 is None:
                print('实体1不在句子中')
            if match_obj2 is None:
                print('实体2不在句子中')
    return res


def predict_without_output_fixed_subj(hparams, items):
    """
        api调用使用，给定头实体，不需要双向预测
        items: list, 其中的item包含：'text'、'subject'、'object'
        return: 返回list, 其中item样式如下
        {
            "h": {
                "name": "唐纳德·特朗普",
                "pos": [0, 7]},
            "t": {
                "name": "纽约州",
                "pos": [13, 16]},
            "text": "唐纳德·特朗普出生并成长于纽约州纽约市皇后区，为特朗普集团前任董事长兼总裁及特朗普娱乐公司的创办人。",
            "relation": "出生地",
            "relation_prob": 6.747187614440918}
        """
    device = hparams.device
    seed = hparams.seed
    torch.manual_seed(seed)

    pretrained_model_path = hparams.pretrained_model_path
    additional_file_path = hparams.additional_tokens_file
    label_set_file = hparams.label_set_file
    model_file = hparams.model_file
    is_add_entity_type = hparams.is_add_entity_type
    special_type_file = hparams.special_type_file

    idx2label = get_idx2label(label_set_file)
    hparams.label_set_size = len(idx2label)
    model = SentenceRE(hparams).to(device)  # model 和所有的tensor都要放到GPU上！
    model.load_state_dict(torch.load(model_file, map_location=device))
    model.eval()

    tokenizer = Mytokenizer(pretrained_model_path=pretrained_model_path,
                            mask_entity=False,
                            is_add_entity_type=is_add_entity_type,
                            additional_file_path=additional_file_path, special_type_file=special_type_file)
    res = []
    for item in items:
        entity1 = item['subject']
        entity1_type = item['subj_type']
        entity2 = item['object']
        entity2_type = item['obj_type']
        text = item['text']
        match_obj1 = re.search(entity1, text)
        match_obj2 = re.search(entity2, text)
        if match_obj1 and match_obj2:
            e1_pos = match_obj1.span()  # 左开右闭的区间
            e2_pos = match_obj2.span()
            item_1 = {
                'h': {
                    'name': entity1,
                    'pos': e1_pos,
                    'type': entity1_type
                },
                't': {
                    'name': entity2,
                    'pos': e2_pos,
                    'type': entity2_type
                },
                'text': text
            }
            pre1 = do_predict(item_1, hparams, tokenizer, model, idx2label)

            if pre1['rel'] == 'no_relation':
                continue

            item_1['relation'] = pre1['rel']
            item_1['relation_prob'] = pre1['prob']

            # pprint(item_1)
            res.append(item_1)

        else:
            if match_obj1 is None:
                print('实体1不在句子中')
            if match_obj2 is None:
                print('实体2不在句子中')
    return res

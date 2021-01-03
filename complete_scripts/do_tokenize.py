import stanza
import json
from tqdm import tqdm
import re

from .hparams import hparams

zh_nlp = stanza.Pipeline(
    'zh', verbose=False, use_gpu=False)
tok_nlp = stanza.Pipeline(lang='zh', processors='tokenize',
                          tokenize_no_ssplit=True, verbose=False, use_gpu=False)


def handle_not_find_entity(tokens: list, e_name: str, text: str):
    entity_name = e_name.replace(' ', '')
    '''重组token，在其中找到实体，解决not find subj_end等问题'''
    res = {}
    entity_idx_start = text.replace(' ', '').index(
        entity_name.replace(' ', ''))
    entity_idx_end = entity_idx_start + len(entity_name)
    sum_length = 0  # 累加的长度
    is_find_start = False
    is_find_end = False

    i = 0
    for _ in tokens:
        if is_find_start and is_find_end:  # 都找到了
            break

        if sum_length == entity_idx_start and (is_find_start == False):
            # print(1)
            res['start'] = i
            is_find_start = True
        # 马上就走过开始位置了
        elif (sum_length + len(tokens[i].replace(' ', '')) > entity_idx_start) and (is_find_start == False):
            is_find_start = True
            # 下一个token属于这个实体名  '小习近''平啊'
            # '小伊利'、'二氏'中的'伊利' ;
            # '作低','频猎',  '人',
            # print(sum_length + len(tokens[i].replace(' ', '')), entity_idx_end)
            if (i + 1 != len(tokens)) and (sum_length + len(tokens[i].replace(' ', '')) < entity_idx_end):
                # print(2)
                if tokens[i + 1] in entity_name:  # 下一个token也属于这个实体  '小伊利'、'二氏'
                    idx_next = entity_name.index(
                        tokens[i + 1])  # 实体名中下一个token出现的开始位置
                    need_find_name = entity_name[0:idx_next]  # 需要的实体 - 伊利
                    assert need_find_name in tokens[i], str(tokens)
                    idx_pre = tokens[i].index(need_find_name)
                    split_token = tokens[i][0:idx_pre]  # 这是要被拆分出去的 - 小
                    tokens[i] = need_find_name
                    tokens.insert(i, split_token)
                    # 把刚刚插入的那个词，长度加上
                    sum_length += len(tokens[i].replace(' ', ''))
                    i += 1  # 多了一个了
                    res['start'] = i
                else:
                    merge = ''.join(tokens[i:i + 2])  # 合并两个
                    # :圣地亚','哥加利福尼亚' -> ":圣地亚哥加利福尼亚" 中的 1
                    assert entity_name in merge, str(tokens)
                    idx_pre = merge.index(entity_name)
                    split_token = tokens[i][0:idx_pre]  # 这是要被拆分出去的 - :
                    need_find_name = tokens[i][len(split_token):]  # 圣地亚
                    tokens[i] = need_find_name
                    tokens.insert(i, split_token)
                    # 把刚刚插入的那个词，长度加上
                    sum_length += len(tokens[i].replace(' ', ''))
                    i += 1  # 多了一个了
                    res['start'] = i

            else:  # 没有下一个token了，直接分割它 "习近平今年"、“小习近平啊”、‘小习近平’
                # print(3)
                # 不会出现‘习’‘近平啊’，因为我们是在找开始
                assert entity_name in tokens[i], str(tokens)
                now_start_idx = tokens[i].index(entity_name)
                now_end_idx = now_start_idx + len(entity_name)

                split_token_pre = ''
                split_token_end = ''

                if now_start_idx != 0:
                    split_token_pre = tokens[i][0:now_start_idx]  # 这是前面要被拆分出去的
                if now_end_idx < len(tokens[i].replace(' ', '')):
                    split_token_end = tokens[i][now_end_idx:]  # 这是后面要被拆分出去的

                tokens[i] = entity_name  # 把当前的改了

                if split_token_pre != '':
                    tokens.insert(i, split_token_pre)
                    sum_length += len(tokens[i].replace(' ', ''))
                    i += 1  # 多了一个了

                res['start'] = i
                res['end'] = i

                if split_token_end != '':
                    tokens.insert(i + 1, split_token_end)
                    sum_length += len(tokens[i].replace(' ', ''))
                    i += 1  # 多了一个了

                is_find_end = True

        if sum_length + len(tokens[i].replace(' ', '')) == entity_idx_end and (is_find_end == False):
            # print(4)
            res['end'] = i
            is_find_end = True
        # 马上就走过结尾的位置了'习''近' '平今年' 中的‘平’
        elif (sum_length + len(tokens[i].replace(' ', '')) > entity_idx_end) and (is_find_end == False):
            # "信作"
            if i - 1 >= 0 and (sum_length > entity_idx_start):  # 上一个token存在
                # print(5)
                assert tokens[i - 1] in entity_name, str(tokens)
                # 约翰·克里斯蒂安·巴赫 -> 有两个一样的·，多加一个字符识别吧
                idx = entity_name.index(
                    tokens[i - 1] + tokens[i][0]) + len(tokens[i - 1])
                need_find_name = entity_name[idx:]  # 我们需要找的"平"
                assert need_find_name in tokens[i], str(tokens)
                split_token_start = tokens[i].index(
                    need_find_name) + len(need_find_name)
                split_token = tokens[i][split_token_start:]  # 被分出去的部分
                tokens[i] = need_find_name  # 把当前的改了
                res['end'] = i
                if split_token != '':
                    tokens.insert(i + 1, split_token)  # 插到后面
                    sum_length += len(tokens[i].replace(' ', ''))
                    i += 1
                is_find_end = True
            else:
                # print(6)
                assert entity_name in tokens[i], str(tokens)
                split_start_idx = len(entity_name)
                split_token = tokens[i][split_start_idx:]
                tokens[i] = entity_name  # 把当前的改了
                res['end'] = i
                tokens.insert(i + 1, split_token)  # 插到后面
                sum_length += len(tokens[i].replace(' ', ''))
                i += 1
                is_find_end = True

        sum_length += len(tokens[i].replace(' ', ''))
        i += 1  # 等效于循环变量，但是它会随着我们往token中加入值而变化
    res['tokens'] = tokens
    assert ''.join(tokens[res['start']:res['end'] + 1]).replace(' ',
                                                                '') == entity_name.replace(' ', ''), str(
        res) + '实体名：' + str(entity_name)
    return res


def change_dataset_format(input_dataset):
    '''将原本的一个对象转换为baidu数据的格式
        {
        "text": "伊丽莎白二世生于伦敦，为约克公爵及公爵夫人（日后的乔治六世及伊丽莎白王后）长女，在家中接受私人教育",
        "subject": "伊丽莎白二世",
        "subj_type": "PERSON",
        "object": "伊丽莎白",
        "obj_type": "PERSON"
        }
         TO:
         {{
            "text": "...",
            "token": [...],
            "subj_start": 3,
            "subj_end": 4,
            "subj_type": "PERSON",
            "obj_start": 6,
            "obj_end": 6,
            "obj_type": "ORG"
        }
    '''
    global zh_nlp
    all_converted = []
    for data in input_dataset:
        subject_name = data['subject']
        object_name = data['object']
        sen = data['text']
        s = sen.replace('\n', '')  # 去除句子中的换行符
        term = {}
        term['text'] = s
        term['subject'] = subject_name
        term['subj_type'] = data['subj_type']
        term['object'] = object_name
        term['obj_type'] = data['obj_type']
        zh_doc = zh_nlp(s)
        tokens = []
        token_ids = []  # 索引位置
        for sent in zh_doc.sentences:
            for token in sent.tokens:
                tokens.append(token.text.replace(' ', ''))
                token_ids.append(token.id)
        term['token'] = tokens

        if subject_name.replace(' ', '') not in s.replace(' ', ''):
            print('找不到subject: ' + s.replace(' ', '') + str(subject_name))
            continue
        if object_name.replace(' ', '') not in s.replace(' ', ''):
            print('找不到object: ' + s.replace(' ', '') + str(object_name))
            continue

        sub_idx = s.replace(' ', '').index(
            subject_name.replace(' ', ''))
        obj_idx = s.replace(' ', '').index(
            object_name.replace(' ', ''))  # Zsuzsanna Zsohar 尾实体识别时也要去空格
        # stanza 在分词时把空格去掉了（Premio Príncipe de Asturias de las Letras）->"（"Premio","Príncipe","de","Asturias","de","las","Letras","）",

        is_find_subj_start = False
        is_find_obj_start = False

        # 先拆分, 为了好找type，把ners也跟着一起改，给分出来的统一命名为'O'
        try:
            res = handle_not_find_entity(
                term['token'], term['subject'], term['text'])
            term['subj_start'] = res['start']
            term['subj_end'] = res['end']

            res_2 = handle_not_find_entity(
                term['token'], term['object'], term['text'])
            term['obj_start'] = res_2['start']
            term['obj_end'] = res_2['end']
        except Exception as e:
            print(e)
            continue

        # 找到分词后的list中 subject和object 的起止范围
        sum_length = 0  # 累加的长度
        for i in range(len(tokens)):
            if sum_length == sub_idx and is_find_subj_start == False:
                term['subj_start'] = i
                is_find_subj_start = True

            if (is_find_subj_start == False) and (sum_length > sub_idx):
                is_find_subj_start = True

            if sum_length + len(tokens[i]) == sub_idx + len(term['subject'].replace(' ', '')):
                term['subj_end'] = i

            if sum_length == obj_idx and is_find_obj_start == False:
                is_find_obj_start = True
                term['obj_start'] = i

            if (sum_length > obj_idx) and (is_find_obj_start == False):  # 同理，给分词错误的实体也来一个NER
                is_find_obj_start = True

            if sum_length + len(tokens[i]) == obj_idx + len(term['object'].replace(' ', '')):
                term['obj_end'] = i

            sum_length += len(tokens[i].replace(' ', ''))  # '/ 胜'

        try:
            assert 'subj_start' in term, 'subj_start: ' + str(term)
            assert 'subj_type' in term, 'subj_type: ' + str(term)
            assert 'subj_end' in term, 'subj_end: ' + str(term)
            assert 'obj_start' in term, 'obj_start: ' + str(term)
            assert 'obj_type' in term, 'obj_type: ' + str(term)
            assert 'obj_end' in term, 'obj_end: ' + str(term)
            assert "".join(
                tokens[term['subj_start']:term['subj_end'] + 1]).replace(' ', '') == term['subject'].replace(' ',
                                                                                                             ''), '合并后不是subject' + str(
                term)
            assert "".join(
                tokens[term['obj_start']:term['obj_end'] + 1]).replace(' ', '') == term['object'].replace(' ',
                                                                                                          ''), '合并后不是object' + str(
                term)
            all_converted.append(term)
        except Exception as e:
            print(e)
        continue
    return all_converted


def return_tokens(sent):
    """
    对给定的句子分词
    """
    tokens = []
    doc_1 = tok_nlp(sent)
    for _, sentence in enumerate(doc_1.sentences):
        tokens.extend([token.text for token in sentence.tokens])
    return tokens


def re_tokenize(text, subj_name, obj_name):
    """
    将text进行分词，并且保证subj_name和obj_name是某几个token合并后的结果
    return: 返回tokens, [subj_start, subj_end], [obj_start, obj_end] 
    """
    if subj_name not in text or obj_name not in text:
        print('\nsubj_name or obj_name not in text')
        return None

    match_subj_list = [i.start() for i in re.finditer(subj_name, text)]
    match_obj_list = [i.start() for i in re.finditer(obj_name, text)]

    subj_i = 0
    obj_i = 0
    while True:
        flag = True
        if match_subj_list[subj_i] < match_obj_list[obj_i]:
            # 找第一个开始位置靠前的
            subj_start = match_subj_list[subj_i]
            subj_end = match_subj_list[subj_i] + len(subj_name) - 1
            # 靠后的obj 的开始位置位于subj的区间内
            if match_obj_list[obj_i] >= subj_start and match_obj_list[obj_i] <= subj_end:
                flag = False
                if obj_i != len(match_obj_list) - 1:  # 如果obj还有下一个，就去找下一个开始位置
                    obj_i += 1
                elif subj_i != len(match_subj_list) - 1:  # 如果subj 还有下一个，就subj换位置
                    subj_i += 1
                else:
                    print('\nsubj和obj的位置重合了！', text, subj_name, obj_name)
                    return None
        else:  # obj 比较靠前
            obj_start = match_obj_list[obj_i]
            obj_end = match_obj_list[obj_i] + len(obj_name) - 1
            # 靠后的subj 的开始位置位于obj的区间内
            if match_subj_list[subj_i] >= obj_start and match_subj_list[subj_i] <= obj_end:
                flag = False
                if subj_i != len(match_subj_list) - 1:  # 如果subj还有下一个，就去找下一个开始位置
                    subj_i += 1
                elif obj_i != len(match_obj_list) - 1:  # 如果obj 还有下一个，就obj换位置
                    obj_i += 1
                else:
                    print('\nsubj和obj的位置重合了！', text, subj_name, obj_name)
                    return None

        # 没有重合的情况发生
        if flag == True:
            break

    subject_index = [match_subj_list[subj_i],
                     match_subj_list[subj_i] + len(subj_name) - 1]
    object_index = [match_obj_list[obj_i],
                    match_obj_list[obj_i] + len(obj_name) - 1]

    if subject_index[0] < object_index[0]:
        front_index = subject_index
        back_index = object_index
        front = 'subj'
    else:
        front_index = object_index
        back_index = subject_index
        front = 'obj'

    tokens = []
    first_sent = text[0: front_index[0]]
    second_sent = text[front_index[0]: front_index[1] + 1]
    third_sent = text[front_index[1] + 1: back_index[0]]
    forth_sent = text[back_index[0]: back_index[1] + 1]
    fifth_sent = text[back_index[1] + 1:]
    assert first_sent + second_sent + third_sent + \
        forth_sent + fifth_sent == text, '\n5个句子拼凑失败：' + text[0:5]

    tokens.extend(return_tokens(first_sent))
    front_in_tokens = [len(tokens)]

    tokens.extend(return_tokens(second_sent))
    front_in_tokens.append(len(tokens) - 1)

    tokens.extend(return_tokens(third_sent))
    back_in_tokens = [len(tokens)]

    tokens.extend(return_tokens(forth_sent))
    back_in_tokens.append(len(tokens) - 1)

    tokens.extend(return_tokens(fifth_sent))

    if front == 'subj':  # 分词会把实体中的空格分掉，所以我们比较的时候也去掉空格
        assert ''.join(tokens[front_in_tokens[0]:front_in_tokens[1]+1]
                       ).replace(' ', '') == subj_name.replace(' ', ''), '\nsubj_name 拼凑失败: ' + subj_name.replace(' ', '')
        assert ''.join(tokens[back_in_tokens[0]:back_in_tokens[1]+1]
                       ).replace(' ', '') == obj_name.replace(' ', ''), '\nobj_name 拼凑失败: ' + obj_name.replace(' ', '')
        return tokens, front_in_tokens, back_in_tokens
    elif front == 'obj':
        assert ''.join(tokens[front_in_tokens[0]:front_in_tokens[1]+1]
                       ).replace(' ', '') == obj_name.replace(' ', ''), '\nobj_name 拼凑失败: ' + obj_name.replace(' ', '')
        assert ''.join(tokens[back_in_tokens[0]:back_in_tokens[1]+1]
                       ).replace(' ', '') == subj_name.replace(' ', ''), '\nsubj_name 拼凑失败: ' + subj_name.replace(' ', '')
        return tokens, back_in_tokens, front_in_tokens


def change_dataset_format2(file_in, file_out):
    '''将原本的一个对象转换为baidu数据的格式
        {
            "h":{
                "name":"叶莉",
                "pos":[
                        22,
                        24
                    ],
            "type":"PERSON"
        },
            "t":{
                    "name":"姚明",
                    "pos":[
                            9,
                            11
                        ],
            "type":"PERSON"
            },
            "relation":"配偶",
            "text":"也是在2004年，姚明找到了人生的另一半，与叶莉在雅典奥运会闭幕式上高调牵手"
            }
         TO:
         {{
            "text": "...",
            "token": [...],

            "subject": "..",
            "subj_start": 3,
            "subj_end": 4,
            "subj_type": "PERSON",

            "object": "..",
            "obj_start": 6,
            "obj_end": 6,
            "obj_type": "ORG",

            "relation":"配偶",
        }
    '''
    f = open(file_in, 'r+', encoding='utf-8')
    lines = f.readlines()
    print(file_in, '输出样本数', len(lines))
    f.close()
    all_converted = []
    for line_i in tqdm(range(len(lines))):
        line = lines[line_i]
        data = line.strip()
        data = json.loads(data)
        subject_name = data['h']['name']
        object_name = data['t']['name']
        sen = data['text']

        try:
            r = re_tokenize(sen, subject_name, object_name)
        except Exception as e:
            print(e)
            continue

        if r != None:
            term = {}
            tokens, subj_in_tokens, obj_in_tokens = r
            term['text'] = sen
            term['token'] = tokens
            term['subject'] = subject_name
            term['subj_start'] = subj_in_tokens[0]
            term['subj_end'] = subj_in_tokens[1]
            term['subj_type'] = data['h']['type']
            term['object'] = object_name
            term['obj_start'] = obj_in_tokens[0]
            term['obj_end'] = obj_in_tokens[1]
            term['obj_type'] = data['t']['type']
            term['relation'] = data['relation']
            all_converted.append(term)

    print(file_out, '输出样本数', len(all_converted))
    with open(file_out, 'w', encoding='utf-8') as f_out:
        for line in all_converted:
            json_str = json.dumps(line, ensure_ascii=False)
            f_out.write('{}\n'.format(json_str))
    return None


# change_dataset_format

# input_dataset = json.load(
#     open(hparams.char_dataset_path, 'r+', encoding='utf-8'))

# # print('输出样本数', len(input_dataset))

# converted_data = change_dataset_format(input_dataset)

# # print('输出样本数', len(converted_data))

# f3 = open(hparams.word_dataset_path, 'w', encoding='utf-8')
# f3.write(json.dumps(converted_data, ensure_ascii=False, indent=1))
# f3.close()

# change_dataset_format2
input_train_file = 'add_no_v3_ner_train_char.jsonl'
word_train_file = 'add_no_v3_ner_train_word.jsonl'
change_dataset_format2(input_train_file, word_train_file)

input_test_file = 'add_no_v3_ner_test_char.jsonl'
word_test_file = 'add_no_v3_ner_test_word.jsonl'
change_dataset_format2(input_test_file, word_test_file)

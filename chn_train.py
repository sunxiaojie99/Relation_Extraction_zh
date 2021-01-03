import os
import json

from relation_extraction.train import train, evaluate
from relation_extraction.hparams import hparams

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
here = os.path.dirname(os.path.abspath(__file__))


def check2(file_in):
    '''统计数据'''
    res_dict = {}
    with open(file_in, 'r+', encoding='utf-8') as f:
        lines = f.readlines()
        for item_in_i in range(len(lines)):
            item_in = lines[item_in_i].strip()
            item_in = json.loads(item_in)  # (将string转换为dict)
            cate = item_in['relation']
            if cate in res_dict:
                res_dict[cate] += 1
            else:
                res_dict[cate] = 1
    sorted(res_dict.items())
    sum_ = 0
    for k, v in res_dict.items():
        sum_ += v
        print(k, '\t', str(v))
    print('总计', sum_)
    return res_dict


def main():
    train(hparams)
    # check2(hparams.validation_file)
    # evaluate(hparams)


if __name__ == '__main__':
    main()

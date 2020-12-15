# Relation_Extraction_zh
Traditional Bert model was used to extract Chinese relationship

## Requirements
This repo was tested on Python 3.6 and Pytorch. The main requirements are:

torch
transformers
tqdm
sklearn
tensorboard
jsonlines

## The main data source(add some data from wikipedia)
https://aistudio.baidu.com/aistudio/loginmid?redirectUri=http%3A%2F%2Faistudio.baidu.com%2Faistudio%2Fdatasetdetail%2F27955%3F_%3D1606105736684

## bert_model

Download chinese-bert-wwm-ext model from https://huggingface.co/models  and unzip it under pretrained_models\

The directory structure
```
├─pretrained_models
│  └─chinese-bert-wwm
│          added_tokens.json
│          config.json
│          pytorch_model.bin
│          special_tokens_map.json
│          tokenizer_config.json
│          vocab.txt
```

## Usage

1. use gpu to train (not add entity tyoe)
```
CUDA_VISIBLE_DEVICES=1 nohup python -u chn_train.py > process_train.log 2>&1 &
```

2. use cpu to train (add entity tyoe)
```
nohup python -u chn_train.py --device cpu --is_add_entity_type > process_train.log 2>&1 &
```

## Evaluation results of 20 epochs of model training:
```
              precision    recall  f1-score   support

 no_relation       0.99      0.99      0.99     14284
         出生地       0.93      0.93      0.93      3638
        出生日期       1.00      1.00      1.00      4048
          职业       0.99      1.00      0.99      1083
          国籍       0.90      0.94      0.92      2504
          母亲       0.97      0.96      0.96       643
          配偶       0.97      1.00      0.98      1839
         居住地       0.73      0.31      0.44        70
          父亲       0.94      0.97      0.95       901
          祖籍       0.87      0.86      0.86       267
        死亡日期       1.00      0.94      0.97       120
         死亡地       0.71      0.47      0.57       159
        所属派别       0.92      1.00      0.96        11
        信仰宗教       0.99      1.00      0.99        70
          性别       1.00      1.00      1.00        25
          子女       0.84      0.64      0.73        25
          学历       1.00      0.90      0.95        20
          别名       0.96      0.94      0.95        79
          专业       1.00      0.17      0.29         6
    就职部门(公司)       0.63      0.84      0.72        31
         外文名       0.00      0.00      0.00         0
        教育背景       0.69      0.64      0.67        14
          母校       0.84      0.76      0.80        70

   micro avg       0.97      0.97      0.97     29907
   macro avg       0.86      0.79      0.81     29907
weighted avg       0.97      0.97      0.97     29907

Precision (micro): 97.044%
   Recall (micro): 97.044%
       F1 (micro): 97.044%
```

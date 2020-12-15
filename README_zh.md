# Relation_rxtraction_zh

## 介绍
based on Bert

## bert_model

从 https://huggingface.co/models 下载chinese-bert-wwm模型，解压在pretrained_models下

chinese-bert-wwm目录结构如下：

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

## 文件结构

```
│  chn_test.py  # 预测
│  chn_train.py  # 训练
│  ner_test.py  # NER测试脚本，可忽略
│  process_train.log  # 训练过程
│  README.en.md
│  README.md
│  requirements.txt
│          
├─complete_scripts  # 结合stanza工具的NER和POS模块，进行完整的实体关系识别
│  │  do_tokenize.py
│  │  hparams.py
│  │  README.md
│  │  run.py
│  │  
│  ├─data
│  │      char_dataset.json
│  │      document.json
│  │      word_dataset.json
│          
├─datasets
│  │  additional_special_tokens.txt  # 额外加入词表的类型信息
│  │  change_format.py
│  │  not_need_label.txt  # 在relation.txt，但我们不需要的label
│  │  relation.txt
│  │  wi_test_char.jsonl
│  │  wi_train_char.jsonl  # 基于字的训练数据集
│  │  
│  ├─demo  # 跑的一个demo，可忽略
│  │      additional_special_tokens.txt
│  │      checkpoint.json
│  │      model.bin
│  │      print.log
│  │      relation.txt
│  │      train_small.jsonl
│  │      val_small.jsonl
│  │      
│  └─word_based  # 基于词的数据集
│          fn_test_word.jsonl
│          fn_train_word.jsonl
│          
├─pretrained_models
│  └─chinese-bert-wwm
│          added_tokens.json
│          config.json
│          pytorch_model.bin
│          special_tokens_map.json
│          tokenizer_config.json
│          vocab.txt
│          
├─relation_extraction
│  │  data_utils.py  # 数据处理，构建dataset
│  │  hparams.py  # 文件位置、模型相关参数
│  │  model.py  # 模型构建
│  │  my_metrics.py  # 自定义度量函数
│  │  predict.py  # 预测
│  │  train.py  # 训练+测试
│  
│  
│          
├─saved_models  # 带模型类型信息训练好的模型
│  │  checkpoint.json
│  │  checkpoint_macro_file.json
│  │  checkpoint_micro_file.json
│  │  model.bin
│  │  
│  ├─char_no_type  # 不带类型信息训练好的模型
│  
│
│              
└─statistics
        handle_data.log
        process_train.log
        stat_cate.log
```


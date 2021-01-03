# 完整调用

## 流程

1. **输入**：和指定subject相关的文档集。位置`data/document.json`，通过`complete_scripts\hparams.py`中参数指定。
2. **提取（sent, subj, obj）三元组** `complete_scripts\run.py`：
	- 遍历文档中所有的句子，把包含subject 的句子抽出。函数 `run(hparams, subject=subject)`
	- 针对各个句子使用NER、POS工具进行处理，以句子为单位，提取出各个句子中包含的entity和NP(NOUN、PROPN)。函数 `deal_one_sent(sentence=sentence, subject=subject)`
	- 【特殊注意】：① 过滤了和subject同名的object名，对于同名候选object优先选取位置靠前的 ② 对于PROPN，进行连续的token合并操作，因为连续出现的往往是一个专有名词 ③ 对于NOUN，将每个出现的NOUN token 独立加入，因为"1936年伯父爱德华"的例子中，"年"和"伯父"都是NOUN，如果合并会造成不当，但在"为约克公爵及公爵夫人" 中，"公爵"和"夫人"也都是NOUN，看上去应该合并，总的来说，合并的话弊大于利，因此选择不合并。
3. **传入关系分类模型进行分类，获取（subj, relation, object）三元组**



## 函数介绍

1. `run(hparams, subject=subject)`：

- 输入：subject名，以及通过`hparams.documents_path` 指定的和给出subject相关的文档集合。

- 输出：`deal_one_sent(sentence=sentence, subject=subject)`返回结果的合并，文件位置`hparams.char_dataset_path` 

  ```json
  [
      {
          "text": "伊丽莎白二世生于伦敦，为约克公爵及公爵夫人（日后的乔治六世及伊丽莎白王后）长女，在家中接受私人教育",
          "subjet": "伊丽莎白二世",
          "object": "伊丽莎白",
          "obj_type": "PERSON"
      },
      {
          "text": "伊丽莎白二世生于伦敦，为约克公爵及公爵夫人（日后的乔治六世及伊丽莎白王后）长女，在家中接受私人教育",
          "subjet": "伊丽莎白二世",
          "object": "伦敦",
          "obj_type": "GPE"
      },
  ]
  ```

2. `deal_one_sent(sentence=sentence, subject=subject)`

- 输入：包含subject的单个句子，subject 名

- 输出：

  ```json
  [
      {
          "text": "伊丽莎白二世生于伦敦，为约克公爵及公爵夫人（日后的乔治六世及伊丽莎白王后）长女，在家中接受私人教育",
          "subjet": "伊丽莎白二世",
          "object": "伊丽莎白",
          "obj_type": "PERSON"
      },
      {
          "text": "伊丽莎白二世生于伦敦，为约克公爵及公爵夫人（日后的乔治六世及伊丽莎白王后）长女，在家中接受私人教育",
          "subjet": "伊丽莎白二世",
          "object": "伦敦",
          "obj_type": "GPE"
      },
  ]
  ```

  

## 传入基于词的RE模型，数据格式转换

将`hparams.char_dataset_path` 输入到`complete_scripts/do_tokenize.py` 中，转化为基于词输入的数据格式：`hparams.word_dataset_path`
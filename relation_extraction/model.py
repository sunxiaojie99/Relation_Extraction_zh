import os
import logging
import torch
from transformers import BertTokenizer, BertModel, BertConfig
import torch.nn as nn
from .data_utils import get_additional_tokens

here = os.path.dirname(os.path.abspath(__file__))


class FCLayer(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=0.0, use_activation=True):
        super(FCLayer, self).__init__()
        self.use_activation = use_activation
        self.drop = nn.Dropout(p=dropout_rate)
        self.dense = nn.Linear(input_dim, output_dim)
        self.activate = nn.Tanh()

    def forward(self, x):
        x = self.drop(x)
        if self.use_activation:
            x = self.activate(x)
        x = self.dense(x)
        return x


class SentenceRE(nn.Module):

    def __init__(self, hparams):
        super(SentenceRE, self).__init__()  # 对继承自父类的属性进行初始化
        self.pretrained_model_path = hparams.pretrained_model_path or 'hfl/chinese-bert-wwm'
        self.embedding_dim = hparams.embedding_dim
        self.dropout = hparams.dropout
        self.label_set_size = hparams.label_set_size  # 在train.py 中设置
        self.model_config = BertConfig.from_pretrained(self.pretrained_model_path)
        self.model_config.output_hidden_states = False
        self.model_config.output_attentions = False

        # 添加特殊标记
        self.additional_tokens_file = hparams.additional_tokens_file
        self.additional_tokens = get_additional_tokens(self.additional_tokens_file)
        self.bert_tokenizer = BertTokenizer.from_pretrained(self.pretrained_model_path)
        self.bert_tokenizer.add_special_tokens({'additional_special_tokens': self.additional_tokens})

        # 加载bert模型参数，如果想冻结可以参考注释
        self.bert_model = BertModel.from_pretrained(self.pretrained_model_path, config=self.model_config)
        # # Notice: resize_token_embeddings expect to receive the full size of the new vocabulary
        self.bert_model.resize_token_embeddings(len(self.bert_tokenizer))
        # for param in self.bert_model.parameters():
        #     param.requires_grad = False

        # 如果用一个，可以达到两个实体共享同一个FC层
        self.entity1_fc_layer = FCLayer(self.embedding_dim, self.embedding_dim, self.dropout)
        self.entity2_fc_layer = FCLayer(self.embedding_dim, self.embedding_dim, self.dropout)
        self.cls_fc_layer = FCLayer(self.embedding_dim, self.embedding_dim, self.dropout)
        self.label_classifier = FCLayer(self.embedding_dim * 3, self.label_set_size, self.dropout, use_activation=False)

        self.dense = nn.Linear(self.embedding_dim, self.embedding_dim)  # 全连接层
        self.drop = nn.Dropout(p=self.dropout)  # 防止或减轻过拟合
        self.activate = nn.Tanh()
        self.norm = nn.LayerNorm(self.embedding_dim * 3)  # 我们输出是 [cls] + 实体1 + 实体2 的embedding

    def forward(self, token_ids, token_type_ids, attention_mask, e1_mask, e2_mask):
        outputs = self.bert_model(token_ids, token_type_ids,
                                  attention_mask)  # # sequence_output, pooled_output, (hidden_states), (attentions)
        sequence_ouput = outputs[0]
        pooled_output = outputs[1]
        # sequence_ouptut 每个token的output, [batch_size, seq_length, embedding_size]
        # pooled_output 句子的output, [batch_size, embedding_size]

        # 每个实体对应范围向量的平均值
        e1_h = self.entity_average(sequence_ouput, e1_mask)
        e2_h = self.entity_average(sequence_ouput, e2_mask)

        # Dropout -> tanh -> fc_layer (Share FC layer for e1 and e2)
        e1_h = self.entity1_fc_layer(e1_h)
        e2_h = self.entity2_fc_layer(e2_h)
        pooled_output = self.cls_fc_layer(pooled_output)

        concat_h = torch.cat([pooled_output, e1_h, e2_h], dim=-1)  # [batch_size, embedding_size * 3]
        concat_h = self.norm(concat_h)
        logits = self.label_classifier(concat_h)  # [batch_size, label_set_size] 最后一层不加activate了

        return logits

    @staticmethod
    def entity_average(sequence_ouput, e_mask):
        """
        Average the entity hidden state vectors (H_i ~ H_j)
        :param sequence_ouptut: [batch_size, max_seq_length, embedding_size]
        :param e_mask: [batch_size, max_seq_length]
        :return: [batch_size, embedding_size]
        """
        # print(sequence_ouput)
        # sequence_ouptut = torch.tensor(sequence_ouput)

        e_mask_unsqueeze = e_mask.unsqueeze(1)  # [batch_size, 1, max_seq_length]
        # torch.bmm：两个tensor的矩阵乘法，两个tensor的维度必须为3
        sum_vector = torch.bmm(e_mask_unsqueeze.float(),
                               sequence_ouput)  # [batch_size, 1, max_seq_length] * [batch_size, max_seq_length, embedding_size] = [batch_size, 1, embedding_size]
        sum_vector = sum_vector.squeeze(1)  # [batch_size, 1, embedding_size] -> [batch_size, embedding_size]

        len_tensor = (e_mask != 0).sum(dim=1).unsqueeze(1)  # [batch_size, ] -> [batch_size, 1] 计算每个实体包含了多少个token
        avg_vector = sum_vector.float() / len_tensor.float()  # [batch_size, embedding_size] / [batch_size, 1] ：broadcasting
        return avg_vector

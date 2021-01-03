import os
import argparse
import torch


here = os.path.dirname(os.path.abspath(__file__))

default_pretrained_model_path = os.path.join(
    here, '../pretrained_models/chinese-bert-wwm')

# default_train_file = os.path.join(here, '../datasets/wi_train_char.jsonl')
# default_validation_file = os.path.join(here, '../datasets/wi_test_char.jsonl')

# default_train_file = os.path.join(
#     here, '../datasets/add_no_train_char_v2_fn.jsonl')
# default_validation_file = os.path.join(
#     here, '../datasets/add_no_test_char_v2_fn.jsonl')

default_train_file = os.path.join(
    here, '../datasets/add_no_train_char_v3_ner_fn.jsonl')
default_validation_file = os.path.join(
    here, '../datasets/add_no_test_char_v3_ner_fn.jsonl')

default_output_dir = os.path.join(here, '../saved_models')
default_log_dir = os.path.join(default_output_dir, 'runs')
default_label_set_file = os.path.join(here, '../datasets/relation.txt')
default_additional_tokens_file = os.path.join(
    here, '../datasets/additional_special_tokens.txt')
default_model_file = os.path.join(default_output_dir, 'params_model.bin')
default_complete_model_file = os.path.join(
    default_output_dir, 'complete_model.pth')
default_checkpoint_file = os.path.join(default_output_dir, 'checkpoint.json')
default_device = "cuda" if torch.cuda.is_available() else "cpu"
default_not_need_label_file = os.path.join(
    here, '../datasets/not_need_label.txt')
default_checkpoint_macro_file = os.path.join(
    default_output_dir, 'checkpoint_macro_file.json')
default_checkpoint_micro_file = os.path.join(
    default_output_dir, 'checkpoint_micro_file.json')
default_special_type_file = os.path.join(here, '../datasets/special_type.txt')


parser = argparse.ArgumentParser()

parser.add_argument("--pretrained_model_path", type=str,
                    default=default_pretrained_model_path)
parser.add_argument("--train_file", type=str, default=default_train_file)
parser.add_argument("--validation_file", type=str,
                    default=default_validation_file)
parser.add_argument("--output_dir", type=str, default=default_output_dir)
parser.add_argument("--log_dir", type=str, default=default_log_dir)
parser.add_argument("--label_set_file", type=str,
                    default=default_label_set_file)
parser.add_argument("--additional_tokens_file", type=str,
                    default=default_additional_tokens_file)
parser.add_argument("--model_file", type=str, default=default_model_file)
parser.add_argument("--complete_model_file", type=str,
                    default=default_complete_model_file)
parser.add_argument("--checkpoint_file", type=str,
                    default=default_checkpoint_file)
parser.add_argument("--checkpoint_macro_file", type=str,
                    default=default_checkpoint_macro_file)
parser.add_argument("--checkpoint_micro_file", type=str,
                    default=default_checkpoint_micro_file)
parser.add_argument("--not_need_label_file", type=str,
                    default=default_not_need_label_file)
parser.add_argument("--special_type_file", type=str,
                    default=default_special_type_file)

# model
parser.add_argument('--embedding_dim', type=int, default=768,
                    required=False, help='embedding_dim')
parser.add_argument('--dropout', type=float, default=0.1,
                    required=False, help='dropout')

parser.add_argument('--device', type=str, default=default_device)
parser.add_argument("--seed", type=int, default=12345)
parser.add_argument("--max_len", type=int, default=128)
parser.add_argument("--train_batch_size", type=int, default=8)
parser.add_argument("--validation_batch_size", type=int, default=8)
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--learning_rate", type=float, default=1e-5)
parser.add_argument("--weight_decay", type=float,
                    default=0)  # 正则化项前的系数，默认不进行正则化
parser.add_argument("--is_add_entity_type",
                    action='store_true', help='default false')  # 默认为false

hparams = parser.parse_args()

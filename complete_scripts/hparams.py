import os
import argparse


here = os.path.dirname(os.path.abspath(__file__))

default_documents_path = os.path.join(here, 'data/document.json')
default_char_dataset_path = os.path.join(here, 'data/char_dataset.json')
default_word_dataset_path = os.path.join(here, 'data/word_dataset.json')

parser = argparse.ArgumentParser()

parser.add_argument("--documents_path", type=str,
                    default=default_documents_path)
parser.add_argument("--char_dataset_path", type=str,
                    default=default_char_dataset_path)
parser.add_argument("--word_dataset_path", type=str,
                    default=default_word_dataset_path)

hparams = parser.parse_args()

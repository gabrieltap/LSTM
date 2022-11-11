from io import open
from typing import List
from src.datasets.utils import parse_data
from src.datasets.utils import create_input_tensor
from src.datasets.utils import create_target_tensor
from src.datasets.utils import add_words_to_dict

import random


class Dataset:
    def __init__(self, filename: str, language: str = 'english'):
        self.language_list = None
        self.language_dictionary = None
        self.n_unique_words = None
        self.test_tensors = None
        self.val_tensors = None
        self.train_tensors = None
        self.dataset_sizes = None
        self.dataloaders = None
        self.language = language
        lines = open(filename, encoding='utf-8').read().strip().split('\n')

        sentences_complete = parse_data(lines)
        self.language_sentences = self._get_sentences_by_language(sentences_complete)
        random.shuffle(self.language_sentences)

    def _get_sentences_by_language(self, sentences_complete) -> List[str]:
        if self.language.lower() == 'english':
            language_code = 0
        elif self.language.lower() == 'spanish':
            language_code = 1
        else:
            print(f'{self.language} is not defined, using english instead')
            language_code = 0

        return [pair[language_code] for pair in sentences_complete]

    def split_dataset(self, train: int, val: int, test: int):
        train_sentences = self.language_sentences[:train]
        val_sentences = self.language_sentences[train:train + val]
        test_sentences = self.language_sentences[train + val:train + val + test]
        
        lst = []
        for sentence in train_sentences+val_sentences+test_sentences:
            lst += sentence.split()
        self.n_unique_words = len(set(lst))

        language_dictionary = {}
        language_list = []
        add_words_to_dict(language_dictionary, language_list, train_sentences)
        add_words_to_dict(language_dictionary, language_list, val_sentences)
        add_words_to_dict(language_dictionary, language_list, test_sentences)
        
        self.train_tensors = [
            (create_input_tensor(sentence, language_dictionary),
             create_target_tensor(sentence, language_dictionary)) for
            sentence in train_sentences]
        self.val_tensors = [
            (create_input_tensor(sentence, language_dictionary),
             create_target_tensor(sentence, language_dictionary)) for
            sentence in val_sentences]
        self.test_tensors = [
            (create_input_tensor(sentence, language_dictionary),
             create_target_tensor(sentence, language_dictionary)) for
            sentence in test_sentences]
        
        self.language_dictionary = language_dictionary
        self.language_list = language_list

    def define_dataloaders_and_sizes(self):
        self.dataloaders = {'train': self.train_tensors,
                            'val': self.val_tensors,
                            'test': self.test_tensors}

        self.dataset_sizes = {'train': len(self.train_tensors),
                              'val': len(self.val_tensors),
                              'test': len(self.test_tensors)}

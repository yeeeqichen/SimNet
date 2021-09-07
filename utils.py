import random
from pytorch_transformers import BertTokenizer
from os import path, mkdir
import numpy
from tqdm import tqdm
import re
import torch

MAX_LENGTH = 88


PAD, CLS, SEP = '[PAD]', '[CLS]', '[SEP]'  # padding符号, bert中综合信息符号


def _pad(_token_id, _pad_size, sep_pos=None):
    _seq_len = len(_token_id)
    if len(_token_id) < _pad_size:
        mask = [1] * len(_token_id) + [0] * (_pad_size - len(_token_id))
        _token_id += ([0] * (_pad_size - len(_token_id)))
    else:
        mask = [1] * _pad_size
        _token_id = _token_id[:_pad_size]
        _seq_len = _pad_size
    seg_id = None
    if sep_pos is not None:
        seg_id = [0] * (sep_pos + 1) + [1] * (_pad_size - sep_pos - 1)
    return _token_id, mask, _seq_len, seg_id


def build_dataset_2(train_path, dev_path, test_path, tokenizer, padding_size):
    def load_dataset(path, pad_size=120):
        contents = []
        with open(path, 'r', encoding='UTF-8') as f:
            for line in tqdm(f):
                sent1, sent2, label = line.strip('\n').split('\t')
                tokens_1 = tokenizer.tokenize(sent1)
                tokens_2 = tokenizer.tokenize(sent2)
                tokens = [CLS] + tokens_1 + [SEP] + tokens_2
                sep_pos = len(tokens_1) + 1
                token_ids = tokenizer.convert_tokens_to_ids(tokens)
                token_ids, mask, seq_len, seg_id = _pad(token_ids, pad_size, sep_pos)
                contents.append((token_ids, int(label), seq_len, mask, seg_id))
        return contents
    train = load_dataset(train_path, padding_size)
    dev = load_dataset(dev_path, padding_size)
    test = load_dataset(test_path, padding_size)
    return train, dev, test


def build_dataset(train_path, dev_path, test_path, tokenizer, padding_size):

    def load_dataset(path, pad_size=32):
        contents = []
        with open(path, 'r', encoding='UTF-8') as f:
            for line in tqdm(f):
                sent1, sent2, label = line.strip('\n').split('\t')
                token_1 = tokenizer.tokenize(sent1)
                token_1 = [CLS] + token_1
                token_ids_1 = tokenizer.convert_tokens_to_ids(token_1)
                token_ids_1, mask_1, seq_len_1, _ = _pad(token_ids_1, pad_size)

                token_2 = tokenizer.tokenize(sent2)
                token_2 = [CLS] + token_2
                token_ids_2 = tokenizer.convert_tokens_to_ids(token_2)
                token_ids_2, mask_2, seq_len_2, _ = _pad(token_ids_2, pad_size)

                contents.append((token_ids_1, token_ids_2, int(label), seq_len_1, seq_len_2, mask_1, mask_2))
        return contents
    train = load_dataset(train_path, padding_size)
    dev = load_dataset(dev_path, padding_size)
    test = load_dataset(test_path, padding_size)
    return train, dev, test


class DatasetIterator(object):
    def __init__(self, batches, batch_size, device, method=0):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.residue = False  # 记录batch数量是否为整数
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device
        self._to_tensor = self._to_tensor1 if method == 0 else self._to_tensor2

    def _to_tensor1(self, datas):
        x_1 = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        x_2 = torch.LongTensor([_[1] for _ in datas]).to(self.device)
        y = torch.LongTensor([_[2] for _ in datas]).to(self.device)

        # pad前的长度(超过pad_size的设为pad_size)
        seq_len_1 = torch.LongTensor([_[3] for _ in datas]).to(self.device)
        seq_len_2 = torch.LongTensor([_[4] for _ in datas]).to(self.device)
        mask_1 = torch.LongTensor([_[5] for _ in datas]).to(self.device)
        mask_2 = torch.LongTensor([_[6] for _ in datas]).to(self.device)
        return (x_1, x_2, seq_len_1, seq_len_2, mask_1, mask_2), y

    def _to_tensor2(self, datas):
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)

        # pad前的长度(超过pad_size的设为pad_size)
        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        mask = torch.LongTensor([_[3] for _ in datas]).to(self.device)
        seg_id = torch.LongTensor([_[4] for _ in datas]).to(self.device)
        return (x, seq_len, mask, seg_id), y

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


# def split_dataset():
#     """
#     split the data into three parts: train, valid and test
#     size: train - 300000, valid - 30000, test - 37373
#     :return: train, valid, test
#     """
#     with open('data/train_snli.txt') as f:
#         data = []
#         cnt = 0
#         for line in f:
#             cnt += 1
#             sent1, sent2, label = line.strip('\n').split('\t')
#             data.append([sent1, sent2, label])
#         print('total sentence pairs: {}'.format(cnt))
#         random.shuffle(data)
#         train_data = data[:300000]
#         valid_data = data[300000:337373]
#         test_data = data[337373:]
#         print('train size: {}, valid size: {}, test size: {}'.format(len(train_data), len(valid_data), len(test_data)))
#         write_file('data/train.txt', train_data)
#         write_file('data/valid.txt', valid_data)
#         write_file('data/test.txt', test_data)


if __name__ == '__main__':
    # tokenizer = BertTokenizer.from_pretrained('D:/PythonProjects/语言模型/bert-base-uncased')
    # data_set, _, _ = build_dataset('data/test.txt', 'data/test.txt', 'data/test.txt',  tokenizer=tokenizer, padding_size=32)
    # iter = DatasetIterator(data_set, batch_size=32, device='cuda:0')
    # for batch in iter:
    #     print(batch)
    pass


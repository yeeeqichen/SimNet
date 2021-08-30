import random
from pytorch_transformers import BertTokenizer
from os import path, mkdir
import numpy
import tqdm

MAX_LENGTH = 88


def read_file(file_name, bert_path):
    cache_file = 'data/cache/' + file_name[5:].replace('.txt', '.npy')
    if path.exists(cache_file):
        print('loading cached data from {}'.format(cache_file))
        data = numpy.load(cache_file, allow_pickle=True)
        print(data.shape)
        return data
    tokenizer = BertTokenizer.from_pretrained(bert_path)
    data = []
    with open(file_name) as f:
        for line in tqdm.tqdm(f):
            sent1, sent2, label = line.strip('\n').split('\t')
            tokens1 = ['[CLS]'] + tokenizer.tokenize(sent1) + ['[SEP]']
            ids1 = tokenizer.convert_tokens_to_ids(tokens1)
            tokens2 = ['[CLS]'] + tokenizer.tokenize(sent2) + ['[SEP]']
            ids2 = tokenizer.convert_tokens_to_ids(tokens2)
            mask1 = [1] * len(ids1)
            mask2 = [1] * len(ids2)
            mask1 += [0] * (MAX_LENGTH - len(ids1))
            ids1 += [0] * (MAX_LENGTH - len(ids1))
            mask2 += [0] * (MAX_LENGTH - len(ids2))
            ids2 += [0] * (MAX_LENGTH - len(ids2))
            data.append([numpy.array([ids1, mask1], dtype=object),
                         numpy.array([ids2, mask2], dtype=object),
                         numpy.array(int(label), dtype=object)]
                        )
    data = numpy.array(data, dtype=object)
    if not path.exists('data/cache'):
        mkdir('data/cache')
    numpy.save(cache_file, data)
    print(data.shape)
    return data


def write_file(file_name, triplets):
    with open(file_name, 'w') as file:
        for triple in triplets:
            file.write('\t'.join(triple) + '\n')


def eval_dataset():
    tokenizer = BertTokenizer.from_pretrained('/Users/yeeeqichen/Desktop/Python Projects/BERT_预训练模型/bert_base_uncased')
    data = read_file('data/train_snli.txt')
    print('total sentence pairs: {}'.format(len(data)))
    max_length = -1
    for triple in data:
        tokens1 = tokenizer.tokenize(triple[0])
        tokens2 = tokenizer.tokenize(triple[1])
        max_length = max(max_length, len(tokens1), len(tokens2))
    print(max_length)


def split_dataset():
    """
    split the data into three parts: train, valid and test
    size: train - 300000, valid - 30000, test - 37373
    :return: train, valid, test
    """
    with open('data/train_snli.txt') as f:
        data = []
        cnt = 0
        for line in f:
            cnt += 1
            sent1, sent2, label = line.strip('\n').split('\t')
            data.append([sent1, sent2, label])
        print('total sentence pairs: {}'.format(cnt))
        random.shuffle(data)
        train_data = data[:300000]
        valid_data = data[300000:337373]
        test_data = data[337373:]
        print('train size: {}, valid size: {}, test size: {}'.format(len(train_data), len(valid_data), len(test_data)))
        write_file('data/train.txt', train_data)
        write_file('data/valid.txt', valid_data)
        write_file('data/test.txt', test_data)


def test():
    # split_dataset()
    # eval_dataset()
    read_file('data/test.txt', bert_path='')
    pass


if __name__ == '__main__':
    test()

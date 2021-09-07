from pytorch_transformers import BertTokenizer
from tqdm import tqdm
import torch
from torch.utils.data import Dataset

PAD, CLS, SEP = '[PAD]', '[CLS]', '[SEP]'  # padding符号, bert中综合信息符号


class MyDataset(Dataset):
    @staticmethod
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

    def _build_dataset(self, file_path, tokenizer, padding_size):
        contents = []
        with open(file_path, 'r', encoding='UTF-8') as f:
            for line in tqdm(f):
                sent1, sent2, label = line.strip('\n').split('\t')
                tokens_1 = tokenizer.tokenize(sent1)
                tokens_2 = tokenizer.tokenize(sent2)
                tokens = [CLS] + tokens_1 + [SEP] + tokens_2
                sep_pos = len(tokens_1) + 1
                token_ids = tokenizer.convert_tokens_to_ids(tokens)
                token_ids, mask, seq_len, seg_id = self._pad(token_ids, padding_size, sep_pos)
                contents.append((token_ids, int(label), seq_len, mask, seg_id))
        return contents

    def __init__(self, file_path, pad_size, tokenizer, device):
        self.data_set = self._build_dataset(file_path, tokenizer, pad_size)
        self.len = len(self.data_set)
        self.device = device

    def __getitem__(self, index):
        return self.data_set[index]

    def __len__(self):
        return self.len


class CollateFunc:
    def __init__(self, device):
        self.device = device

    def collate_func(self, inputs):
        x = torch.LongTensor([_[0] for _ in inputs]).to(self.device)
        y = torch.LongTensor([_[1] for _ in inputs]).to(self.device)

        # pad前的长度(超过pad_size的设为pad_size)
        seq_len = torch.LongTensor([_[2] for _ in inputs]).to(self.device)
        mask = torch.LongTensor([_[3] for _ in inputs]).to(self.device)
        seg_id = torch.LongTensor([_[4] for _ in inputs]).to(self.device)
        return (x, seq_len, mask, seg_id), y


if __name__ == '__main__':
    from torch.utils.data.distributed import DistributedSampler
    from torch.utils.data import DataLoader
    tokenizer = BertTokenizer.from_pretrained('D:/PythonProjects/语言模型/bert-base-uncased')
    dataset = MyDataset(file_path='data/test.txt', tokenizer=tokenizer, pad_size=120, device='cuda:0')
    print(dataset)
    # sampler = DistributedSampler(dataset)
    loader = DataLoader(dataset,
                        batch_size=2,
                        collate_fn=CollateFunc(device='cuda:0').collate_func
                        )
    for data in loader:
        print(data)
        break
    pass


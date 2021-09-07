"""
todo:
    1、划分数据集 (done)
    2、数据处理 （done）
    3、模型设计 (done)
    4、训练模型 (调参中)
"""
import argparse
from train import train
from loss import CrossEntropyLoss
from model import Model
from utils import DatasetIterator, build_dataset, build_dataset_2
from pytorch_transformers import BertTokenizer
from pytorch_pretrained_bert import BertAdam
parser = argparse.ArgumentParser()

# path arguments
parser.add_argument('--bert_path', type=str,
                    default='D:/PythonProjects/语言模型/bert-base-uncased',
                    help='the pretrained bert model path')
parser.add_argument('--data_path', type=str, default='data/',
                    help='the path of the data directory')

# model parameter configurations
parser.add_argument('--batch_size', type=int, default=32,
                    help='the size of data for training')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='the step size of the optimizer, a.k.a learning rate')
parser.add_argument('--hidden_size', type=int, default=128,
                    help='the size of hidden_state vector, which encoded by the siamese network')
parser.add_argument('--num_epochs', type=int, default=10,
                    help='how many times to train on the dataset')
parser.add_argument('--pad_size_1', type=int, default=50)
parser.add_argument('--pad_size_2', type=int, default=120)

# other settings
parser.add_argument('--train', type=bool, default=True,
                    help='training mode')
parser.add_argument('--device', type=str, default='cuda:0',
                    help='code running on which device: cpu or cuda')
parser.add_argument('--dev_steps', type=int, default=20)
parser.add_argument('--method', type=int, default=0)

args = parser.parse_args()

if args.train:
    tokenizer = BertTokenizer.from_pretrained(args.bert_path)
    if args.method == 0:
        train_data, dev_data, test_data = build_dataset(args.data_path + 'train.txt',
                                                        args.data_path + 'valid.txt',
                                                        args.data_path + 'test.txt',
                                                        tokenizer=tokenizer,
                                                        padding_size=args.pad_size_1)
    else:
        train_data, dev_data, test_data = build_dataset_2(args.data_path + 'train.txt',
                                                        args.data_path + 'valid.txt',
                                                        args.data_path + 'test.txt',
                                                        tokenizer=tokenizer,
                                                        padding_size=args.pad_size_2)
    train_iter = DatasetIterator(train_data, batch_size=args.batch_size, device=args.device, method=args.method)
    dev_iter = DatasetIterator(dev_data, batch_size=args.batch_size, device=args.device, method=args.method)
    model = Model(output_size=args.hidden_size,
                  bert_path=args.bert_path,
                  device=args.device,
                  method=args.method).to(args.device)
    # dataloader = DataLoader(file_path=args.data_path + 'train.txt',
    #                         batch_size=args.batch_size,
    #                         bert_path=args.bert_path)
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=args.lr,
                         warmup=0.05,
                         t_total=len(train_iter) * args.num_epochs
                         )
    train(train_iter=train_iter,
          dev_iter=dev_iter,
          model=model,
          optimizer=optimizer,
          dev_steps=args.dev_steps,
          num_epochs=args.num_epochs
          )




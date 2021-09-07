"""
todo:
    1、划分数据集 (done)
    2、数据处理 （done）
    3、模型设计 (done)
    4、训练模型 (调参中)
"""
import argparse
from train import train
import os
from model import Model
from utils import MyDataset, CollateFunc
from pytorch_transformers import BertTokenizer
from pytorch_pretrained_bert import BertAdam
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


def main():
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
    parser.add_argument('--pad_size', type=int, default=120)

    # other settings
    parser.add_argument('--train', type=bool, default=True,
                        help='training mode')
    # parser.add_argument('--device', type=str, default='cuda:0',
    #                     help='code running on which device: cpu or cuda')
    parser.add_argument('--dev_steps', type=int, default=20)
    parser.add_argument('--n_gpu', type=int, default=1)
    parser.add_argument('--visible_gpus', type=str, default='0')

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.visible_gpus
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '2385'
    args.world_size = args.n_gpu
    torch.multiprocessing.spawn(train_proc, nprocs=args.n_gpu, args=(args,))


def train_proc(gpu, args):
    dist.init_process_group(backend='nccl',
                            init_method='env://',
                            world_size=args.world_size,
                            rank=gpu)
    device = 'cuda:' + str(gpu)
    print('using device:', device)
    torch.manual_seed(0)
    # torch.cuda.set_device(gpu)
    tokenizer = BertTokenizer.from_pretrained(args.bert_path)
    model = Model(output_size=args.hidden_size,
                  bert_path=args.bert_path,
                  device=device).to(device)
    train_data_set = MyDataset(file_path=args.data_path + 'train.txt',
                               pad_size=args.pad_size,
                               tokenizer=tokenizer,
                               device=device)
    train_sampler = DistributedSampler(train_data_set, num_replicas=args.world_size, rank=gpu)
    train_dataloader = DataLoader(dataset=train_data_set,
                                  batch_size=args.batch_size,
                                  sampler=train_sampler,
                                  collate_fn=CollateFunc(device).collate_func)
    dev_data_set = MyDataset(file_path=args.data_path + 'valid.txt',
                             pad_size=args.pad_size,
                             tokenizer=tokenizer,
                             device=device)
    dev_sampler = DistributedSampler(dev_data_set, num_replicas=args.world_size, rank=gpu)
    dev_dataloader = DataLoader(dataset=dev_data_set,
                                batch_size=args.batch_size,
                                sampler=dev_sampler,
                                collate_fn=CollateFunc(device).collate_func)
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=args.lr,
                         warmup=0.05,
                         t_total=len(train_dataloader) * args.num_epochs
                         )
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
    train(train_iter=train_dataloader,
          dev_iter=dev_dataloader,
          model=model,
          optimizer=optimizer,
          dev_steps=args.dev_steps,
          num_epochs=args.num_epochs,
          train_sampler=train_sampler)


if __name__ == '__main__':
    main()


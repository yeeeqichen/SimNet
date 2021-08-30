"""
todo:
    1、划分数据集 (done)
    2、数据处理 （done）
    3、模型设计 (done)
    4、训练模型 (done)
"""
import argparse
from train import train
from loss import CrossEntropyLoss
from model import Model
from dataloader import DataLoader
parser = argparse.ArgumentParser()

# path arguments
parser.add_argument('--bert_path', type=str,
                    default='/Users/yeeeqichen/Desktop/Python Projects/BERT_预训练模型/bert_base_uncased',
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

# other settings
parser.add_argument('--train', type=bool, default=True,
                    help='training mode')
parser.add_argument('--device', type=str, default='cpu',
                    help='code running on which device: cpu or cuda')
parser.add_argument('--log_steps', type=int, default=1000)

args = parser.parse_args()

if args.train:
    loss_func = CrossEntropyLoss(input_size=2 * args.hidden_size, device=args.device).to(args.device)
    model = Model(output_size=args.hidden_size, bert_path=args.bert_path, device=args.device).to(args.device)
    dataloader = DataLoader(file_path=args.data_path + 'train.txt',
                            batch_size=args.batch_size,
                            bert_path=args.bert_path)
    train(dataloader=dataloader,
          loss_func=loss_func,
          model=model,
          lr=args.lr,
          num_epochs=args.num_epochs,
          device=args.device,
          log_steps=args.log_steps
          )



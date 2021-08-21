"""
todo:
    1、划分数据集 (done)
    2、数据处理 （done）
    3、模型设计
    4、训练模型
"""
import argparse

parser = argparse.ArgumentParser()

# path arguments
parser.add_argument('--bert_path', type=str,
                    default='/Users/yeeeqichen/Desktop/Python Projects/BERT_预训练模型/bert_base_uncased',
                    help='the pretrained bert model path')

# model parameter configurations
parser.add_argument('--batch_size', type=int, default=32,
                    help='the size of data for training')

# other settings


args = parser.parse_args()

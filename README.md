# 文本语义相似度分析

## 简介

基于Pytorch，使用BERT进行文本语义相似度的分析，支持多GPU训练

[数据集下载](https://pan.baidu.com/s/13fx0G_mJmh76xTVxp8_xDA) ,提取码 yeee

## 模型设计

借鉴BERT预训练过程中使用的Sentence Pair Classification Task，将要分析的两个句子拼接输入BERT模型，对[CLS]标签对应的输出进行进一步处理得到分类结果

![](https://github.com/yeeeqichen/Pictures/blob/master/SentencePair.png?raw=true)

## 模型训练

```shell
python3 main.py \
  --bert_path <bert预训练模型目录> \
  --data_path <训练数据目录> \
  --n_gpu <使用几块GPU进行训练> \
  --visible_gpus <使用哪些GPU进行训练,例如 2,3>
```
## 训练结果及预训练模型下载


## Todo:

- 尝试别的预训练语言模型（如ERNIE）
- 完成test部分代码
- 封装相似度分析功能
- 上传预训练模型
- 实验数据展示
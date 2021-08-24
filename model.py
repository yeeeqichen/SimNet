import torch
from pytorch_transformers import BertModel


class Model(torch.nn.Module):
    def __init__(self, output_size, bert_path, device='cuda:0'):
        super(Model, self).__init__()
        self.device = device
        self.bert_encoder = BertModel.from_pretrained(bert_path)
        self.nonlinear_func = torch.nn.Sequential(
            torch.nn.Linear(768, 256),
            torch.nn.PReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.PReLU(),
            torch.nn.Linear(128, output_size)
        )

    def to_tensor(self, inputs):
        return torch.LongTensor(inputs).to(self.device)

    def forward(self, ids_1, masks_1, ids_2, masks_2):
        encode_1 = self.bert_encoder(
            input_ids=self.to_tensor(ids_1),
            attention_mask=self.to_tensor(masks_1)
        )
        encode_2 = self.bert_encoder(
            input_ids=self.to_tensor(ids_2),
            attention_mask=self.to_tensor(masks_2)
        )
        hidden_1 = torch.mean(encode_1[0], dim=1)
        hidden_2 = torch.mean(encode_2[0], dim=1)
        output_1 = self.nonlinear_func(hidden_1)
        output_2 = self.nonlinear_func(hidden_2)
        return output_1, output_2


if __name__ == '__main__':
    import numpy
    from loss import CrossEntropyLoss
    a = Model(
        bert_path='/Users/yeeeqichen/Desktop/Python Projects/BERT_预训练模型/bert_base_uncased',
        output_size=64,
        device='cpu',
    )
    id_1 = numpy.array(
        [[1, 2, 3], [2, 8, 0]]
    )
    mask_1 = numpy.array(
        [[1, 1, 1], [1, 1, 0]]
    )
    id_2 = numpy.array(
        [[2, 3, 0], [2, 3, 8]]
    )
    mask_2 = numpy.array(
        [[1, 1, 0], [1, 1, 1]]
    )
    label = numpy.array(
        [1, 0]
    )
    outputs = a(id_1, mask_1, id_2, mask_2)
    loss_func = CrossEntropyLoss(input_size=128, device='cpu')
    print(loss_func(outputs[0], outputs[1], label))
    pass


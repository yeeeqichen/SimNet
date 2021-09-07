import torch
from pytorch_transformers import BertModel
import torchsnooper


class Model(torch.nn.Module):
    def __init__(self, output_size, bert_path, device='cuda:0', method=0):
        super(Model, self).__init__()
        self.method = method
        self.device = device
        self.bert_encoder = BertModel.from_pretrained(bert_path)
        for param in self.bert_encoder.parameters():
            param.requires_grad = True
        self.nonlinear_func = torch.nn.Sequential(
            torch.nn.Linear(768, 256),
            torch.nn.PReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.PReLU(),
            torch.nn.Linear(128, output_size)
        )
        if self.method == 0:
            self.classification = torch.nn.Linear(2 * output_size, 2)
        else:
            self.classification = torch.nn.Linear(output_size, 2)

    # @torchsnooper.snoop()
    def forward(self, inputs):
        if self.method == 0:
            encode_1 = self.bert_encoder(
               inputs[0], attention_mask=inputs[4]
            )
            encode_2 = self.bert_encoder(
                inputs[1], attention_mask=inputs[5]
            )
            hidden_1 = torch.mean(encode_1[0], dim=1)
            hidden_2 = torch.mean(encode_2[0], dim=1)
            output_1 = self.nonlinear_func(hidden_1)
            output_2 = self.nonlinear_func(hidden_2)
            return self.classification(torch.cat((output_1, output_2), dim=-1))
        else:
            hidden_state = self.bert_encoder(
                inputs[0],
                attention_mask=inputs[2],
                token_type_ids=inputs[3]
            )[1]
            output = self.nonlinear_func(hidden_state)
            return self.classification(output)


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


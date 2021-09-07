import torch
from pytorch_transformers import BertModel
import torchsnooper


class Model(torch.nn.Module):
    def __init__(self, output_size, bert_path, device='cuda:0'):
        super(Model, self).__init__()
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
        self.classification = torch.nn.Linear(output_size, 2)

    # @torchsnooper.snoop()
    def forward(self, inputs):
        hidden_state = self.bert_encoder(
            inputs[0],
            attention_mask=inputs[2],
            token_type_ids=inputs[3]
        )[1]
        output = self.nonlinear_func(hidden_state)
        return self.classification(output)


if __name__ == '__main__':
    pass


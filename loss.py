import torch


class CrossEntropyLoss(torch.nn.Module):
    def __init__(self, input_size, device='cuda:0'):
        self.device = device
        super(CrossEntropyLoss, self).__init__()
        self.classification = torch.nn.Sequential(
            torch.nn.Linear(input_size, 2),
            torch.nn.Softmax(dim=1)
        )
        self.loss_func = torch.nn.CrossEntropyLoss()

    def forward(self, hidden_1, hidden_2, label):
        label = torch.LongTensor(label).to(self.device)
        scores = self.classification(torch.cat((hidden_1, hidden_2), dim=-1))
        loss = self.loss_func(scores, label)
        return loss


class DistanceLoss(torch.nn.Module):
    def __init__(self):
        super(DistanceLoss, self).__init__()
        pass




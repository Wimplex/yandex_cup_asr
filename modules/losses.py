import torch
import torch.nn as nn
import torch.nn.functional as F


def jaccard(intersection, union, eps=1e-15):
    return (intersection + eps) / (union - intersection + eps)


def dice(intersection, union, eps=1e-15):
    return (2. * intersection + eps) / (union + eps)


class BCESoftJaccardDice:
    def __init__(self, bce_weight=0.5, mode="dice", eps=1e-15, weight=None):
        self.nll_loss = torch.nn.BCEWithLogitsLoss(weight=weight)
        self.bce_weight = bce_weight
        self.eps = eps
        self.mode = mode

    def __call__(self, outputs, targets):
        loss = self.bce_weight * self.nll_loss(outputs, targets)

        if self.bce_weight < 1.:
            targets = (targets == 1).float()  # .half()
            outputs = torch.sigmoid(outputs)
            intersection = (outputs * targets).sum()
            union = outputs.sum() + targets.sum()
            if self.mode == "dice":
                score = dice(intersection, union, self.eps)
            elif self.mode == "jaccard":
                score = jaccard(intersection, union, self.eps)
            loss -= (1 - self.bce_weight) * torch.log(score)
        return loss


class AMSoftmaxLoss(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.4):
        """ AM Softmax Loss """
        
        super(AMSoftmaxLoss, self).__init__()
        self.s = s
        self.m = m
        self.in_features = in_features
        self.out_features = out_features
        self.fc = nn.Linear(in_features, out_features, bias=False)
        self.__init_weights()

    def __init_weights(self):
        pass

    def forward(self, x, labels):        
        for W in self.fc.parameters(): W = F.normalize(W, dim=1)
        x = F.normalize(x, dim=1)

        wf = self.fc(x)
        numerator = self.s * (torch.diagonal(wf.transpose(0, 1)[labels]) - self.m)
        excl = torch.cat([torch.cat((wf[i, :y], wf[i, y+1:])).unsqueeze(0) for i, y in enumerate(labels)], dim=0)
        denominator = torch.exp(numerator) + torch.sum(torch.exp(self.s * excl), dim=1)
        l = numerator - torch.log(denominator)
        return -torch.mean(l)
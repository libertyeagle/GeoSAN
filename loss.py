import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryCELoss(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)

    def forward(self, pos_score, neg_score, probs):
        loss = -F.logsigmoid(pos_score.squeeze()) + torch.sum(F.softplus(neg_score) / neg_score.size(2), dim=-1)
        return loss


class  BPRLoss(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)

    def forward(self, pos_score, neg_score, probs):
        # only supports one negative sample
        loss = -F.logsigmoid(pos_score.squeeze() - neg_score.squeeze())
        return loss


class WeightedBinaryCELoss(nn.Module):
    def __init__(self, temperature=1.0):
        nn.Module.__init__(self)
        self.temperature = temperature

    def forward(self, pos_score, neg_score, probs):
        weight = F.softmax(neg_score / self.temperature, -1)
        loss = -F.logsigmoid(pos_score.squeeze()) + torch.sum(F.softplus(neg_score) * weight, dim=-1) 
        return loss


class WeightedProbBinaryCELoss(nn.Module):
    def __init__(self, temperature=1.0):
        nn.Module.__init__(self,)
        self.temperature = temperature

    def forward(self, pos_score, neg_score, probs):
        weight = F.softmax(neg_score / self.temperature - torch.log(probs), -1)
        loss = -F.logsigmoid(pos_score.squeeze()) + torch.sum(F.softplus(neg_score) * weight, dim=-1)
        return loss
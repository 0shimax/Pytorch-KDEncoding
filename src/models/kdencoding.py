import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.nn.parameter import Parameter


def calculate_Encoding(pai, temperature):
    o_hat = F.softmax(pai/temperature, dim=2)
    _, indices = torch.max(o_hat, 2)
    # o_one_hoet = torch.zeros_like(self.pai_concept)
    # o_one_hoet.scatter_(1, indices, 1)
    # with torch.no_grad():
    #     tmp = o_one_hoet - o_hat_concept
    return indices


class SimpleKDEncoding(nn.Module):

    def __init__(self, voc_size, k_concept, k_character, D):
        super().__init__()
        self.pai_concept = Parameter(torch.randn(voc_size, D, k_concept))
        self.pai_character = Parameter(torch.randn(voc_size, D, k_character))
        self.temperature = 1.6

    def forward(self, voc_idxs, debug=False):
        out = []
        for vi in torch.transpose(voc_idxs, 1, 0):
            o_enc_concept = calculate_Encoding(
                self.pai_concept[vi], self.temperature)
            o_enc_character = calculate_Encoding(
                self.pai_character[vi], self.temperature)
            out.append(o_enc_concept + o_enc_character)
            if debug:
                print(o_enc_concept)
                print(o_enc_character)
        return torch.stack(out, dim=0)

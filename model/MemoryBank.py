from __future__ import print_function
from __future__ import absolute_import
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from .wordrep import WordRep
import numpy as np 

class MemoryBank(nn.Module):
    def __init__(self, data):
        super(MemoryBank, self).__init__()
        # print("build memory network...")
        self.gpu = data.HP_gpu

        total_num = data.word_idx
        self.word_dim = data.word_emb_dim
        self.word_mat = data.word_mat
        self.sent_dim = data.HP_hidden_dim

        bankmem = torch.Tensor(total_num, self.sent_dim)
        nn.init.uniform_(bankmem, -1, 1)     
        self.bankmem = nn.Parameter(bankmem, requires_grad = False)
        self.bankmem.data[0] = torch.zeros(self.sent_dim)
        
        wordmem = torch.Tensor(total_num, self.word_dim)
        nn.init.uniform_(wordmem, -1, 1)
        self.wordmem = nn.Parameter(wordmem, requires_grad = False) 
        self.wordmem.data[0] = torch.zeros(data.word_emb_dim)

        self.dropout = nn.Dropout(data.mem_bank_dropout)
        self.idx = None
    

    def forward(self, word_idx, word_embs):
        """
            input:
                word_idx: (total_num, )
                word_embs: (total_num, hidden_dim)
            output:
                Variable(total_num, hidden_dim)
        """
        if self.idx is None:
            return word_embs.new_full((word_embs.size(0), self.sent_dim), fill_value=0)
        
        num = word_embs.size(0)
        idx = [list(self.word_mat[i][:500]) for i in word_idx]
        word_idx_len = list(map(len, idx))
        max_word_idx_len = max(word_idx_len)
        idx = [idx[i] + [0] * (max_word_idx_len - len(idx[i])) for i in range(len(idx))]
        idx = torch.tensor(idx).type_as(self.idx).view(-1)

        mask = torch.zeros((num, max_word_idx_len), requires_grad=False).type_as(self.idx)
        for i in range(num):
            mask[i, :word_idx_len[i]] = torch.Tensor([1]*word_idx_len[i])

        score = torch.bmm(F.normalize(word_embs.unsqueeze(1), dim=-1), F.normalize(self.wordmem[idx].view(word_embs.size(0), max_word_idx_len, word_embs.size(1)), dim=-1).transpose(2,1)).squeeze(1)
        score = self.partial_softmax(score, mask, 1)
                 
        doc_hidden = torch.bmm(score.unsqueeze(1), self.bankmem[idx].view(num, max_word_idx_len, -1)).squeeze(1)
        
        return doc_hidden
        
    def update(self, idx, word_embs, hidden_embs):
        """
            input:
                idx: update idx (update_num,)
                word_embs:  (update_num, word_dim)
                hidden_embs:  (update_num, hiden_dim)
        """
        self.bankmem.data[idx] = hidden_embs.data 
        self.wordmem.data[idx] = word_embs.data

    def make_idx(self, idx_list):
        """
            input:
                idx: update idx (update_num,)
        """
        if len(idx_list) == 0:
            self.idx = None
        else:
            self.idx = idx_list

    def partial_softmax(self, inputs, mask, dim):
        """
            input:
                inputs: (batch_size, sent_len)
            output:
                Variable(batch_size, sent_len)
        """
        exp_inp = torch.exp(inputs)
        exp_inp_weighted = torch.mul(exp_inp, mask.float())
        exp_inp_sum = torch.sum(exp_inp_weighted, dim=dim, keepdim=True)
        partial_softmax_score = torch.div(exp_inp_weighted, exp_inp_sum)
        return partial_softmax_score
        
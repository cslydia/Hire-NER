from __future__ import print_function
from __future__ import absolute_import
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from .wordrep import WordRep
import numpy as np 
from .MemoryBank import MemoryBank
from .SentenceRep import SentenceRep

class WordSequence(nn.Module):
    def __init__(self, data):
        super(WordSequence, self).__init__()
        print("build word sequence feature extractor: %s..."%(data.word_feature_extractor))
        self.gpu = data.HP_gpu
        self.data = data
        self.use_char = data.use_char
        self.label_alphabet_size = data.label_alphabet.size()
        self.droplstm = nn.Dropout(data.rnn_dropout)
        self.bilstm_flag = data.HP_bilstm
        self.lstm_layer = data.HP_lstm_layer
        self.wordrep = WordRep(data)
        self.input_size = data.word_emb_dim + data.global_hidden_dim
        if self.use_char:
            kernel_type = data.HP_intNet_kernel_type
            char_dim = data.HP_char_hidden_dim
            self.input_size += int( (data.HP_intNet_layer - 1) // 2 * char_dim * kernel_type + char_dim * 2 * kernel_type)
        
        if self.bilstm_flag:
            lstm_hidden = data.HP_hidden_dim // 2
        else:
            lstm_hidden = data.HP_hidden_dim

        self.word_feature_extractor = data.word_feature_extractor
        if self.word_feature_extractor == "GRU":
            self.lstm = nn.GRU(self.input_size, lstm_hidden, num_layers=self.lstm_layer, batch_first=True, bidirectional=self.bilstm_flag)
        elif self.word_feature_extractor == "LSTM":
            self.lstm = nn.LSTM(self.input_size, lstm_hidden, num_layers=self.lstm_layer, batch_first=True, bidirectional=self.bilstm_flag)
        
        # document-level
        self.mem_alpha = data.mem_bank_alpha
        self.mem_bank = MemoryBank(data)
        self.mem2tag = nn.Linear(data.HP_hidden_dim, data.label_alphabet_size)
      
        # sentence-level
        self.label2cnn = nn.Conv1d(self.label_alphabet_size, self.label_alphabet_size, kernel_size=data.global_kernel_size, padding=data.global_kernel_size//2)
        self.sentrep = SentenceRep(data)
       
        self.hidden2tag = nn.Linear(data.HP_hidden_dim, data.label_alphabet_size)
        
        if self.gpu:
            self.droplstm = self.droplstm.cuda()
            self.hidden2tag = self.hidden2tag.cuda()
            self.mem2tag = self.mem2tag.cuda()  
            self.lstm = self.lstm.cuda()
            self.mem_bank = self.mem_bank.cuda()
            self.label2cnn = self.label2cnn.cuda()

    def partial_softmax(self, inputs, mask, dim):
        """
            input:
                Att_v: (batch_size, sent_len)
            output:
                Variable(batch_size, sent_len)
        """
        exp_inp = torch.exp(inputs)
        exp_inp_weighted = torch.mul(exp_inp, mask.float())
        exp_inp_sum = torch.sum(exp_inp_weighted, dim=dim, keepdim=True)
        partial_softmax_score = torch.div(exp_inp_weighted, exp_inp_sum)
        return partial_softmax_score


    def get_sentence_embedd(self, word_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover, label_embs, mask):
        """
            input:
                word_inputs: (batch_size, sent_len)
                word_seq_lengths: list of batch_size, (batch_size,1)
                char_inputs: (batch_size*sent_len, word_length)
                char_seq_lengths: list of whole batch_size for char, (batch_size*sent_len, 1)
                char_seq_recover: variable which records the char order information, used to recover char order
                label_embs: cosine distance between word embeddings and label embeddings 
            output:
                Variable(batch_size, sent_len, sent_hidden_dim)
        """
        batch_size, seq_len = word_inputs.size()
        sentence_hidden = self.sentrep(word_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover)
        label_feature = F.relu(self.label2cnn(label_embs.transpose(2,1).contiguous())).transpose(2,1).contiguous()
        label_feature_max, _ = torch.max(label_feature, dim=-1)
        label_feature_max = self.partial_softmax(label_feature_max, mask, dim=1)
        sentence_represent = torch.bmm(label_feature_max.unsqueeze(1), sentence_hidden).repeat(1, seq_len, 1) 
        return sentence_represent


    def get_document_embedd(self, word_inputs, hidden_embs, word_embs, idx_inputs, mask):
        """
            input:
                word_inputs: (batch_size, sent_len)
                hidden_embs: (batch_size, sent_len, hidden_dim)
                word_embs: (batch_size, sent_len, word_dim)
                idx_inputs: index of words for memory network (batch_size, sent_len)
            output:
                Variable(batch_size, sent_len, doc_hidden_dim)
        """
        batch_size, seq_len = word_inputs.size()
        idx_inputs = torch.masked_select(idx_inputs, mask)
        word_inputs = word_inputs.unsqueeze(-1)

        total_len = len(idx_inputs)
        hidden_dim, word_dim = hidden_embs.size(-1), word_embs.size(-1)

        hidden_inp = torch.zeros(total_len, hidden_dim).type_as(hidden_embs)
        word_inp = torch.zeros(total_len, word_dim).type_as(word_embs)
        inp = torch.zeros(total_len, 1).type_as(word_inputs)

        start_idx = 0
        for i in range(batch_size):
            hidden_inp[start_idx: start_idx + mask[i].sum()] = hidden_embs[i][:mask[i].sum()]
            word_inp[start_idx: start_idx + mask[i].sum()] = word_embs[i][:mask[i].sum()]
            inp[start_idx: start_idx + mask[i].sum()] = word_inputs[i][:mask[i].sum()]
            start_idx = start_idx + mask[i].sum()
       
        document_hidden = self.mem_bank(inp, word_inp)
        self.mem_bank.update(idx_inputs, word_inp, hidden_inp)

        document_represent = torch.zeros_like(hidden_embs) 
        start_idx = 0 
        for idx in range(batch_size):
            document_represent[i, :mask[i].sum()] = document_hidden[start_idx: start_idx+mask[i].sum()]
            start_idx = start_idx + mask[i].sum()
        document_represent = self.mem2tag(document_represent)

        return document_represent



    def forward(self, word_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover, mask, idx_inputs=None):
        """
            input:
                word_inputs: (batch_size, sent_len)
                feature_inputs: [(batch_size, sent_len), ...] list of variables
                word_seq_lengths: list of batch_size, (batch_size,1)
                char_inputs: (batch_size*sent_len, word_length)
                char_seq_lengths: list of whole batch_size for char, (batch_size*sent_len, 1)
                char_seq_recover: variable which records the char order information, used to recover char order
            output:
                Variable(batch_size, sent_len, hidden_dim)
        """
        
        batch_size, seq_len = word_inputs.size()
        word_represent, label_embs, word_embs = self.wordrep(word_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover)
        
        sentence_represent = self.get_sentence_embedd(word_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover, label_embs, mask)
        word_represent = torch.cat([word_represent, sentence_represent], 2)

        packed_words = pack_padded_sequence(word_represent, word_seq_lengths.cpu().numpy(), True)
        hidden = None
        lstm_out, hidden = self.lstm(packed_words, hidden)
        lstm_out, _ = pad_packed_sequence(lstm_out)
        feature_out = self.droplstm(lstm_out.transpose(1,0).contiguous())

        outputs = self.hidden2tag(feature_out)

        doc_represent = self.get_document_embedd(word_inputs, feature_out, word_embs, idx_inputs, mask)            
        outputs = outputs * (1 - self.mem_alpha) + doc_represent * self.mem_alpha

        return outputs 





from __future__ import print_function
from __future__ import absolute_import
import torch
import torch.nn as nn
import numpy as np
from .IntNet import IntNet
import torch.nn.functional as F   
import numpy as np 

seed_num=42
torch.manual_seed(seed_num)
np.random.seed(seed_num)
torch.cuda.manual_seed(seed_num)

class WordRep(nn.Module):
    def __init__(self, data):
        super(WordRep, self).__init__()
        # print("build word representation...")
        self.gpu = data.HP_gpu
        self.use_char = data.use_char
        self.batch_size = data.HP_batch_size
        
        if self.use_char:
            self.char_hidden_dim = data.HP_char_hidden_dim
            self.char_embedding_dim = data.char_emb_dim
            self.char_feature = IntNet(data.char_alphabet.size(), self.char_embedding_dim, data.HP_intNet_layer, data.HP_intNet_kernel_type, data.HP_dropout, self.char_hidden_dim, self.gpu)
           
        self.embedding_dim = data.word_emb_dim
        self.drop = nn.Dropout(data.HP_dropout)
        self.word_embedding = nn.Embedding(data.word_alphabet.size(), self.embedding_dim)
        if data.pretrain_word_embedding is not None:
            self.word_embedding.weight.data.copy_(torch.from_numpy(data.pretrain_word_embedding))
        else:
            self.word_embedding.weight.data.copy_(torch.from_numpy(self.random_embedding(data.word_alphabet.size(), self.embedding_dim)))


        self.label_embedding_dim = data.word_emb_dim
        self.label_alphabet_size = data.label_alphabet.size()
        self.label_embedding = nn.Embedding(self.label_alphabet_size, self.label_embedding_dim)
        self.label_type = torch.from_numpy(np.array([i for i in range(self.label_alphabet_size)]))
        if data.pretrain_label_embedding is not None:
            self.label_embedding.weight.data.copy_(torch.from_numpy(data.pretrain_label_embedding))
        else:
            self.label_embedding.weight.data.copy_(torch.from_numpy(self.random_embedding(self.label_alphabet_size, self.label_embedding_dim)))

        self.cos_embs = nn.CosineSimilarity(dim=-1)

        if self.gpu:
            self.drop = self.drop.cuda()
            self.word_embedding = self.word_embedding.cuda()
            self.label_embedding = self.label_embedding.cuda()
            self.cos_embs = self.cos_embs.cuda()
            self.label_type = self.label_type.cuda()
         

    def random_embedding(self, vocab_size, embedding_dim):
        pretrain_emb = np.empty([vocab_size, embedding_dim])
        scale = np.sqrt(3.0 / embedding_dim)
        for index in range(vocab_size):
            pretrain_emb[index,:] = np.random.uniform(-scale, scale, [1, embedding_dim])
        return pretrain_emb

   
    def forward(self, word_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover, sentence_level=False):
        """
            input:
                word_inputs: (batch_size, sent_len)
                word_seq_lengths: list of batch_size, (batch_size,1)
                char_inputs: (batch_size*sent_len, word_length)
                char_seq_lengths: list of whole batch_size for char, (batch_size*sent_len, 1)
                char_seq_recover: variable which records the char order information, used to recover char order
                sentence_level: label embedding attention or not
            output:
                Variable(batch_size, sent_len, hidden_dim)
        """
        batch_size, sent_len = word_inputs.size()[:2]
        word_embs =  self.word_embedding(word_inputs)
        word_list = [word_embs]
        orig_word_embs = word_embs
      
        if self.use_char:
            char_features = self.char_feature.get_last_hiddens(char_inputs, char_seq_lengths.cpu().numpy())
            char_features = char_features[char_seq_recover]
            char_features = char_features.view(batch_size,sent_len,-1)
            word_list.append(char_features) 

        if not sentence_level:          
            label_embs = self.label_embedding(self.label_type) 
            emb_batch = orig_word_embs.unsqueeze(2).repeat(1, 1, self.label_alphabet_size, 1) 
            new_label_emb = label_embs.unsqueeze(0).unsqueeze(0).repeat(batch_size, sent_len, 1, 1) 
            LS_embs = self.drop(self.cos_embs(emb_batch, new_label_emb).view(batch_size, sent_len, -1))
        else:
            LS_embs = None
            
        word_embs = torch.cat(word_list, 2)
        word_represent = self.drop(word_embs)
        return word_represent, LS_embs, orig_word_embs

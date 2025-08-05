import torch.nn as nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence,pad_sequence
from .wordrep import WordRep
from model.attention import XCA,XCA_label_1,LPI,attention_xca_stack,attention_xca_xca_label_stack
import math




class WordSequence(nn.Module):
    def __init__(self, data):
        super(WordSequence, self).__init__()
        print("build word sequence feature extractor: %s..." % (data.word_feature_extractor))
        self.gpu = data.HP_gpu
        self.use_char = data.use_char


        self.bilstm_flag = data.HP_bilstm
        self.num_of_lstm_layer = data.HP_lstm_layer
        # word embedding
        self.wordrep = WordRep(data)
        alpha_reverse = dict(data.word_alphabet.iteritems())
        # print('alpha_reverse',alpha_reverse)
        labelalpha_reverse = dict(data.label_alphabet.iteritems())
        # print('labelalpha_reverse',labelalpha_reverse)
        # for cbox dataset
        # labelalpha_reverse['TAG'] = 63#############
        # labelalpha_reverse['TGA'] = 64###############
        self.input_size = data.word_emb_dim
        if self.use_char:
            self.input_size += data.HP_char_hidden_dim
            if data.char_feature_extractor == "ALL":
                self.input_size += data.HP_char_hidden_dim
        for idx in range(data.feature_num):
            self.input_size += data.feature_emb_dims[idx]
        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        if self.bilstm_flag:
            lstm_hidden = data.HP_hidden_dim // 2
        else:
            lstm_hidden = data.HP_hidden_dim

        self.word_feature_extractor = data.word_feature_extractor

        self.lstm_first = nn.LSTM(self.input_size, lstm_hidden, num_layers=2, batch_first=True,
                                  bidirectional=self.bilstm_flag)
        
        self.lstm_first_1 = nn.LSTM(self.input_size, lstm_hidden, num_layers=2, batch_first=True,
                                  bidirectional=self.bilstm_flag)
        

        self.xca_attention = XCA(data.HP_hidden_dim, num_heads=20, qkv_bias=False, gpu=self.gpu)
        self.xca_attention_1 = XCA_label_1(data.HP_hidden_dim, num_heads=20, qkv_bias=False, gpu=self.gpu)
        self.xca_attention_2 = XCA(data.HP_hidden_dim, num_heads=20, qkv_bias=False, gpu=self.gpu,attn_drop=0.,proj_drop=0.)
        self.xca_attention_3 = XCA(data.HP_hidden_dim, num_heads=20, qkv_bias=False, gpu=self.gpu,attn_drop=0.,proj_drop=0.)

        
        self.LPI = LPI(8)

        self.ff0 = nn.Sequential(nn.Linear(data.HP_hidden_dim,data.HP_hidden_dim),nn.ReLU(),
                                nn.Linear(data.HP_hidden_dim,data.HP_hidden_dim))

        self.norm1 = nn.LayerNorm(data.HP_hidden_dim)
        self.norm2 = nn.LayerNorm(data.HP_hidden_dim)
        self.norm3 = nn.LayerNorm(data.HP_hidden_dim)

        self.layer = 1
        
        self.lstm_first_stack = nn.ModuleList(
            [nn.LSTM(self.input_size, lstm_hidden, num_layers=2, batch_first=True,
                                  bidirectional=self.bilstm_flag) for _ in range(self.layer)])
        
        self.attention_stack = nn.ModuleList(
            [attention_xca_stack(data) for _ in range(self.layer)])

        self.ff_stack = nn.ModuleList(
            [(nn.Sequential(nn.Linear(data.HP_hidden_dim,data.HP_hidden_dim),nn.ReLU(),
                                nn.Linear(data.HP_hidden_dim,data.HP_hidden_dim))) for _ in range(self.layer)])


        if self.gpu:

            self.lstm_first = self.lstm_first.cuda()
            self.lstm_first_1 = self.lstm_first_1.cuda()
            self.LPI = self.LPI.cuda()
            self.xca_attention = self.xca_attention.cuda()
            self.xca_attention_1 = self.xca_attention_1.cuda()
            self.xca_attention_2 = self.xca_attention_2.cuda()
            self.xca_attention_3 = self.xca_attention_3.cuda()
            self.lstm_first_stack.cuda()


            self.norm1 = self.norm1.cuda()
            self.norm2 = self.norm2.cuda()
            self.norm3 = self.norm3.cuda()
            self.attention_stack = self.attention_stack.cuda()
            self.ff_stack = self.ff_stack.cuda()

            self.ff0 = self.ff0.cuda()




    def forward(self, word_inputs, feature_inputs, word_seq_lengths, char_inputs, char_seq_lengths, char_seq_recover,
                input_label_seq_tensor):
        """
            input:
                word_inputs: (batch_size, sent_len)
                word_seq_lengths: list of batch_size, (batch_size,1)
                char_inputs: (batch_size*sent_len, word_length)
                char_seq_lengths: list of whole batch_size for char, (batch_size*sent_len, 1)
                char_seq_recover: variable which records the char order information, used to recover char order
                label_size: nubmer of label
            output:
                Variable(batch_size, sent_len, hidden_dim)
        """

        word_represent, label_embs = self.wordrep(word_inputs, feature_inputs, word_seq_lengths, char_inputs,
                                                  char_seq_lengths, char_seq_recover, input_label_seq_tensor)

        """
        First LSTM layer (input word only)
        """

        lstm_out = word_represent
        lstm_out_1 = word_represent 
               
        hidden = None
        lstm_out, hidden = self.lstm_first(lstm_out, hidden)
        
        hidden = None
        lstm_out_1, hidden = self.lstm_first_1(lstm_out_1, hidden)
        
        xca_out = self.xca_attention(lstm_out) + lstm_out
        xca_out_1 = self.xca_attention_1(lstm_out_1,label_embs) + lstm_out_1
        xca_out_2 = self.xca_attention_2(xca_out_1)+xca_out_1
        xca_out_2 = self.norm3(xca_out_2)
                
        out = xca_out+xca_out_2
        out = self.norm1(out)
        out = self.LPI(out)  + out
        
        out = self.norm2(out)
        out = self.ff0(out) + out
        return out


    
class PositionalEmbedding(nn.Module):
    """
    Standard positional embedding.
    From the paper "Attention is all you need".
    Changed the constant from 10k to 100k, since this may be better for longer sequence lengths.
    """
    def __init__(self, channels):
        super(PositionalEmbedding, self).__init__()
        inv_freq = 1. / (100000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, tensor):
        pos = torch.arange(tensor.shape[1], device=tensor.device).type(self.inv_freq.type())
        sin_inp = torch.einsum("i,j->ij", pos, self.inv_freq)
        emb = torch.cat((sin_inp.sin(), sin_inp.cos()), dim=-1)
        return emb[None,:,:]

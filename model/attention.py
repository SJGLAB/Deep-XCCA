from __future__ import print_function
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import *
import math
import numpy as np
from torch.nn.utils.rnn import PackedSequence
from torch.nn.utils.rnn import pad_sequence
from Bio.Seq import Seq
from typing import Tuple
import itertools
from collections import defaultdict



class XCA(nn.Module):
    def __init__(self, dim, num_heads=6, qkv_bias=False, gpu=True, attn_drop=0.,proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads,1,1))
        self.qkv = nn.Linear(dim,dim*3,bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim,dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.gpu = gpu
        if self.gpu:
            # self.temperature = self.temperature.cuda()
            self.qkv = self.qkv.cuda()
            self.attn_drop = self.attn_drop.cuda()
            self.proj = self.proj.cuda()
            self.proj_dropt = self.proj_drop.cuda()
        
    def forward(self,x):
        B,N,C = x.shape
        qkv = self.qkv(x).reshape(B,N,3,self.num_heads,C//self.num_heads).permute(2,0,3,4,1)
        q,k,v = qkv.unbind(0)
        
        q = torch.nn.functional.normalize(q,dim=-1)
        k = torch.nn.functional.normalize(k,dim=-1)
        attn = (q@k.transpose(-2,-1)) * self.temperature
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn@v).permute(0,3,1,2).reshape(B,N,C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
class XCA_label(nn.Module):
    def __init__(self, dim, num_heads=6, qkv_bias=False, gpu=True, attn_drop=0.,proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads,1,1))
        self.qkv = nn.Linear(dim,dim*2,bias=qkv_bias)
        # self.v = nn.Linear(65,dim,bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim,dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.gpu = gpu
        if self.gpu:
            # self.temperature = self.temperature.cuda()
            self.qkv = self.qkv.cuda()
            self.attn_drop = self.attn_drop.cuda()
            self.proj = self.proj.cuda()
            self.proj_dropt = self.proj_drop.cuda()
        
    def forward(self,x,y): 
        B,N,C = x.shape
        B_1,N_1,C_1 = y.shape
        qkv = self.qkv(x).reshape(B,N,2,self.num_heads,C//self.num_heads).permute(2,0,3,4,1)
        v = y.reshape(B_1,N_1,self.num_heads,C_1//self.num_heads).permute(0,2,3,1)
        q,k = qkv.unbind(0)
        
        q = torch.nn.functional.normalize(q,dim=-1)
        k = torch.nn.functional.normalize(k,dim=-1)
        attn = (q@k.transpose(-2,-1)) * self.temperature
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn@v)
        # print(x.shape)
        x = x.permute(0,3,1,2).reshape(B_1,N_1,C_1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
class XCA_label_1(nn.Module):
    def __init__(self, dim, num_heads=6, qkv_bias=False, gpu=True, attn_drop=0.,proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads,1,1))
        self.kv = nn.Linear(dim,dim*2,bias=qkv_bias)
        self.q = nn.Linear(dim,dim,bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim,dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.gpu = gpu
        if self.gpu:
            # self.temperature = self.temperature.cuda()
            self.kv = self.kv.cuda()
            self.attn_drop = self.attn_drop.cuda()
            self.proj = self.proj.cuda()
            self.proj_dropt = self.proj_drop.cuda()
        
    def forward(self,x,y): 
        #x:lstm emb
        #y:label emb
        B,N,C = x.shape
        B_1,N_1,C_1 = y.shape
        q = self.q(x).reshape(B,N,self.num_heads,C//self.num_heads).permute(0,2,1,3)
        kv = self.kv(y).reshape(B_1,N_1,2,self.num_heads,C_1//self.num_heads).permute(2,0,3,1,4)

        k,v = kv.unbind(0)
        
        q = torch.nn.functional.normalize(q,dim=-1)
        k = torch.nn.functional.normalize(k,dim=-1)
        attn = (q@k.transpose(-2,-1)) * self.temperature
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn@v)
        # print(x.shape)
        x = x.permute(0,3,1,2).reshape(B,N,C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
class LPI(nn.Module):
    """
    Local Patch Interaction module that allows explicit communication between tokens in 3x3 windows
    to augment the implicit communcation performed by the block diagonal scatter attention.
    Implemented using 2 layers of separable 3x3 convolutions with GeLU and BatchNorm2d
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU,
                 drop=0., kernel_size=3):
        super().__init__()
        self.C = in_features
        out_features = out_features or in_features

        padding = kernel_size // 2

        self.conv1 = torch.nn.Conv2d(in_features, out_features, kernel_size=kernel_size,
                                     padding=padding, groups=out_features)
        self.act = act_layer()
        self.bn = nn.SyncBatchNorm(in_features)
        self.conv2 = torch.nn.Conv2d(in_features, out_features, kernel_size=kernel_size,
                                     padding=padding, groups=out_features)

    def forward(self, x):
        B, N, C = x.shape
        x = x.reshape(B, self.C, N, C//self.C)
        x = self.conv1(x)
        x = self.act(x)
        x = self.bn(x)
        x = self.conv2(x)
        x = x.reshape(B, C, N).permute(0, 2, 1)

        return x

class attention_xca_stack(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, data):
        super(attention_xca_stack, self).__init__()

        self.xca_attention = XCA(data.HP_hidden_dim, num_heads=8, qkv_bias=False)
        
        self.gpu = data.HP_gpu
        if self.gpu:
            self.xca_attention = self.xca_attention.cuda()

    def forward(self, lstm_out, label_embs):
        lstm_out = self.xca_attention(lstm_out)+lstm_out
        return lstm_out
    
class attention_xca_xca_label_stack(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, data):
        super(attention_xca_xca_label_stack, self).__init__()

        self.xca_attention = XCA(data.HP_hidden_dim, num_heads=8, qkv_bias=False)
        self.xca_attention_label = XCA_label(data.HP_hidden_dim, num_heads=8, qkv_bias=False)
        
        self.gpu = data.HP_gpu
        if self.gpu:
            self.xca_attention = self.xca_attention.cuda()
            self.xca_attention_label = self.xca_attention_label.cuda()

    def forward(self, lstm_out, label_embs):
        lstm_out = self.xca_attention_label(label_embs,lstm_out)+lstm_out
        lstm_out = self.xca_attention(lstm_out)+lstm_out
        return lstm_out
    
class attention_xca_xca_label_1_stack(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, data):
        super(attention_xca_xca_label_1_stack, self).__init__()

        self.xca_attention = XCA(data.HP_hidden_dim, num_heads=8, qkv_bias=False)
        self.xca_attention_label = XCA_label_1(data.HP_hidden_dim, num_heads=8, qkv_bias=False)
        
        self.gpu = data.HP_gpu
        if self.gpu:
            self.xca_attention = self.xca_attention.cuda()
            self.xca_attention_label = self.xca_attention_label.cuda()

    def forward(self, lstm_out, label_embs):
        lstm_out = self.xca_attention(lstm_out)+lstm_out
        lstm_out = self.xca_attention_label(lstm_out, label_embs)+lstm_out
        return lstm_out
    

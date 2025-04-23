import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.modules.rnn import RNNCellBase
from torch.nn.parameter import Parameter
import math
from torch.nn import LayerNorm

class LSTMCell(RNNCellBase):
    '''
    Copied from torch.nn
    '''
    def __init__(self, input_size, hidden_size):
        super(LSTMCell, self).__init__(input_size, hidden_size,bias=True,num_chunks=4)

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = Parameter(torch.Tensor(4 * hidden_size, input_size))
        self.weight_hh = Parameter(torch.Tensor(4 * hidden_size, hidden_size))

        self.bias_ih = Parameter(torch.Tensor(4 * hidden_size))
        self.bias_hh = Parameter(torch.Tensor(4 * hidden_size))


        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
    def forward(self, input, hx,cx,update_mode=''):


        gates = F.linear(input, self.weight_ih, self.bias_ih) + F.linear(hx, self.weight_hh, self.bias_hh)

        ingate, forgetgate, cellgate, outgate_ = gates.chunk(4, 1)   #####todo:no layernormalization


        ingate = F.sigmoid(ingate)
        forgetgate = F.sigmoid(forgetgate)
        cellgate = F.tanh(cellgate)
        outgate = F.sigmoid(outgate_ )

        cy = forgetgate * cx +ingate * cellgate
        hy = outgate * F.tanh(cy)

        return outgate,hy, cy


class HyperLSTM_re_trajectron(RNNCellBase):
    def __init__(self,input_size,hyper_feat_size, hidden_size,hidden_size_hat,z_size):
        super(HyperLSTM_re_trajectron, self).__init__(input_size, hidden_size, bias=True, num_chunks=4)

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.weight_ih =  Parameter(torch.Tensor(4 * hidden_size, input_size))
        self.weight_hh = Parameter(torch.Tensor(4 * hidden_size, hidden_size))

        self.bias_ih = None
        self.bias_hh = None

        # self.hyperlstm=LSTMCell(input_size+hidden_size,hidden_size_hat)
        self.h_hat_hyper=nn.Linear(hyper_feat_size,hidden_size_hat)
        self.c_hat_hyper = nn.Linear(hyper_feat_size , hidden_size_hat)
        self.h_hat_h=nn.Linear(hidden_size_hat,z_size*4)
        self.h_hat_x = nn.Linear(hidden_size_hat, z_size*4)
        self.h_hat_b = nn.Linear(hidden_size_hat, z_size*4,bias=False)

        self.h_z=nn.Linear(z_size*4,hidden_size*4,bias=False)
        self.x_z=nn.Linear(z_size*4,hidden_size*4,bias=False)
        self.b_z=nn.Linear(z_size*4,hidden_size*4)

        self.layernorm=LayerNorm(4*hidden_size)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self,input,contrast_feature,hx,cx,h_hat,c_hat):

        x_hat=contrast_feature
        # x_hat=input
        # _,h_hat_next,c_hat_next=self.hyperlstm(x_hat,h_hat,c_hat)
        h_hat_next=self.h_hat_hyper(x_hat)
        c_hat_next=self.c_hat_hyper(x_hat)

        z_h=self.h_hat_h(h_hat)
        z_x=self.h_hat_x(h_hat)
        z_b=self.h_hat_b(h_hat)


        d_h=self.h_z(z_h)
        d_x=self.x_z(z_x)
        bias=self.b_z(z_b)   ######todo: split


        gates = torch.mul(d_x,F.linear(input, self.weight_ih)) + torch.mul(d_h,F.linear(hx, self.weight_hh)) + bias
        gates = self.layernorm(gates)

        ingate, forgetgate, cellgate, outgate_ = gates.chunk(4, 1)

        ingate = F.sigmoid(ingate)
        forgetgate = F.sigmoid(forgetgate)
        cellgate = F.tanh(cellgate)
        outgate = F.sigmoid(outgate_ )

        cy = forgetgate * cx +ingate * cellgate
        hy = outgate * F.tanh(cy)


        return outgate, hy, cy,h_hat_next,c_hat_next


class HyperLSTM_re(RNNCellBase):
    def __init__(self,input_size, hidden_size,hidden_size_hat,z_size):
        super(HyperLSTM_re, self).__init__(input_size, hidden_size, bias=True, num_chunks=4)

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.weight_ih =  Parameter(torch.Tensor(4 * hidden_size, input_size))
        self.weight_hh = Parameter(torch.Tensor(4 * hidden_size, hidden_size))

        self.bias_ih = None
        self.bias_hh = None

        # self.hyperlstm=LSTMCell(input_size+hidden_size,hidden_size_hat)
        self.h_hat_hyper=nn.Linear(input_size+hidden_size,hidden_size_hat)
        self.c_hat_hyper = nn.Linear(input_size + hidden_size, hidden_size_hat)
        self.h_hat_h=nn.Linear(hidden_size_hat,z_size*4)
        self.h_hat_x = nn.Linear(hidden_size_hat, z_size*4)
        self.h_hat_b = nn.Linear(hidden_size_hat, z_size*4,bias=False)

        self.h_z=nn.Linear(z_size*4,hidden_size*4,bias=False)
        self.x_z=nn.Linear(z_size*4,hidden_size*4,bias=False)
        self.b_z=nn.Linear(z_size*4,hidden_size*4)

        self.layernorm=LayerNorm(4*hidden_size)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self,input,contrast_feature,hx,cx,h_hat,c_hat):

        x_hat=contrast_feature
        # x_hat=input
        # _,h_hat_next,c_hat_next=self.hyperlstm(x_hat,h_hat,c_hat)
        h_hat_next=self.h_hat_hyper(x_hat)
        c_hat_next=self.c_hat_hyper(x_hat)

        z_h=self.h_hat_h(h_hat)
        z_x=self.h_hat_x(h_hat)
        z_b=self.h_hat_b(h_hat)


        d_h=self.h_z(z_h)
        d_x=self.x_z(z_x)
        bias=self.b_z(z_b)   ######todo: split


        gates = torch.mul(d_x,F.linear(input, self.weight_ih)) + torch.mul(d_h,F.linear(hx, self.weight_hh)) + bias
        gates = self.layernorm(gates)

        ingate, forgetgate, cellgate, outgate_ = gates.chunk(4, 1)

        ingate = F.sigmoid(ingate)
        forgetgate = F.sigmoid(forgetgate)
        cellgate = F.tanh(cellgate)
        outgate = F.sigmoid(outgate_ )

        cy = forgetgate * cx +ingate * cellgate
        hy = outgate * F.tanh(cy)


        return outgate, hy, cy,h_hat_next,c_hat_next

class HyperLSTM(RNNCellBase):
    def __init__(self,input_size, hidden_size,hidden_size_hat,z_size):
        super(HyperLSTM, self).__init__(input_size, hidden_size, bias=True, num_chunks=4)

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.weight_ih =  Parameter(torch.Tensor(4 * hidden_size, input_size))
        self.weight_hh = Parameter(torch.Tensor(4 * hidden_size, hidden_size))

        self.bias_ih = None
        self.bias_hh = None

        self.hyperlstm=LSTMCell(input_size+hidden_size,hidden_size_hat)
        self.h_hat_h=nn.Linear(hidden_size_hat,z_size*4)
        self.h_hat_x = nn.Linear(hidden_size_hat, z_size*4)
        self.h_hat_b = nn.Linear(hidden_size_hat, z_size*4,bias=False)

        self.h_z=nn.Linear(z_size*4,hidden_size*4,bias=False)
        self.x_z=nn.Linear(z_size*4,hidden_size*4,bias=False)
        self.b_z=nn.Linear(z_size*4,hidden_size*4)

        self.layernorm=LayerNorm(4*hidden_size)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self,input,hx,cx,h_hat,c_hat):

        x_hat=torch.concat([input,hx],dim=-1)
        # x_hat=input
        _,h_hat_next,c_hat_next=self.hyperlstm(x_hat,h_hat,c_hat)

        z_h=self.h_hat_h(h_hat)
        z_x=self.h_hat_x(h_hat)
        z_b=self.h_hat_b(h_hat)


        d_h=self.h_z(z_h)
        d_x=self.x_z(z_x)
        bias=self.b_z(z_b)   ######todo: split


        gates = torch.mul(d_x,F.linear(input, self.weight_ih)) + torch.mul(d_h,F.linear(hx, self.weight_hh)) + bias
        gates = self.layernorm(gates)

        ingate, forgetgate, cellgate, outgate_ = gates.chunk(4, 1)

        ingate = F.sigmoid(ingate)
        forgetgate = F.sigmoid(forgetgate)
        cellgate = F.tanh(cellgate)
        outgate = F.sigmoid(outgate_ )

        cy = forgetgate * cx +ingate * cellgate
        hy = outgate * F.tanh(cy)


        return outgate, hy, cy,h_hat_next,c_hat_next




class HyperLSTM_center(RNNCellBase):
    def __init__(self,input_size, hidden_size,hidden_size_hat,z_size):
        super(HyperLSTM, self).__init__(input_size, hidden_size, bias=True, num_chunks=4)

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.weight_ih =  Parameter(torch.Tensor(4 * hidden_size, input_size))
        self.weight_hh = Parameter(torch.Tensor(4 * hidden_size, hidden_size))

        self.bias_ih = None
        self.bias_hh = None

        self.hyperlstm=LSTMCell(input_size+hidden_size,hidden_size_hat)
        self.h_hat_h=nn.Linear(hidden_size_hat,z_size*4)
        self.h_hat_x = nn.Linear(hidden_size_hat, z_size*4)
        self.h_hat_b = nn.Linear(hidden_size_hat, z_size*4,bias=False)

        self.h_z=nn.Linear(z_size*4,hidden_size*4,bias=False)
        self.x_z=nn.Linear(z_size*4,hidden_size*4,bias=False)
        self.b_z=nn.Linear(z_size*4,hidden_size*4)

        self.layernorm=LayerNorm(4*hidden_size)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self,input,center_input,hx,cx,h_hat,c_hat):

        # x_hat=torch.concat([input,hx],dim=-1)
        x_hat=input
        _,h_hat_next,c_hat_next=self.hyperlstm(x_hat,h_hat,c_hat)

        z_h=self.h_hat_h(h_hat)
        z_x=self.h_hat_x(h_hat)
        z_b=self.h_hat_b(h_hat)


        d_h=self.h_z(z_h)
        d_x=self.x_z(z_x)
        bias=self.b_z(z_b)   ######todo: split


        gates = torch.mul(d_x,F.linear(input, self.weight_ih)) + torch.mul(d_h,F.linear(hx, self.weight_hh)) + bias
        gates = self.layernorm(gates)

        ingate, forgetgate, cellgate, outgate_ = gates.chunk(4, 1)

        ingate = F.sigmoid(ingate)
        forgetgate = F.sigmoid(forgetgate)
        cellgate = F.tanh(cellgate)
        outgate = F.sigmoid(outgate_ )

        cy = forgetgate * cx +ingate * cellgate
        hy = outgate * F.tanh(cy)


        return outgate, hy, cy,h_hat_next,c_hat_next






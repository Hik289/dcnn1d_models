import torch
from torch import nn

class Conv1d_GRUcell_v1(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias = True):
        """
        Initialize GRU_LSTM_v1 cell.
        v1: Conv1d + linear
        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        bias: bool
            Whether or not to add the bias.
        """

        super(Conv1d_GRUcell_v1, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.bias = bias

        self.conv = nn.Conv1d(in_channels = self.input_dim + self.hidden_dim,
                              out_channels = 2*self.hidden_dim,
                              kernel_size = self.kernel_size,
                              stride = 1,
                              padding = self.padding,
                              bias = self.bias,
                              padding_mode = 'replicate')

        self.linear = nn.Linear(in_features= self.input_dim+self.hidden_dim, out_features=self.hidden_dim,bias= self.bias)


    def init_hidden(self, batch_size, section_size):
        return (None, torch.zeros(batch_size, self.hidden_dim, section_size, device=self.conv.weight.device))

    def forward(self, x, cur_state):
        c, h = cur_state

        combined1 = torch.cat([x, h], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined1)
        cc_r, cc_z = torch.split(combined_conv, self.hidden_dim, dim=1)
        r = torch.sigmoid(cc_r)
        z = torch.sigmoid(cc_z)

        combined2 = torch.cat([x, r*h], dim=1)  # concatenate along channel axis

        combined2 = combined2.squeeze(-1).squeeze(0)
        if combined2.shape[0] == self.input_dim+self.hidden_dim:
            combined2 = combined2.T
        cc_n = self.linear(combined2)
        n = torch.sigmoid(cc_n).unsqueeze(-1)
        
        if n.shape[0] != h.shape[0]:
            n = n.permute(2,1,0)

        h = (1-z)*n + z*h

        y = n
        if h.shape[0] != x.shape[0]:
            y = y.permute(2,1,0)
            h = h.permute(2,1,0)
        return y, (c, h)


class Conv1d_GRUcell_v2(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size = 1, bias = True):
        """
        Initialize GRU_LSTM_v2 cell.
        v2: Conv1d + Conv1d
        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        bias: bool
            Whether or not to add the bias.
        """

        super(Conv1d_GRUcell_v2, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        self.bias = bias

        self.conv1 = nn.Conv1d(in_channels = self.input_dim + self.hidden_dim,
                              out_channels = 2*self.hidden_dim,
                              kernel_size = self.kernel_size,
                              stride = 1,
                              padding = self.padding,
                              bias = self.bias,
                              padding_mode = 'replicate')

        self.conv2 = nn.Conv1d(in_channels = self.input_dim + self.hidden_dim,
                              out_channels = self.hidden_dim,
                              kernel_size = self.kernel_size,
                              stride = 1,
                              padding = self.padding,
                              bias = self.bias,
                              padding_mode = 'replicate')


    def init_hidden(self, batch_size, section_size):
        return (None, torch.zeros(batch_size, self.hidden_dim, section_size, device=self.conv1.weight.device))

    def forward(self, x, cur_state):
        c, h = cur_state

        combined1 = torch.cat([x, h], dim=1)  # concatenate along channel axis

        combined_conv1 = self.conv1(combined1)
        cc_r, cc_z = torch.split(combined_conv1, self.hidden_dim, dim=1)
        r = torch.sigmoid(cc_r)
        z = torch.sigmoid(cc_z)

        combined2 = torch.cat([x, r*h], dim=1)  # concatenate along channel axis

        cc_n = self.conv2(combined2)
        n = torch.sigmoid(cc_n)

        h = (1-z)*n + z*h

        y = n
        return y, (c, h)


class Conv1d_GRUcell_v3(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias = True):
        """
        Initialize GRU_LSTM_v3 cell.
        v3: linear + Conv1d
        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        bias: bool
            Whether or not to add the bias.
        """

        super(Conv1d_GRUcell_v3, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        self.bias = bias

        self.linear = nn.Linear(in_features= self.input_dim+self.hidden_dim, out_features=2*self.hidden_dim,bias= self.bias)

        self.conv = nn.Conv1d(in_channels = self.input_dim + self.hidden_dim,
                              out_channels = self.hidden_dim,
                              kernel_size = self.kernel_size,
                              stride = 1,
                              padding = self.padding,
                              bias = self.bias,
                              padding_mode = 'replicate')


    def init_hidden(self, batch_size, section_size):
        return (None,torch.zeros(batch_size, self.hidden_dim, section_size, device=self.conv.weight.device))

    def forward(self, x, cur_state):
        c, h = cur_state
        
        if x.squeeze(0).shape[0] == self.input_dim:
            mark = 1
            x = x.squeeze(0).T
            h = h.squeeze(0).T
        else:
            mark = 0
            x = x.squeeze(-1)
            h = h.squeeze(-1)

        combined1 = torch.cat([x, h], dim=1)  # concatenate along channel axis

        combined_linear1 = self.linear(combined1)
        cc_r, cc_z = torch.split(combined_linear1, self.hidden_dim, dim=1)
        r = torch.sigmoid(cc_r)
        z = torch.sigmoid(cc_z)

        combined2 = torch.cat([x, r*h], dim=1)  # concatenate along channel axis

        if mark == 1:
            combined2 = combined2.T.unsqueeze(0)
        else:
            combined2 = combined2.unsqueeze(-1)

        cc_n = self.conv(combined2)
        n = torch.sigmoid(cc_n)

        if mark == 1:
            n = n.squeeze(0).T
        else:
            n = n.squeeze(-1)


        h = (1-z)*n + z*h

        if mark == 1:
            h = h.T.unsqueeze(0)
            y = n.T.unsqueeze(0)
        else:
            h = h.unsqueeze(-1)
            y = n.unsqueeze(-1)

        return y, (c, h)
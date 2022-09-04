import torch
from torch import nn

class PASScell_2(torch.nn.Module):
    """
    A simple PASS cell network
    """
    def __init__(self, input_length=3, hidden_length=20):
        super(PASScell_2, self).__init__()
        self.input_length = input_length
        self.hidden_length = hidden_length

        self.linear_f_u1 = nn.Linear(self.input_length, self.hidden_length, bias=True)
        self.linear_f_h1 = nn.Linear(self.hidden_length, self.hidden_length, bias=False)
        self.linear_f_y1 = nn.Linear(self.hidden_length, self.hidden_length, bias=False)
        self.sigmoid_f = nn.Sigmoid()

        self.linear_i_z2 = nn.Linear(self.input_length+ self.hidden_length, self.hidden_length, bias=True)
        self.sigmoid_z = nn.Sigmoid()


        self.final_activation = nn.ReLU()

    def forward(self, x, o, h):

        z = torch.cat([h, x],axis = 1)

        z_temp = self.linear_i_z2(z)
        o = self.sigmoid_z(z_temp)

        x_temp = self.linear_f_u1(x)
        h_temp = self.linear_f_h1(h)
        o_temp = self.linear_f_y1(o)
        f = self.sigmoid_f(x_temp + h_temp + o_temp)

        return o, f


class PASScell_2_v2(nn.Module):

    def __init__(self, input_dim =3, hidden_dim =20, kernel_size = None, bias = True):
        """
        Initialize PASScell_2 cell.
        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        bias: bool
            Whether or not to add the bias.
        """

        super(PASScell_2_v2, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.bias = bias

        self.linear1 = nn.Linear(in_features= self.input_dim+self.hidden_dim, out_features=self.hidden_dim,bias= self.bias)

        self.linear2 = nn.Linear(in_features= self.input_dim+self.hidden_dim*2, out_features=self.hidden_dim*3,bias= self.bias)

    def init_hidden(self, batch_size, section_size = None):
        return (None, torch.zeros(batch_size, self.hidden_dim, device=self.linear1.weight.device))

    def forward(self, x, cur_state):
        c, h = cur_state

        combined1 = torch.cat([x, h], dim=1)  # concatenate along channel axis

        combined_linear1 = self.linear1(combined1)
        o = torch.sigmoid(combined_linear1)

        combined2 = torch.cat([x,o,h], dim = 1) # concatenate along channel axis
        combined_linear2 = self.linear2(combined2)
        x, o, h= torch.split(combined_linear2, self.hidden_dim, dim=1)
        x = torch.sigmoid(x)
        o = torch.sigmoid(o)
        h = torch.sigmoid(h)

        h = x + o + h

        y = o
        return y, (c, h)


class Conv1d_PASScell_2_v1(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias = True):
        """
        Initialize COnv1d_PASScell_2 cell.
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

        super(Conv1d_PASScell_2_v1, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.bias = bias

        self.conv = nn.Conv1d(in_channels = self.input_dim + self.hidden_dim,
                              out_channels = self.hidden_dim,
                              kernel_size = self.kernel_size,
                              stride = 1,
                              padding = self.padding,
                              bias = self.bias,
                              padding_mode = 'replicate')

        self.linear = nn.Linear(in_features= self.input_dim+2*self.hidden_dim, out_features=3*self.hidden_dim,bias= self.bias)


    def init_hidden(self, batch_size, section_size):
        return (None, torch.zeros(batch_size, self.hidden_dim, section_size, device=self.conv.weight.device))

    def forward(self, x, cur_state):
        c, h = cur_state

        combined1 = torch.cat([x, h], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined1)
        o = torch.sigmoid(combined_conv)

        combined2 = torch.cat([x,o,h], dim = 1) # concatenate along channel axis

        combined2 = combined2.squeeze(-1).squeeze(0)
        if combined2.shape[0] == self.input_dim+2*self.hidden_dim:
            combined2 = combined2.T
        combined_linear = self.linear(combined2)
        m, o, h= torch.split(combined_linear, self.hidden_dim, dim=1)
        m = torch.sigmoid(m)
        o = torch.sigmoid(o)
        h = torch.sigmoid(h)

        h = m + o + h

        y = o

        if y.shape[0] != x.shape[0]:
            y = y.unsqueeze(-1).permute(2,1,0)
            h = h.unsqueeze(-1).permute(2,1,0)
        else:
            y = y.unsqueeze(-1)
            h = h.unsqueeze(-1)
        return y, (c, h)


class Conv1d_PASScell_2_v2(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size = 1, bias = True):
        """
        Initialize Conv1d_PASScell_2 cell.
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

        super(Conv1d_PASScell_2_v2, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        self.bias = bias

        self.conv1 = nn.Conv1d(in_channels = self.input_dim + self.hidden_dim,
                              out_channels = self.hidden_dim,
                              kernel_size = self.kernel_size,
                              stride = 1,
                              padding = self.padding,
                              bias = self.bias,
                              padding_mode = 'replicate')

        self.conv2 = nn.Conv1d(in_channels = self.input_dim + 2*self.hidden_dim,
                              out_channels = 3*self.hidden_dim,
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
        o = torch.sigmoid(combined_conv1)

        combined2 = torch.cat([x, o, h], dim=1)  # concatenate along channel axis

        combined_conv2 = self.conv2(combined2)
        x, o, h= torch.split(combined_conv2, self.hidden_dim, dim=1)
        x = torch.sigmoid(x)
        o = torch.sigmoid(o)
        h = torch.sigmoid(h)

        h = x + o + h

        y = o
        return y, (c, h)



class Conv1d_PASScell_2_v3(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias = True):
        """
        Initialize Conv1d_PASScell_2 cell.
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

        super(Conv1d_PASScell_2_v3, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        self.bias = bias

        self.linear = nn.Linear(in_features= self.input_dim+self.hidden_dim, out_features=self.hidden_dim,bias= self.bias)

        self.conv = nn.Conv1d(in_channels = self.input_dim + 2*self.hidden_dim,
                              out_channels = 3*self.hidden_dim,
                              kernel_size = self.kernel_size,
                              stride = 1,
                              padding = self.padding,
                              bias = self.bias,
                              padding_mode = 'replicate')


    def init_hidden(self, batch_size, section_size):
        return (None, torch.zeros(batch_size, self.hidden_dim, section_size, device=self.conv.weight.device))

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

        combined_linear = self.linear(combined1)
        o = torch.sigmoid(combined_linear)

        combined2 = torch.cat([x, o, h], dim=1)  # concatenate along channel axis

        if mark == 1:
            combined2 = combined2.T.unsqueeze(0)
        else:
            combined2 = combined2.unsqueeze(-1)

        combined_conv = self.conv(combined2)
        x, o, h= torch.split(combined_conv, self.hidden_dim, dim=1)
        x = torch.sigmoid(x)
        o = torch.sigmoid(o)
        h = torch.sigmoid(h)

        h = x + o + h

        y = o
        return y, (c, h)
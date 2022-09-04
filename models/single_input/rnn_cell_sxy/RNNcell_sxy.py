import torch
from torch import nn

class RNNcell(torch.nn.Module):
    """
    A simple RNN cell network
    """
    def __init__(self, input_length=3, hidden_length=20):
        super(RNNcell, self).__init__()
        self.input_length = input_length
        self.hidden_length = hidden_length

        self.linear_rnn_w1 = nn.Linear(self.input_length, self.hidden_length, bias=True)
        self.linear_rnn_r1 = nn.Linear(self.hidden_length, self.hidden_length, bias=True)
        self.sigmoid_rnn = nn.Sigmoid()

        self.activation_final = nn.Tanh()



    def forward(self, x, h):

        x_temp = self.linear_rnn_w1(x)
        h_temp = self.linear_rnn_r1(h)
        h = self.sigmoid_rnn(x_temp + h_temp)

        return h, h


class RNNcell_v2(nn.Module):

    def __init__(self, input_dim =3, hidden_dim =20, kernel_size = None, bias = True):
        """
        Initialize RNN cell.
        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        bias: bool
            Whether or not to add the bias.
        """

        super(RNNcell_v2, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.bias = bias

        self.linear = nn.Linear(in_features= self.input_dim+self.hidden_dim, out_features=self.hidden_dim,bias= self.bias)

    def init_hidden(self, batch_size, section_size = None):
        return (None,torch.zeros(batch_size, self.hidden_dim, device=self.linear.weight.device))

    def forward(self, x, cur_state):
        c, h = cur_state

        combined = torch.cat([x, h], dim=1)  # concatenate along channel axis

        cc_h = self.linear(combined)
        h = torch.sigmoid(cc_h)

        y = h
        return y, (c, h)


class Conv1d_RNNcell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias = True):
        """
        Initialize ConvRNN cell.
        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(Conv1d_RNNcell, self).__init__()

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

    def init_hidden(self, batch_size, section_size):
        return (None, torch.zeros(batch_size, self.hidden_dim, section_size, device=self.conv.weight.device))

    def forward(self, x, cur_state):
        c, h = cur_state

        combined = torch.cat([x, h], dim=1)  # concatenate along channel axis

        cc_h = self.conv(combined)
        h = torch.sigmoid(cc_h)

        y = h

        return y, (c, h)
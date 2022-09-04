import torch
from torch import nn

class Conv1d_LSTMCcell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias = True):
        """
        Initialize ConvLSTMC cell.
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

        super(Conv1d_LSTMCcell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.bias = bias

        self.conv = nn.Conv1d(in_channels = self.input_dim + self.hidden_dim* 2,
                              out_channels = 4 * self.hidden_dim,
                              kernel_size = self.kernel_size,
                              stride = 1,
                              padding = self.padding,
                              bias = self.bias,
                              padding_mode = 'replicate')

    def init_hidden(self, batch_size, section_size):
        return (torch.zeros(batch_size, self.hidden_dim, section_size, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, section_size, device=self.conv.weight.device))

    def forward(self, x, cur_state):
        c, h = cur_state

        combined = torch.cat([x, c, h], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c = f * c + i * g
        h = o * torch.tanh(c)

        y = h

        return y, (c, h)
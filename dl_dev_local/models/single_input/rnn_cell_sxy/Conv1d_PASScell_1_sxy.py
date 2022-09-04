import torch
from torch import nn

class Conv1d_PASScell_1(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias = True):
        """
        Initialize Conv1d_PASScell_1 cell.
        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        bias: bool
            Whether or not to add the bias.
        """

        super(Conv1d_PASScell_1, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.bias = bias

        self.conv = nn.Conv1d(in_channels = self.input_dim + self.hidden_dim,
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
        o, h = cur_state

        combined = torch.cat([x, h], dim=1)  # concatenate along channel axis

        combined_linear = self.conv(combined)
        cc_f, cc_i, cc_k, cc_g = torch.split(combined_linear, self.hidden_dim, dim=1)
        f = torch.sigmoid(cc_f)
        i = torch.sigmoid(cc_i)
        k = torch.sigmoid(cc_k)
        g = torch.sigmoid(cc_g)

        o = f * o - i* torch.relu(g)
        h = k * torch.relu(o)

        y = o

        return y, (o, h)
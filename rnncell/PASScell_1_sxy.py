import torch
from torch import nn

class PASScell_1(torch.nn.Module):
    """
    A simple PASS cell network
    """
    def __init__(self, input_length=3, hidden_length=20):
        super(PASScell_1, self).__init__()
        self.input_length = input_length
        self.hidden_length = hidden_length

        self.linear_f_w1 = nn.Linear(self.input_length, self.hidden_length, bias=True)
        self.linear_f_r1 = nn.Linear(self.hidden_length, self.hidden_length, bias=False)
        self.sigmoid_f = nn.Sigmoid()

        self.linear_i_w2 = nn.Linear(self.input_length, self.hidden_length, bias=True)
        self.linear_i_r2 = nn.Linear(self.hidden_length, self.hidden_length, bias=False)
        self.sigmoid_i = nn.Sigmoid()

        self.linear_k_w3 = nn.Linear(self.input_length, self.hidden_length, bias=True)
        self.linear_k_r3 = nn.Linear(self.hidden_length, self.hidden_length, bias=False)
        self.sigmoid_k = nn.Sigmoid()

        self.linear_g_w4 = nn.Linear(self.input_length, self.hidden_length, bias=True)
        self.linear_g_r4 = nn.Linear(self.hidden_length, self.hidden_length, bias=False)
        self.sigmoid_g = nn.Sigmoid()

        self.final_activation = nn.ReLU()

    def forward(self, x, o, h):

        x_temp = self.linear_f_w1(x)
        h_temp = self.linear_f_r1(h)
        f = self.sigmoid_f(x_temp + h_temp)

        x_temp = self.linear_i_w2(x)
        h_temp = self.linear_i_r2(h)
        i =  self.sigmoid_i(x_temp + h_temp)

        x_temp = self.linear_k_w3(x)
        h_temp = self.linear_k_r3(h)
        k = self.sigmoid_k(x_temp + h_temp)

        x_temp = self.linear_g_w4(x)
        h_temp = self.linear_g_r4(h)
        g = self.sigmoid_g(x_temp + h_temp)

        o = f * o - i* self.final_activation(g)
        h = k * self.final_activation(o)

        return o, h


class PASScell_1_v2(nn.Module):

    def __init__(self, input_dim =3, hidden_dim =20, kernel_size = None, bias = True):
        """
        Initialize PASScell_1 cell.
        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        bias: bool
            Whether or not to add the bias.
        """

        super(PASScell_1_v2, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.bias = bias

        self.linear = nn.Linear(in_features= self.input_dim+self.hidden_dim, out_features=4*self.hidden_dim,bias= self.bias)

    def init_hidden(self, batch_size, section_size = None):
        return (torch.zeros(batch_size, self.hidden_dim, device=self.linear.weight.device),
                torch.zeros(batch_size, self.hidden_dim, device=self.linear.weight.device))

    def forward(self, x, cur_state):
        o, h = cur_state

        combined = torch.cat([x, h], dim=1)  # concatenate along channel axis

        combined_linear = self.linear(combined)
        cc_f, cc_i, cc_k, cc_g = torch.split(combined_linear, self.hidden_dim, dim=1)
        f = torch.sigmoid(cc_f)
        i = torch.sigmoid(cc_i)
        k = torch.sigmoid(cc_k)
        g = torch.sigmoid(cc_g)

        o = f * o - i* torch.relu(g)
        h = k * torch.relu(o)

        y = o
        return y, (o, h)


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

    def init_hidden(self, batch_size, section_size = None):
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


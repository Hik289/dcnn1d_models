import torch
from torch import nn

class LSTMCcell(torch.nn.Module):
    """
    A simple LSTMC cell network
    """
    def __init__(self, input_length=3, hidden_length=20):
        super(LSTMCcell, self).__init__()
        self.input_length = input_length
        self.hidden_length = hidden_length

        self.linear_gate_w1 = nn.Linear(self.input_length, self.hidden_length, bias=True)
        self.linear_gate_r1 = nn.Linear(self.hidden_length, self.hidden_length, bias=False)
        self.linear_gate_c1 = nn.Linear(self.hidden_length, self.hidden_length, bias=False)
        self.sigmoid_gate_1 = nn.Sigmoid()

        self.linear_gate_w2 = nn.Linear(self.input_length, self.hidden_length, bias=True)
        self.linear_gate_r2 = nn.Linear(self.hidden_length, self.hidden_length, bias=False)
        self.linear_gate_c2 = nn.Linear(self.hidden_length, self.hidden_length, bias=False)
        self.sigmoid_gate_2 = nn.Sigmoid()

        self.linear_gate_w3 = nn.Linear(self.input_length, self.hidden_length, bias=True)
        self.linear_gate_r3 = nn.Linear(self.hidden_length, self.hidden_length, bias=False)
        self.linear_gate_c3 = nn.Linear(self.hidden_length, self.hidden_length, bias=False)
        self.sigmoid_gate_3 = nn.Tanh()

        self.linear_gate_w4 = nn.Linear(self.input_length, self.hidden_length, bias=True)
        self.linear_gate_r4 = nn.Linear(self.hidden_length, self.hidden_length, bias=True)
        self.linear_gate_c4 = nn.Linear(self.hidden_length, self.hidden_length, bias=True)
        self.sigmoid_gate_4 = nn.Sigmoid()

        self.activation_final = nn.ReLU()

    def forward(self, x, c, h):

        x_temp = self.linear_gate_w1(x)
        h_temp = self.linear_gate_r1(h)
        c_temp = self.linear_gate_c1(c)
        i = self.sigmoid_gate_1(x_temp + h_temp + c_temp)

        x_temp = self.linear_gate_w2(x)
        h_temp = self.linear_gate_r2(h)
        c_temp = self.linear_gate_c2(c)
        f = self.sigmoid_gate_2(x_temp + h_temp + c_temp)

        x_temp = self.linear_gate_w3(x)
        h_temp = self.linear_gate_r3(h)
        c_temp = self.linear_gate_c3(c)
        k = self.sigmoid_gate_3(x_temp + h_temp + c_temp)
        c = f*c + i* self.activation_final(k)

        x_temp = self.linear_gate_w4(x)
        h_temp = self.linear_gate_r4(h)
        c_temp = self.linear_gate_c4(c)
        o = self.sigmoid_gate_4(x_temp + h_temp + c_temp)

        h = o * self.activation_final(c)

        return o, c, h


class LSTMCcell_v2(nn.Module):

    def __init__(self, input_dim =3, hidden_dim =20, kernel_size = None, bias = True):
        """
        Initialize LSTMC cell.
        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        bias: bool
            Whether or not to add the bias.
        """

        super(LSTMCcell_v2, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.bias = bias

        self.linear = nn.Linear(in_features= self.input_dim+self.hidden_dim*2, out_features=4*self.hidden_dim,bias= self.bias)

    def init_hidden(self, batch_size, section_size = None):
        return (torch.zeros(batch_size, self.hidden_dim, device=self.linear.weight.device),
                torch.zeros(batch_size, self.hidden_dim, device=self.linear.weight.device))

    def forward(self, x, cur_state):
        c, h = cur_state

        combined = torch.cat([x, c, h], dim=1)  # concatenate along channel axis

        combined_linear = self.linear(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_linear, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c = f * c + i * g
        h = o * torch.tanh(c)

        y = o
        return y, (c, h)

    
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


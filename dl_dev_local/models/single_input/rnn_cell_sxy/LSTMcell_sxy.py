import torch
from torch import nn

class LSTMcell(torch.nn.Module):
    """
    A simple LSTM cell network
    """
    def __init__(self, input_length=3, hidden_length=20):
        super(LSTMcell, self).__init__()
        self.input_length = input_length
        self.hidden_length = hidden_length

        # forget gate components
        self.linear_forget_w1 = nn.Linear(self.input_length, self.hidden_length, bias=True)
        self.linear_forget_r1 = nn.Linear(self.hidden_length, self.hidden_length, bias=True)
        self.sigmoid_forget = nn.Sigmoid()

        # input gate components
        self.linear_gate_w2 = nn.Linear(self.input_length, self.hidden_length, bias=True)
        self.linear_gate_r2 = nn.Linear(self.hidden_length, self.hidden_length, bias=True)
        self.sigmoid_gate = nn.Sigmoid()

        # cell memory components
        self.linear_gate_w3 = nn.Linear(self.input_length, self.hidden_length, bias=True)
        self.linear_gate_r3 = nn.Linear(self.hidden_length, self.hidden_length, bias=True)
        self.activation_gate = nn.Tanh()

        # out gate components
        self.linear_gate_w4 = nn.Linear(self.input_length, self.hidden_length, bias=True)
        self.linear_gate_r4 = nn.Linear(self.hidden_length, self.hidden_length, bias=True)
        self.sigmoid_hidden_out = nn.Sigmoid()

        self.activation_final = nn.Tanh()



    def forward(self, x, c, h):

        # Equation 1. input gate
        x_temp = self.linear_gate_w2(x)
        h_temp = self.linear_gate_r2(h)
        i = self.sigmoid_gate(x_temp + h_temp)

        # Equation 2. forget gate
        x_temp = self.linear_forget_w1(x)
        h_temp = self.linear_forget_r1(h)
        f =  self.sigmoid_forget(x_temp + h_temp)

        # Equation 3. updating the cell memory
        x_temp = self.linear_gate_w3(x)
        h_temp = self.linear_gate_r3(h)
        k = self.activation_gate(x_temp + h_temp)
        g = k * i
        c = f * c
        c = g + c

        # Equation 4. calculate the main output gate
        x_temp = self.linear_gate_w4(x)
        h_temp = self.linear_gate_r4(h)
        o =  self.sigmoid_hidden_out(x_temp + h_temp)

        # Equation 5. produce next hidden output
        h = o * self.activation_final(c)

        return o, c, h


class LSTMcell_v2(nn.Module):

    def __init__(self, input_dim =3, hidden_dim =20, kernel_size = None, bias = True):
        """
        Initialize LSTM cell.
        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        bias: bool
            Whether or not to add the bias.
        """

        super(LSTMcell_v2, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.bias = bias

        self.linear = nn.Linear(in_features= self.input_dim+self.hidden_dim, out_features=4*self.hidden_dim,bias= self.bias)

    def init_hidden(self, batch_size, section_size = None):
        return (torch.zeros(batch_size, self.hidden_dim, device=self.linear.weight.device),
                torch.zeros(batch_size, self.hidden_dim, device=self.linear.weight.device))

    def forward(self, x, cur_state):
        c, h = cur_state

        combined = torch.cat([x, h], dim=1)  # concatenate along channel axis

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


class Conv1d_LSTMcell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias = True):
        """
        Initialize ConvLSTM cell.
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

        super(Conv1d_LSTMcell, self).__init__()

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
        c, h = cur_state

        combined = torch.cat([x, h], dim=1)  # concatenate along channel axis

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
from models.BasicModule import BasicModule
import torch.nn as nn
import torch

class cnn_test2(BasicModule):
    def __init__(self,input_dim=482,drop_rate=0.5):
        super(cnn_test2,self).__init__()
        self.model_name = 'cnn_test'
        
        self.cnn = nn.Conv2d(in_channels=input_dim, out_channels=10, kernel_size=1)

        self.end_layer = nn.Linear(10,1,bias=False)

        # for p in self.cnn.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)
        # for p in self.end_layer.parameters():
        #     nn.init.normal_(p,mean=0.0,std = 0.001)

    def forward(self,x):

        x_signal = self.cnn(x.unsqueeze(-1))
        x_signal = torch.mean(x_signal, axis = 2).squeeze(-1)
        x = self.end_layer(x_signal)

        self.model_out['y_pred'] = x
        self.model_out['signals'] = x_signal
        return self.model_out